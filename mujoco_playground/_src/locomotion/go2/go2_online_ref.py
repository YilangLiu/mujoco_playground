"""On-the-fly trot reference generation for Go2SampleAPG via DIAL-MPC sampling.

Runs the DIAL-MPC node-diffusion sampler directly on the playground
Go2SampleAPG model — same physics, same actuation path as training
(ctrl = home_pose + clip(a, -1, 1) * action_scale) — scored by the dial-mpc
trot gait reward ported from Go2Trot. Produces a kinematic (qpos, qvel)
reference at trainer startup, replacing the offline play_go2_sampling.py
stage: the reference is dynamically feasible under the training model by
construction, needs no leg-order conversion (generated in the training
model's own joint order), and is non-looping (no wrap teleport).

The sampler math (node2u/u2node splines, reverse_once MPPI update, shift
warm-start, execute -> shift -> anneal receding-horizon loop with
num_diffuse_init extra anneals at t=0) mirrors DialNodeController in
learning/play_go2_sampling.py, which itself mirrors dial-mpc's MBDPI.
"""

from typing import Tuple

import functools

import jax
import jax.numpy as jp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from ml_collections import config_dict
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_sampling


def default_gen_config() -> config_dict.ConfigDict:
  return config_dict.create(
      # Sampler (offline-quality settings from learning/play_go2_sampling.py).
      num_samples=2048,
      hsample=20,
      hnode=5,
      num_diffuse=2,
      num_diffuse_init=10,
      temperature=0.05,
      horizon_diffuse_factor=0.9,
      traj_diffuse_factor=0.5,
      # Trajectory length: 480 steps = 240-step episodes + 240 RSI slack, so
      # every episode start frame has full reference coverage (no end clamp).
      num_steps=480,
      # Task (dial-mpc unitree_go2_trot.yaml via go2_sampling.Go2Trot).
      gait="trot",
      default_vx=0.8,
      default_vy=0.0,
      ramp_up_time=1.0,
      base_height=0.3,
      reward_scales=config_dict.create(
          gaits=0.1,
          upright=0.5,
          yaw=0.3,
          vel=1.0,
          ang_vel=1.0,
          height=1.0,
      ),
  )


class OnlineReferenceGenerator:
  """DIAL-MPC receding-horizon reference generator on the training model."""

  def __init__(self, env, cfg: config_dict.ConfigDict = None):
    # `env` is a Go2SampleAPG instance; only its model/actuation constants are
    # used (its reference/reward machinery plays no role in generation).
    self.cfg = cfg or default_gen_config()
    cfg = self.cfg

    self._mj_model = env.mj_model
    self._mjx_model = env.mjx_model
    self._n_substeps = env.n_substeps
    self._dt = env.dt
    self._nu = env.mjx_model.nu
    self._action_loc = env.action_loc
    self._action_scale = env.action_scale
    self._init_q = jp.array(env._init_q)
    self._default_pose = jp.array(env._init_q[7:])
    self._torso_body_id = env.base_id
    # Gait phase order matches dial-mpc: (FL, FR, RL, RR).
    self._feet_site_ids = jp.array([
        env.mj_model.site(f"{name}_foot").id
        for name in ("FL", "FR", "RL", "RR")
    ])
    self._gait_phases = go2_sampling._GAIT_PHASES[cfg.gait]
    self._gait_params = go2_sampling._GAIT_PARAMS[cfg.gait]
    self._vel_cmd = jp.array([cfg.default_vx, cfg.default_vy, 0.0])

    self.hsample = int(cfg.hsample)
    self.hnode = int(cfg.hnode)
    self.num_samples = int(cfg.num_samples)
    self._node_spline_order = 2 if self.hnode >= 2 else 1
    self._u_spline_order = 2 if self.hsample >= 2 else 1
    self.sigma_control = (
        cfg.horizon_diffuse_factor
        ** jp.arange(self.hnode + 1, dtype=jp.float32)[::-1]
    )
    self._step_us = jp.linspace(0.0, self._dt * self.hsample, self.hsample + 1)
    self._step_nodes = jp.linspace(0.0, self._dt * self.hsample, self.hnode + 1)

    self._generate_jit = jax.jit(self._generate)

  # ---------------------------------------------------------------- physics

  def _make_initial_data(self):
    # No naconmax/njmax overrides: the jax impl sizes contacts from the
    # model's fixed collision pairs (10 for the collision-free scene), and on
    # the warp impl those knobs are totals across all vmapped worlds — the
    # defaults are correct here, explicit values would be a trap.
    data = mjx_env.make_data(
        self._mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=self._init_q.dtype),
        ctrl=self._default_pose,
        impl=self._mjx_model.impl.value,
    )
    from mujoco import mjx  # local import to avoid polluting module namespace

    return mjx.forward(self._mjx_model, data)

  def _step_physics(self, data, action):
    # Exactly Go2SampleAPG.step's actuation path: the sampler plans in the
    # policy's own action space, so the reference is reachable by training.
    ctrl = self._action_loc + jp.clip(action, -1.0, 1.0) * self._action_scale
    return mjx_env.step(self._mjx_model, data, ctrl, self._n_substeps)

  # ------------------------------------------------------------ task reward

  def _task_reward(self, data, t):
    """Go2Trot._get_reward ported to the playground model (yaw command 0)."""
    cfg = self.cfg
    ramp = jp.minimum(t / cfg.ramp_up_time, 1.0)
    vel_tar = self._vel_cmd * ramp

    duty_ratio, cadence, amplitude = (
        self._gait_params[0], self._gait_params[1], self._gait_params[2]
    )
    z_feet_tar = go2_sampling._get_foot_step(
        duty_ratio, cadence, amplitude, self._gait_phases, t
    )
    z_feet = data.site_xpos[self._feet_site_ids][:, 2]
    gaits = -jp.sum(jp.square((z_feet_tar - z_feet) / 0.05))

    quat = data.xquat[self._torso_body_id]
    up = go2_sampling._Go2DialBase._rotate_inv(jp.array([0.0, 0.0, 1.0]), quat)
    # NOTE: Go2Trot uses math.rotate(z, quat) (body-z in world); the inverse
    # rotation of world-z gives the same "how far from upright" magnitude.
    upright = -jp.sum(jp.square(up - jp.array([0.0, 0.0, 1.0])))

    rpy = go2_sampling._Go2DialBase._roll_pitch_yaw(quat)
    yaw = -jp.square(go2_sampling._Go2DialBase._wrap_to_pi(rpy[2]))

    vb = go2_sampling._Go2DialBase._rotate_inv(data.qvel[:3], quat)
    ab = go2_sampling._Go2DialBase._rotate_inv(data.qvel[3:6], quat)
    vel = -jp.sum(jp.square(vb[:2] - vel_tar[:2]))
    ang_vel = -jp.square(ab[2])

    height = -jp.square(
        data.xpos[self._torso_body_id, 2] - cfg.base_height
    )

    s = cfg.reward_scales
    return (
        s.gaits * gaits
        + s.upright * upright
        + s.yaw * yaw
        + s.vel * vel
        + s.ang_vel * ang_vel
        + s.height * height
    )

  # ---------------------------------------------------------------- sampler

  def _node2u(self, nodes):
    spline = InterpolatedUnivariateSpline(
        self._step_nodes, nodes, k=self._node_spline_order
    )
    return spline(self._step_us)

  def _u2node(self, us):
    spline = InterpolatedUnivariateSpline(
        self._step_us, us, k=self._u_spline_order
    )
    return spline(self._step_nodes)

  def _rollout_score(self, data, us, t0):
    def step_fn(carry, xs):
      data = carry
      u, k = xs
      data = self._step_physics(data, u)
      r = self._task_reward(data, t0 + (k + 1.0) * self._dt)
      return data, r

    # node2u yields hsample+1 control points (spline endpoints inclusive);
    # roll out all of them, matching the offline rollout_us.
    _, rews = jax.lax.scan(
        step_fn, data, (us, jp.arange(self.hsample + 1, dtype=jp.float32))
    )
    return rews.mean()

  def _reverse_once(self, data, t0, rng, ybar, noise_scale):
    rng, sample_rng = jax.random.split(rng)
    eps = jax.random.normal(
        sample_rng, (self.num_samples, self.hnode + 1, self._nu),
        dtype=ybar.dtype,
    )
    y0s = eps * noise_scale[None, :, None] + ybar
    y0s = y0s.at[:, 0].set(ybar[0])
    y0s = jp.concatenate([y0s, ybar[None]], axis=0)
    y0s = jp.clip(y0s, -1.0, 1.0)

    node2u_vmap = jax.vmap(self._node2u, in_axes=1, out_axes=1)
    us = jax.vmap(node2u_vmap)(y0s)
    scores = jax.vmap(self._rollout_score, in_axes=(None, 0, None))(
        data, us, t0
    )
    nominal_score = scores[-1]
    score_std = jp.maximum(scores.std(), 1e-6)
    logp = (scores - nominal_score) / score_std / self.cfg.temperature
    weights = jax.nn.softmax(logp)
    ybar = jp.einsum("n,nij->ij", weights, y0s)
    return rng, ybar

  def _shift(self, ybar):
    node2u_vmap = jax.vmap(self._node2u, in_axes=1, out_axes=1)
    u2node_vmap = jax.vmap(self._u2node, in_axes=1, out_axes=1)
    us = node2u_vmap(ybar)
    us = jp.roll(us, -1, axis=0)
    us = us.at[-1].set(jp.zeros(self._nu, dtype=ybar.dtype))
    return u2node_vmap(us)

  def _anneal(self, data, t0, rng, ybar, n_diffuse: int):
    # factors[i] = sigma_control * traj_diffuse_factor**i, as in the offline
    # loop (play_go2_sampling.py:255-258).
    factors = self.sigma_control[None, :] * (
        self.cfg.traj_diffuse_factor
        ** jp.arange(n_diffuse, dtype=jp.float32)[:, None]
    )

    def body(carry, factor):
      rng, ybar = carry
      rng, ybar = self._reverse_once(data, t0, rng, ybar, factor)
      return (rng, ybar), None

    (rng, ybar), _ = jax.lax.scan(body, (rng, ybar), factors)
    return rng, ybar

  # -------------------------------------------------------------- main loop

  def _generate(self, rng):
    """Mirrors the offline receding-horizon loop: execute ybar[0], step,
    shift, anneal (num_diffuse_init anneals at t=0, num_diffuse after)."""
    data = self._make_initial_data()
    ybar = jp.zeros((self.hnode + 1, self._nu), dtype=data.qpos.dtype)
    qpos0, qvel0 = data.qpos, data.qvel

    # t = 0 iteration (heavier init annealing) outside the scan.
    data = self._step_physics(data, ybar[0])
    ybar = self._shift(ybar)
    rng, ybar = self._anneal(
        data, 1.0 * self._dt, rng, ybar, int(self.cfg.num_diffuse_init)
    )
    frame0 = (data.qpos, data.qvel, self._task_reward(data, 1.0 * self._dt))

    def ctrl_step(carry, step_idx):
      data, ybar, rng = carry
      data = self._step_physics(data, ybar[0])
      ybar = self._shift(ybar)
      t_now = (step_idx + 1.0) * self._dt
      rng, ybar = self._anneal(data, t_now, rng, ybar, int(self.cfg.num_diffuse))
      r = self._task_reward(data, t_now)
      return (data, ybar, rng), (data.qpos, data.qvel, r)

    (_, _, rng), (qpos_seq, qvel_seq, rews) = jax.lax.scan(
        ctrl_step,
        (data, ybar, rng),
        jp.arange(1, self.cfg.num_steps, dtype=jp.float32),
    )

    qpos = jp.concatenate(
        [qpos0[None], frame0[0][None], qpos_seq], axis=0
    )
    qvel = jp.concatenate(
        [qvel0[None], frame0[1][None], qvel_seq], axis=0
    )
    rews = jp.concatenate([frame0[2][None], rews], axis=0)
    return qpos, qvel, rews

  def generate(self, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos, qvel, rews = self._generate_jit(rng)
    return np.asarray(qpos), np.asarray(qvel), np.asarray(rews)


def validate_reference(qpos: np.ndarray, qvel: np.ndarray, dt: float) -> dict:
  """Post-hoc sanity gates on a generated reference; raises on failure."""
  base_z = qpos[:, 2]
  # Roll/pitch from the base quaternion.
  w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
  duration = (qpos.shape[0] - 1) * dt
  stats = {
      "frames": int(qpos.shape[0]),
      "duration_s": float(duration),
      "mean_vx": float((qpos[-1, 0] - qpos[0, 0]) / duration),
      "net_dy": float(qpos[-1, 1] - qpos[0, 1]),
      "base_z_min": float(base_z.min()),
      "base_z_max": float(base_z.max()),
      "max_abs_roll": float(np.abs(roll).max()),
      "max_abs_pitch": float(np.abs(pitch).max()),
      "max_joint_vel": float(np.abs(qvel[:, 6:]).max()),
  }
  problems = []
  if stats["base_z_min"] < 0.15:
    problems.append(f"base too low ({stats['base_z_min']:.3f} m) — likely fell")
  if stats["max_abs_roll"] > 0.7 or stats["max_abs_pitch"] > 0.7:
    problems.append("base rolled/pitched > 0.7 rad — likely unstable")
  if not 0.2 < stats["mean_vx"] < 1.2:
    problems.append(
        f"mean forward velocity {stats['mean_vx']:.2f} m/s outside [0.2, 1.2]"
    )
  if problems:
    raise RuntimeError(
        "Generated reference failed validation: "
        + "; ".join(problems)
        + f" (stats: {stats})"
    )
  return stats
