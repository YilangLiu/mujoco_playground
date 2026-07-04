"""Sampling-friendly tasks for Unitree Go2 (sequential jumping and trotting)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

import mujoco

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def _dial_mpc_model_path(model_name: str) -> str:
  try:
    from dial_mpc.utils.io_utils import get_model_path
  except ModuleNotFoundError:
    import sys

    dial_mpc_root = (mjx_env.EXTERNAL_DEPS_PATH / "dial-mpc").as_posix()
    if dial_mpc_root not in sys.path:
      sys.path.insert(0, dial_mpc_root)
    from dial_mpc.utils.io_utils import get_model_path

  return get_model_path("unitree_go2", model_name).as_posix()


def _get_foot_step(
    duty_ratio: float,
    cadence: float,
    amplitude: float,
    phases: jax.Array,
    time: jax.Array,
) -> jax.Array:
  """Vectorized foot-step height profile (mirrors dial-mpc's get_foot_step)."""

  def step_height(t, foot_phase, dr):
    angle = (t + jp.pi - foot_phase) % (2 * jp.pi) - jp.pi
    angle = jp.where(dr < 1, angle * 0.5 / (1 - dr), angle)
    clipped = jp.clip(angle, -jp.pi / 2, jp.pi / 2)
    value = jp.where(dr < 1, jp.cos(clipped), 0.0)
    return jp.where(jp.abs(value) >= 1e-6, jp.abs(value), 0.0)

  t = time * 2 * jp.pi * cadence + jp.pi
  return amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
      t, 2 * jp.pi * phases, duty_ratio
  )


def default_config() -> config_dict.ConfigDict:
  """Default config for the sequential jumping task (kept for backwards compat)."""
  return default_seq_jump_config()


def default_seq_jump_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.02,
      episode_length=400,
      Kp=30.0,
      Kd=0.0,
      action_repeat=1,
      action_scale=1.0,
      soft_joint_pos_limit_factor=1.0,
      impl="jax",
      naconmax=4 * 8192,
      naccdmax=8192,
      njmax=128,
      jump_dt=1.0,
      pose_target_sequence=[
          [0.0, 0.0, 0.27],
          [0.4, 0.0, 0.27],
          [0.8, 0.0, 0.27],
          [1.2, 0.0, 0.27],
          [1.6, 0.0, 0.27],
      ],
      yaw_target_sequence=[0.0, 0.0, 0.0, 0.0, 0.0],
      foot_place_radius=0.1,
      randomize_tasks=False,
      reward_config=config_dict.create(
          scales=config_dict.create(
              alive=10.0,
              position=1.0,
              upright=1.0,
              yaw=0.3,
              contact=0.1,
              bad_contact=-0.1,
              energy=0.0,
              ctrl_rate=0.0,
          ),
      ),
  )


def default_crate_climb_config() -> config_dict.ConfigDict:
  # Mirrors dial-mpc's unitree_go2_crate_climb.yaml + UnitreeGo2CrateEnv:
  # torque leg control (kp=30, kd=0 riding on XML dof_damping=0.65),
  # action_scale=1.0, head-position target on top of the crate. The crate in
  # mjx_scene_force_crate.xml spans x [0.99, 1.61], y [-0.46, 0.46],
  # z [0, 0.6] (box half-extents 0.31 0.46 0.3 at pos 1.3 0 0.3).
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.02,
      episode_length=200,
      Kp=30.0,
      Kd=0.0,
      action_repeat=1,
      action_scale=1.0,
      soft_joint_pos_limit_factor=1.0,
      impl="jax",
      naconmax=4 * 8192,
      naccdmax=8192,
      njmax=128,
      # Head (not torso) position target: dial-mpc drives
      # torso_pos + R @ head_offset toward pos_tar (env line ~712).
      pos_tar=[1.45, 0.0, 0.87],
      yaw_tar=0.0,
      head_offset=[0.285, 0.0, 0.0],
      # Foot-on-crate-top bonus region (dial-mpc's hardcoded contact window:
      # x (1.0, 1.6), y (-0.45, 0.45), z (0.59, 0.61) on CONTACT points).
      # This port tests foot SITE centers, which sit one foot radius
      # (~0.022 m) above the surface, so the z window is shifted up.
      crate_top_x=[1.0, 1.6],
      crate_top_y=[-0.45, 0.45],
      crate_top_z=[0.59, 0.645],
      randomize_tasks=False,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # dial-mpc weights: pos 1.0, upright 0.01, yaw 0.3,
              # contact 0.02; gaits/vel/height/energy/pitch/roll all 0.
              position=1.0,
              upright=0.01,
              yaw=0.3,
              contact=0.02,
              energy=0.0,
              alive=0.0,
          ),
      ),
  )


def default_trot_config() -> config_dict.ConfigDict:
  # Mirrors dial-mpc's unitree_go2_trot.yaml: kp=30, kd=0 (with XML
  # dof_damping=0.65), default_vx=0.8, ramp_up_time=1.0.
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.02,
      episode_length=1000,
      Kp=30.0,
      Kd=0.65,
      action_repeat=1,
      action_scale=1.0,
      soft_joint_pos_limit_factor=1.0,
      impl="jax",
      naconmax=4 * 8192,
      naccdmax=8192,
      njmax=128,
      gait="trot",
      default_vx=0.8,
      default_vy=0.0,
      default_vyaw=0.0,
      ramp_up_time=1.0,
      base_height=0.3,
      randomize_tasks=False,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gaits=0.1,
              upright=0.5,
              yaw=0.3,
              vel=1.0,
              ang_vel=1.0,
              height=1.0,
              energy=0.0,
              alive=0.0,
          ),
      ),
  )


class _Go2DialBase(go2_base.Go2Env):
  """Shared helpers for dial-mpc-style Go2 envs."""

  def _local_linvel(self, data: mjx.Data) -> jax.Array:
    return self._rotate_inv(data.qvel[:3], data.xquat[self._torso_body_id])

  def _local_angvel(self, data: mjx.Data) -> jax.Array:
    return self._rotate_inv(data.qvel[3:6], data.xquat[self._torso_body_id])

  def _upvector(self, data: mjx.Data) -> jax.Array:
    return math.rotate(jp.array([0.0, 0.0, 1.0]), data.xquat[self._torso_body_id])

  @staticmethod
  def _rotate_inv(vec: jax.Array, quat: jax.Array) -> jax.Array:
    quat_inv = quat.at[1:].multiply(-1.0)
    return math.rotate(vec, quat_inv)

  @staticmethod
  def _roll_pitch_yaw(quat: jax.Array) -> jax.Array:
    w, x, y, z = quat
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jp.array([roll, pitch, yaw])

  @staticmethod
  def _wrap_to_pi(angle: jax.Array) -> jax.Array:
    return jp.arctan2(jp.sin(angle), jp.cos(angle))


class Go2SeqJump(_Go2DialBase):
  """Go2 sequential jumping environment based on Dial-MPC's SeqJump task."""

  def __init__(
      self,
      task: str | None = None,
      config: config_dict.ConfigDict = default_seq_jump_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    del task
    super().__init__(
        xml_path=_dial_mpc_model_path("mjx_scene_force.xml"),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # The shared Go2Env init bakes a position-actuator PD into the model
    # (gainprm/biasprm) and overrides dof_damping with config.Kd. With the
    # force XML used for sampling MPC the actuators are raw motors and the
    # joint damping comes from the XML, so undo those mutations and rebuild
    # the mjx model.
    self._mj_model.actuator_gainprm[:, 0] = 1.0
    self._mj_model.actuator_biasprm[:, 1] = 0.0
    self._mj_model.dof_damping[6:] = 0.65
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._physical_lowers, self._physical_uppers = self.mj_model.jnt_range[1:].T
    self._joint_lowers = jp.maximum(
        jp.array(self._physical_lowers),
        jp.array([
            -0.5, 0.4, -2.3,
            -0.5, 0.4, -2.3,
            -0.5, 0.4, -2.3,
            -0.5, 0.4, -2.3,
        ]),
    )
    self._joint_uppers = jp.minimum(
        jp.array(self._physical_uppers),
        jp.array([
            0.5, 2.0, -1.3,
            0.5, 2.0, -1.3,
            0.5, 1.4, -1.3,
            0.5, 1.4, -1.3,
        ]),
    )

    ctrlrange = np.asarray(self._mj_model.actuator_ctrlrange)
    ctrllimited = np.asarray(self._mj_model.actuator_ctrllimited).astype(bool)
    lowers = np.where(ctrllimited, ctrlrange[:, 0], -np.inf)
    uppers = np.where(ctrllimited, ctrlrange[:, 1], np.inf)
    self._torque_lowers = jp.array(lowers)
    self._torque_uppers = jp.array(uppers)

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_ids = jp.array([
        self._mj_model.site(f"{name}_foot").id for name in ("FR", "FL", "RR", "RL")
    ])
    self._feet_geom_ids = jp.array([self._mj_model.geom(name).id for name in ("FR", "FL", "RR", "RL")])

    pose_sequence = jp.array(self._config.pose_target_sequence)
    yaw_sequence = jp.array(self._config.yaw_target_sequence)
    (
        self._contact_targets,
        self._contact_target_radius,
        self._pose_target_sequence,
        self._yaw_target_sequence,
    ) = self.generate_jumping_sequence(
        pose_sequence, yaw_sequence, self._config.foot_place_radius
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    contact_targets = self._contact_targets
    contact_target_radius = self._contact_target_radius
    pose_sequence = self._pose_target_sequence
    yaw_sequence = self._yaw_target_sequence
    if self._config.randomize_tasks:
      rng, task_rng = jax.random.split(rng)
      contact_targets, contact_target_radius, pose_sequence, yaw_sequence = self.sample_command(task_rng)

    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    ctrl = jp.zeros(self.mjx_model.nu)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "rng": rng,
        "step": jp.zeros((), dtype=jp.int32),
        "contact_stage": jp.zeros((), dtype=jp.int32),
        "contact_targets": contact_targets,
        "contact_target_radius": contact_target_radius,
        "pose_target_sequence": pose_sequence,
        "yaw_target_sequence": yaw_sequence,
        "vel_tar": jp.zeros(3),
        "ang_vel_tar": jp.zeros(3),
        "last_ctrl": jp.zeros(self.mjx_model.nu),
    }
    metrics = {}
    for name in self._config.reward_config.scales.keys():
      metrics[f"reward/{name}"] = jp.zeros(())
    metrics["contact_stage"] = jp.zeros(())
    metrics["position_error"] = jp.zeros(())
    metrics["success"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    ctrl = self.act2tau(action, state.data)
    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

    next_step = state.info["step"] + 1
    contact_stage = self._stage_from_step(next_step, state.info["pose_target_sequence"].shape[0])

    info = dict(state.info)
    info["step"] = next_step
    info["contact_stage"] = contact_stage
    info["last_ctrl"] = ctrl

    done = self._get_termination(data, next_step).astype(jp.float32)
    obs = self._get_obs(data, info)
    rewards = self._get_reward(data, ctrl, state.info, info)
    scaled_rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(scaled_rewards.values())

    metrics = dict(state.metrics)
    for k, v in scaled_rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["contact_stage"] = contact_stage.astype(jp.float32)
    metrics["position_error"] = jp.linalg.norm(
        data.xpos[self._torso_body_id] - info["pose_target_sequence"][contact_stage]
    )
    metrics["success"] = self._success(data, info).astype(jp.float32)

    return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

  def act2joint(self, action: jax.Array) -> jax.Array:
    action_normalized = (action * self._config.action_scale + 1.0) / 2.0
    joint_targets = self._joint_lowers + action_normalized * (
        self._joint_uppers - self._joint_lowers
    )
    return jp.clip(joint_targets, self._physical_lowers, self._physical_uppers)

  def act2tau(self, action: jax.Array, data: mjx.Data) -> jax.Array:
    joint_target = self.act2joint(action)
    n = joint_target.shape[0]
    q = data.qpos[7 : 7 + n]
    qd = data.qvel[6 : 6 + n]
    tau = self._config.Kp * (joint_target - q) - self._config.Kd * qd
    return jp.clip(tau, self._torque_lowers, self._torque_uppers)

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    stage = info["contact_stage"]
    pose_target = info["pose_target_sequence"][stage]
    yaw_target = info["yaw_target_sequence"][stage]
    rpy = self._roll_pitch_yaw(data.xquat[self._torso_body_id])
    yaw_error = self._wrap_to_pi(rpy[2] - yaw_target)

    return jp.concatenate([
        info["vel_tar"],
        info["ang_vel_tar"],
        info["last_ctrl"],
        data.xpos[self._torso_body_id] - pose_target,
        rpy[:2],
        yaw_error.reshape(1),
        data.qpos[7:],
        self._local_linvel(data),
        self._local_angvel(data),
        data.qvel[6:],
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      ctrl: jax.Array,
      old_info: Dict[str, Any],
      info: Dict[str, Any],
  ) -> Dict[str, jax.Array]:
    stage = info["contact_stage"]
    pose_target = info["pose_target_sequence"][stage]
    yaw_target = info["yaw_target_sequence"][stage]
    rpy = self._roll_pitch_yaw(data.xquat[self._torso_body_id])

    position_reward = -jp.sum(jp.square(data.xpos[self._torso_body_id] - pose_target))
    upright_reward = -jp.sum(jp.square(self._upvector(data) - jp.array([0.0, 0.0, 1.0])))
    yaw_reward = -jp.square(rpy[2] - yaw_target)
    contact_reward, bad_contact = self._contact_rewards(data, info)
    energy_reward = -jp.sum(
        jp.maximum(ctrl * data.qvel[6:] / 160.0, 0.0) ** 2
    )
    ctrl_rate_reward = -jp.sum(jp.square(ctrl - old_info["last_ctrl"]))

    return {
        "alive": jp.ones(()),
        "position": position_reward,
        "upright": upright_reward,
        "yaw": yaw_reward,
        "contact": contact_reward,
        "bad_contact": bad_contact,
        "energy": energy_reward,
        "ctrl_rate": ctrl_rate_reward,
    }

  def _get_termination(self, data: mjx.Data, step: jax.Array) -> jax.Array:
    joint_angles = data.qpos[7:]
    return (
        (self._upvector(data)[-1] < 0.0)
        | jp.any(joint_angles < self._joint_lowers)
        | jp.any(joint_angles > self._joint_uppers)
        | (data.xpos[self._torso_body_id, 2] < 0.1)
        | (step >= self._config.episode_length)
    )

  def _stage_from_step(self, step: jax.Array, n_stages: int) -> jax.Array:
    stage = jp.floor(step * self.dt / self._config.jump_dt).astype(jp.int32)
    return jp.minimum(stage, n_stages - 1)

  def _contact_rewards(
      self, data: mjx.Data, info: Dict[str, Any]
  ) -> tuple[jax.Array, jax.Array]:
    contact_dist = data.contact.dist
    contact_pos = data.contact.pos
    contact_stage = info["contact_stage"]
    n_stages = info["pose_target_sequence"].shape[0]

    contact_reward = jp.zeros(())
    penalty_contact = contact_dist <= 0.001
    for foot_i in range(4):
      for stage_j in range(n_stages):
        target = info["contact_targets"][stage_j, foot_i, :2]
        radius = info["contact_target_radius"][stage_j, foot_i]
        cond = (
            jp.sum(jp.square(contact_pos[foot_i, :2] - target))
            <= radius * radius
        )
        weight = (stage_j == contact_stage) * jp.clip(
            -contact_dist[foot_i] + 1.0, 0.0, 1.0
        )
        contact_reward = contact_reward + jp.where(cond, weight, 0.0)
        penalty_contact = penalty_contact.at[foot_i].set(
            penalty_contact[foot_i] & (~cond)
        )
    return contact_reward, jp.sum(penalty_contact).astype(jp.float32)

  def _success(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    final_stage = info["pose_target_sequence"].shape[0] - 1
    at_final_stage = info["contact_stage"] == final_stage
    close = jp.linalg.norm(data.xpos[self._torso_body_id] - info["pose_target_sequence"][-1]) < 0.2
    return at_final_stage & close

  @staticmethod
  def generate_jumping_sequence(
      com_pos: jax.Array, com_heading: jax.Array, foot_place_radius: float
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_steps = com_pos.shape[0]
    contact_target_radius = jp.full((n_steps, 4), foot_place_radius)
    offsets = jp.array([
        [0.2, -0.135, 0.0],  # FR
        [0.2, 0.135, 0.0],  # FL
        [-0.2, -0.135, 0.0],  # RR
        [-0.2, 0.135, 0.0],  # RL
    ])

    def make_targets(carry):
      pos, yaw = carry
      c = jp.cos(yaw)
      s = jp.sin(yaw)
      rot = jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
      return pos + offsets @ rot.T

    contact_targets = jax.vmap(make_targets)((com_pos, com_heading))
    return contact_targets, contact_target_radius, com_pos, com_heading

  def sample_command(
      self, rng: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_steps = 10
    com_pos_begin = jp.array([0.0, 0.0, 0.27])
    com_yaw_begin = jp.array(0.0)

    def randomize_pos(last_pos, key):
      next_pos = last_pos.at[:2].add(
          jax.random.uniform(key, (2,), minval=-0.65, maxval=0.65)
      )
      return next_pos, next_pos

    def randomize_yaw(last_yaw, key):
      next_yaw = last_yaw + jax.random.uniform(key, (), minval=-0.5, maxval=0.5)
      return next_yaw, next_yaw

    keys = jax.random.split(rng, n_steps * 2)
    _, com_pos = jax.lax.scan(randomize_pos, com_pos_begin, keys[:n_steps])
    _, com_yaw = jax.lax.scan(randomize_yaw, com_yaw_begin, keys[n_steps:])
    com_pos = jp.concatenate([com_pos_begin.reshape(1, 3), com_pos], axis=0)
    com_yaw = jp.concatenate([com_yaw_begin.reshape(1), com_yaw], axis=0)
    return self.generate_jumping_sequence(com_pos, com_yaw, self._config.foot_place_radius)

  def render(
      self,
      trajectory: Sequence[mjx_env.State],
      height: int = 480,
      width: int = 640,
      camera: Optional[str] = "track",
  ) -> Sequence[np.ndarray]:
    return super().render(list(trajectory), height=height, width=width, camera=camera)


# Backwards-compatible aliases.
Go2Sampling = Go2SeqJump
Go2SeqJumpSampling = Go2SeqJump


class Go2CrateClimb(_Go2DialBase):
  """Port of dial-mpc's UnitreeGo2CrateEnv: torque-controlled crate climbing.

  The robot starts at the origin and must climb onto a 0.6 m crate whose top
  spans x [0.99, 1.61], y [-0.46, 0.46] (mjx_scene_force_crate.xml). Reward
  drives the HEAD point (torso + R @ [0.285, 0, 0]) to pos_tar
  [1.45, 0, 0.87] with upright/yaw shaping and a small bonus per foot placed
  on the crate top.

  Faithful to dial_mpc/envs/unitree_go2_env.py::UnitreeGo2CrateEnv except:
    * the foot-on-crate bonus tests foot SITE centers inside the crate-top
      window instead of indexing data.contact rows by the brax pipeline's
      hardcoded contact order (indices 16-19 there) — contact row order is
      not stable across MJX impls, the geometric test is (window widened by
      one foot radius in z, see default_crate_climb_config);
    * dial-mpc never sets done on this task; here done only fires at
      episode_length so rollout loops have a bound.
  """

  def __init__(
      self,
      task: str | None = None,
      config: config_dict.ConfigDict = default_crate_climb_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    del task
    super().__init__(
        xml_path=_dial_mpc_model_path("mjx_scene_force_crate.xml"),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # Same PD-baking undo as Go2SeqJump: the shared Go2Env init assumes
    # position actuators, but the force XML has raw motors and carries the
    # joint damping itself.
    self._mj_model.actuator_gainprm[:, 0] = 1.0
    self._mj_model.actuator_biasprm[:, 1] = 0.0
    self._mj_model.dof_damping[6:] = 0.65
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._physical_lowers, self._physical_uppers = self.mj_model.jnt_range[1:].T
    # Climbing-specific action ranges (UnitreeGo2CrateEnv.__init__), dial-mpc
    # leg order FR, FL, RR, RL: rear thighs get [0, 1.8] for the push-up.
    self._joint_lowers = jp.maximum(
        jp.array(self._physical_lowers),
        jp.array([
            -0.25, -1.0, -2.7,
            -0.25, -1.0, -2.7,
            -0.25, 0.0, -2.7,
            -0.25, 0.0, -2.7,
        ]),
    )
    self._joint_uppers = jp.minimum(
        jp.array(self._physical_uppers),
        jp.array([
            0.25, 1.4, -1.0,
            0.25, 1.4, -1.0,
            0.25, 1.8, -1.0,
            0.25, 1.8, -1.0,
        ]),
    )

    ctrlrange = np.asarray(self._mj_model.actuator_ctrlrange)
    ctrllimited = np.asarray(self._mj_model.actuator_ctrllimited).astype(bool)
    self._torque_lowers = jp.array(
        np.where(ctrllimited, ctrlrange[:, 0], -np.inf)
    )
    self._torque_uppers = jp.array(
        np.where(ctrllimited, ctrlrange[:, 1], np.inf)
    )

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_ids = jp.array([
        self._mj_model.site(f"{name}_foot").id
        for name in ("FR", "FL", "RR", "RL")
    ])

    self._pos_tar = jp.array(self._config.pos_tar)
    self._head_offset = jp.array(self._config.head_offset)
    self._crate_lo = jp.array([
        self._config.crate_top_x[0],
        self._config.crate_top_y[0],
        self._config.crate_top_z[0],
    ])
    self._crate_hi = jp.array([
        self._config.crate_top_x[1],
        self._config.crate_top_y[1],
        self._config.crate_top_z[1],
    ])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        ctrl=jp.zeros(self.mjx_model.nu),
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "rng": rng,
        "step": jp.zeros((), dtype=jp.int32),
        "pos_tar": self._pos_tar,
        "vel_tar": jp.zeros(3),
        "ang_vel_tar": jp.zeros(3),
        "yaw_tar": jp.zeros(()),
        "last_ctrl": jp.zeros(self.mjx_model.nu),
    }
    metrics = {}
    for name in self._config.reward_config.scales.keys():
      metrics[f"reward/{name}"] = jp.zeros(())
    metrics["head_position_error"] = jp.zeros(())
    metrics["feet_on_crate"] = jp.zeros(())
    metrics["success"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    ctrl = self.act2tau(action, state.data)
    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

    info = dict(state.info)
    info["step"] = state.info["step"] + 1
    info["last_ctrl"] = ctrl

    obs = self._get_obs(data, info)
    rewards = self._get_reward(data, ctrl, state.info, info)
    scaled_rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(scaled_rewards.values())

    head_pos = self._head_pos(data)
    feet_on = jp.sum(self._feet_on_crate(data).astype(jp.float32))
    metrics = dict(state.metrics)
    for k, v in scaled_rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["head_position_error"] = jp.linalg.norm(head_pos - info["pos_tar"])
    metrics["feet_on_crate"] = feet_on
    metrics["success"] = (
        (metrics["head_position_error"] < 0.2) & (feet_on >= 3.0)
    ).astype(jp.float32)

    done = (info["step"] >= self._config.episode_length).astype(jp.float32)
    return state.replace(
        data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
    )

  def act2joint(self, action: jax.Array) -> jax.Array:
    action_normalized = (action * self._config.action_scale + 1.0) / 2.0
    joint_targets = self._joint_lowers + action_normalized * (
        self._joint_uppers - self._joint_lowers
    )
    return jp.clip(joint_targets, self._physical_lowers, self._physical_uppers)

  def act2tau(self, action: jax.Array, data: mjx.Data) -> jax.Array:
    joint_target = self.act2joint(action)
    n = joint_target.shape[0]
    q = data.qpos[7 : 7 + n]
    qd = data.qvel[6 : 6 + n]
    tau = self._config.Kp * (joint_target - q) - self._config.Kd * qd
    return jp.clip(tau, self._torque_lowers, self._torque_uppers)

  def _head_pos(self, data: mjx.Data) -> jax.Array:
    torso_pos = data.xpos[self._torso_body_id]
    return torso_pos + math.rotate(
        self._head_offset, data.xquat[self._torso_body_id]
    )

  def _feet_on_crate(self, data: mjx.Data) -> jax.Array:
    feet = data.site_xpos[self._feet_site_ids]
    return jp.all((feet > self._crate_lo) & (feet < self._crate_hi), axis=-1)

  def _get_reward(
      self,
      data: mjx.Data,
      ctrl: jax.Array,
      old_info: Dict[str, Any],
      info: Dict[str, Any],
  ) -> Dict[str, jax.Array]:
    del old_info
    # dial-mpc integrates the (zero) vel_tar into the target each step.
    pos_tar = info["pos_tar"] + info["vel_tar"] * self.dt * info["step"]
    position_reward = -jp.sum(jp.square(self._head_pos(data) - pos_tar))
    upright_reward = -jp.sum(
        jp.square(self._upvector(data) - jp.array([0.0, 0.0, 1.0]))
    )
    rpy = self._roll_pitch_yaw(data.xquat[self._torso_body_id])
    # dial-mpc uses the raw (unwrapped) yaw difference here; keep it.
    yaw_reward = -jp.square(rpy[2] - info["yaw_tar"])
    contact_reward = jp.sum(self._feet_on_crate(data).astype(jp.float32))
    energy_reward = -jp.sum(
        jp.maximum(ctrl * data.qvel[6:] / 160.0, 0.0) ** 2
    )
    return {
        "position": position_reward,
        "upright": upright_reward,
        "yaw": yaw_reward,
        "contact": contact_reward,
        "energy": energy_reward,
        "alive": jp.ones(()),
    }

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    rpy = self._roll_pitch_yaw(data.xquat[self._torso_body_id])
    return jp.concatenate([
        info["vel_tar"],
        info["ang_vel_tar"],
        info["last_ctrl"],
        self._head_pos(data) - info["pos_tar"],
        rpy,
        data.qpos[7:],
        self._local_linvel(data),
        self._local_angvel(data),
        data.qvel[6:],
    ])

  def render(
      self,
      trajectory: Sequence[mjx_env.State],
      height: int = 480,
      width: int = 640,
      camera: Optional[str] = "track",
  ) -> Sequence[np.ndarray]:
    return super().render(
        list(trajectory), height=height, width=width, camera=camera
    )


def default_crate_climb_pg_config() -> config_dict.ConfigDict:
  # Crate climbing sampled ON THE TRAINING PLANT: playground crate scene
  # (accurate 2 ms substeps, position-PD actuators with Kp=30 riding on
  # dof_damping Kd=0.65, impratio=100) instead of dial-mpc's coarse
  # 20 ms torque-clipped plant. The dial-generated reference is NOT
  # open-loop reproducible on this plant (verified); sampling here makes the
  # reference feasible for the training env by construction.
  cfg = default_crate_climb_config()
  cfg.sim_dt = 0.002
  cfg.Kd = 0.65
  cfg.impratio = 100
  return cfg


class Go2CrateClimbPG(Go2CrateClimb):
  """Crate climbing with the crate reward on the playground training plant.

  Differences from Go2CrateClimb (the dial-faithful variant):
    * model = scene_mjx_sample_crate.xml (playground robot + crate + torso
      collision box), the exact model Go2SampleAPG trains on;
    * position-PD actuation: ctrl is the act2joint target (standard
      playground position actuators, PD baked by the shared Go2Env init),
      stepped with sim_dt=0.002 substeps — not act2tau torque control;
    * references recorded from this env are already in playground joint
      order: do NOT run fix_reference_leg_order.py on them.
  """

  def __init__(
      self,
      task: str | None = None,
      config: config_dict.ConfigDict = default_crate_climb_pg_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    del task
    go2_base.Go2Env.__init__(
        self,
        xml_path=consts.SAMPLE_CRATE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # Keep the PD baked by Go2Env.__init__ (position actuators); only match
    # the training env's solver setting.
    self._mj_model.opt.impratio = float(self._config.impratio)
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._physical_lowers, self._physical_uppers = self.mj_model.jnt_range[1:].T
    # Same climbing action ranges as the dial variant, but in PLAYGROUND leg
    # order (FL, FR, RL, RR): fronts first here, and dial's name->side flip
    # does not apply to the playground model.
    self._joint_lowers = jp.maximum(
        jp.array(self._physical_lowers),
        jp.array([
            -0.25, -1.0, -2.7,
            -0.25, -1.0, -2.7,
            -0.25, 0.0, -2.7,
            -0.25, 0.0, -2.7,
        ]),
    )
    self._joint_uppers = jp.minimum(
        jp.array(self._physical_uppers),
        jp.array([
            0.25, 1.4, -1.0,
            0.25, 1.4, -1.0,
            0.25, 1.8, -1.0,
            0.25, 1.8, -1.0,
        ]),
    )

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_ids = jp.array([
        self._mj_model.site(f"{name}_foot").id
        for name in ("FL", "FR", "RL", "RR")
    ])

    self._pos_tar = jp.array(self._config.pos_tar)
    self._head_offset = jp.array(self._config.head_offset)
    self._crate_lo = jp.array([
        self._config.crate_top_x[0],
        self._config.crate_top_y[0],
        self._config.crate_top_z[0],
    ])
    self._crate_hi = jp.array([
        self._config.crate_top_x[1],
        self._config.crate_top_y[1],
        self._config.crate_top_z[1],
    ])

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    ctrl = self.act2joint(action)  # position targets, not torques
    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

    info = dict(state.info)
    info["step"] = state.info["step"] + 1
    info["last_ctrl"] = ctrl

    obs = self._get_obs(data, info)
    rewards = self._get_reward(data, ctrl, state.info, info)
    scaled_rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(scaled_rewards.values())

    head_pos = self._head_pos(data)
    feet_on = jp.sum(self._feet_on_crate(data).astype(jp.float32))
    metrics = dict(state.metrics)
    for k, v in scaled_rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["head_position_error"] = jp.linalg.norm(head_pos - info["pos_tar"])
    metrics["feet_on_crate"] = feet_on
    metrics["success"] = (
        (metrics["head_position_error"] < 0.2) & (feet_on >= 3.0)
    ).astype(jp.float32)

    done = (info["step"] >= self._config.episode_length).astype(jp.float32)
    return state.replace(
        data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
    )


_GAIT_PHASES: Dict[str, jax.Array] = {
    "stand": jp.zeros(4),
    "walk": jp.array([0.0, 0.5, 0.75, 0.25]),
    "trot": jp.array([0.0, 0.5, 0.5, 0.0]),
    "canter": jp.array([0.0, 0.33, 0.33, 0.66]),
    "gallop": jp.array([0.0, 0.05, 0.4, 0.35]),
}

# duty_ratio, cadence, amplitude.
_GAIT_PARAMS: Dict[str, jax.Array] = {
    "stand": jp.array([1.0, 1.0, 0.0]),
    "walk": jp.array([0.75, 1.0, 0.08]),
    "trot": jp.array([0.45, 2.0, 0.08]),
    "canter": jp.array([0.4, 4.0, 0.06]),
    "gallop": jp.array([0.3, 3.5, 0.10]),
}


class Go2Trot(_Go2DialBase):
  """Position-controlled Go2 trotting environment (dial-mpc UnitreeGo2Env style).

  Uses the same dial-mpc Go2 scene as Go2SeqJump but keeps the position-actuator
  PD that Go2Env bakes into gainprm/biasprm, so the per-step ctrl is interpreted
  as a desired joint angle (in radians). Tracks linear and angular velocity
  commands plus a trot gait reference; intended to be driven by sampling-based
  controllers or by an RL policy.
  """

  def __init__(
      self,
      task: str | None = None,
      config: config_dict.ConfigDict = default_trot_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    del task
    super().__init__(
        xml_path=_dial_mpc_model_path("mjx_scene_force.xml"),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # The dial-mpc force XML defines plain motor actuators (biastype=NONE), so
    # Go2Env's biasprm[:, 1]=-Kp mutation is silently ignored and the actuator
    # behaves as `force = Kp*ctrl` instead of `Kp*(ctrl - q)`. Flip biastype to
    # AFFINE here so the gainprm/biasprm values Go2Env baked in actually take
    # effect, turning the motors into position actuators with in-simulator PD.
    # dof_damping[6:] = Kd was already set by Go2Env.
    self._mj_model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Mirrors dial-mpc UnitreeGo2Env.joint_range: same range for all 4 knees.
    self._physical_lowers, self._physical_uppers = self.mj_model.jnt_range[1:].T
    self._joint_lowers = jp.maximum(
        jp.array(self._physical_lowers),
        jp.array([
            -0.5, 0.4, -2.3,
            -0.5, 0.4, -2.3,
            -0.5, 0.4, -2.3,
            -0.5, 0.4, -2.3,
        ]),
    )
    self._joint_uppers = jp.minimum(
        jp.array(self._physical_uppers),
        jp.array([
            0.5, 1.4, -0.85,
            0.5, 1.4, -0.85,
            0.5, 1.4, -0.85,
            0.5, 1.4, -0.85,
        ]),
    )

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    # Foot order matches the dial-mpc gait phase order: FL, FR, RL, RR.
    self._feet_site_ids = jp.array([
        self._mj_model.site(f"{name}_foot").id for name in ("FL", "FR", "RL", "RR")
    ])

    gait = self._config.gait
    if gait not in _GAIT_PHASES:
      raise ValueError(
          f"Unknown gait '{gait}'. Supported: {sorted(_GAIT_PHASES.keys())}"
      )
    self._gait_phases = _GAIT_PHASES[gait]
    self._gait_params = _GAIT_PARAMS[gait]

  def act2joint(self, action: jax.Array) -> jax.Array:
    action_normalized = (action * self._config.action_scale + 1.0) / 2.0
    joint_targets = self._joint_lowers + action_normalized * (
        self._joint_uppers - self._joint_lowers
    )
    return jp.clip(joint_targets, self._physical_lowers, self._physical_uppers)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    ctrl = jp.array(self._default_pose)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    vel_cmd, ang_vel_cmd = self._default_commands()
    if self._config.randomize_tasks:
      rng, cmd_rng = jax.random.split(rng)
      vel_cmd, ang_vel_cmd = self.sample_command(cmd_rng)

    info = {
        "rng": rng,
        "step": jp.zeros((), dtype=jp.int32),
        "vel_tar": jp.zeros(3),
        "ang_vel_tar": jp.zeros(3),
        "vel_cmd": vel_cmd,
        "ang_vel_cmd": ang_vel_cmd,
        "yaw_tar": jp.zeros(()),
        "z_feet_tar": jp.zeros(4),
        "last_ctrl": jp.array(self._default_pose),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics["fwd_vel"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    ctrl = self.act2joint(action)
    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

    next_step = state.info["step"] + 1
    t = next_step.astype(jp.float32) * self.dt

    # Ramp up velocity commands like dial-mpc's UnitreeGo2Env.
    vel_cmd = state.info["vel_cmd"]
    ang_vel_cmd = state.info["ang_vel_cmd"]
    ramp = jp.minimum(t / self._config.ramp_up_time, 1.0)
    vel_tar = vel_cmd * ramp
    ang_vel_tar = ang_vel_cmd * ramp

    duty_ratio, cadence, amplitude = (
        self._gait_params[0], self._gait_params[1], self._gait_params[2]
    )
    z_feet_tar = _get_foot_step(duty_ratio, cadence, amplitude, self._gait_phases, t)

    info = dict(state.info)
    info["step"] = next_step
    info["vel_tar"] = vel_tar
    info["ang_vel_tar"] = ang_vel_tar
    info["yaw_tar"] = state.info["yaw_tar"] + ang_vel_tar[2] * self.dt
    info["z_feet_tar"] = z_feet_tar
    info["last_ctrl"] = ctrl

    rewards = self._get_reward(data, ctrl, info)
    scaled = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = sum(scaled.values())

    done = self._get_termination(data, next_step).astype(jp.float32)
    obs = self._get_obs(data, info)

    metrics = dict(state.metrics)
    for k, v in scaled.items():
      metrics[f"reward/{k}"] = v
    metrics["fwd_vel"] = self._local_linvel(data)[0]

    return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    return jp.concatenate([
        info["vel_tar"],
        info["ang_vel_tar"],
        info["last_ctrl"],
        data.qpos[3:7],  # base quaternion
        data.qpos[7:],
        self._local_linvel(data),
        self._local_angvel(data),
        data.qvel[6:],
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      ctrl: jax.Array,
      info: Dict[str, Any],
  ) -> Dict[str, jax.Array]:
    z_feet = data.site_xpos[self._feet_site_ids][:, 2]
    z_feet_tar = info["z_feet_tar"]
    gaits_reward = -jp.sum(jp.square((z_feet_tar - z_feet) / 0.05))

    upright_reward = -jp.sum(
        jp.square(self._upvector(data) - jp.array([0.0, 0.0, 1.0]))
    )

    rpy = self._roll_pitch_yaw(data.xquat[self._torso_body_id])
    yaw_err = self._wrap_to_pi(rpy[2] - info["yaw_tar"])
    yaw_reward = -jp.square(yaw_err)

    vb = self._local_linvel(data)
    ab = self._local_angvel(data)
    vel_reward = -jp.sum(jp.square(vb[:2] - info["vel_tar"][:2]))
    ang_vel_reward = -jp.square(ab[2] - info["ang_vel_tar"][2])

    height_reward = -jp.square(
        data.xpos[self._torso_body_id, 2] - self._config.base_height
    )

    energy_reward = -jp.sum(
        jp.maximum(ctrl * data.qvel[6:] / 160.0, 0.0) ** 2
    )

    return {
        "gaits": gaits_reward,
        "upright": upright_reward,
        "yaw": yaw_reward,
        "vel": vel_reward,
        "ang_vel": ang_vel_reward,
        "height": height_reward,
        "energy": energy_reward,
        "alive": jp.ones(()),
    }

  def _get_termination(self, data: mjx.Data, step: jax.Array) -> jax.Array:
    joint_angles = data.qpos[7:]
    return (
        (self._upvector(data)[-1] < 0.0)
        | jp.any(joint_angles < self._joint_lowers)
        | jp.any(joint_angles > self._joint_uppers)
        | (data.xpos[self._torso_body_id, 2] < 0.18)
        | (step >= self._config.episode_length)
    )

  def _default_commands(self) -> tuple[jax.Array, jax.Array]:
    vel = jp.array(
        [self._config.default_vx, self._config.default_vy, 0.0]
    )
    ang = jp.array([0.0, 0.0, self._config.default_vyaw])
    return vel, ang

  def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    key1, key2, key3 = jax.random.split(rng, 3)
    vx = jax.random.uniform(key1, (), minval=-1.5, maxval=1.5)
    vy = jax.random.uniform(key2, (), minval=-0.5, maxval=0.5)
    wyaw = jax.random.uniform(key3, (), minval=-1.5, maxval=1.5)
    return jp.array([vx, vy, 0.0]), jp.array([0.0, 0.0, wyaw])

  def render(
      self,
      trajectory: Sequence[mjx_env.State],
      height: int = 480,
      width: int = 640,
      camera: Optional[str] = "track",
  ) -> Sequence[np.ndarray]:
    return super().render(list(trajectory), height=height, width=width, camera=camera)
