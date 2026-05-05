"""Sequential jumping task for Unitree Go2 with sampling-friendly controls."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

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


def default_config() -> config_dict.ConfigDict:
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


class Go2Sampling(go2_base.Go2Env):
  """Go2 sequential jumping environment based on Dial-MPC's SeqJump task."""

  def __init__(
      self,
      task: str | None = None,
      config: config_dict.ConfigDict = default_config(),
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

  def _local_linvel(self, data: mjx.Data) -> jax.Array:
    return self._rotate_inv(data.qvel[:3], data.xquat[self._torso_body_id])

  def _local_angvel(self, data: mjx.Data) -> jax.Array:
    return self._rotate_inv(data.qvel[3:6], data.xquat[self._torso_body_id])

  def _upvector(self, data: mjx.Data) -> jax.Array:
    return math.rotate(jp.array([0.0, 0.0, 1.0]), data.xquat[self._torso_body_id])

  def _rotate_inv(self, vec: jax.Array, quat: jax.Array) -> jax.Array:
    quat_inv = quat.at[1:].multiply(-1.0)
    return math.rotate(vec, quat_inv)

  def _roll_pitch_yaw(self, quat: jax.Array) -> jax.Array:
    w, x, y, z = quat
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jp.array([roll, pitch, yaw])

  def _wrap_to_pi(self, angle: jax.Array) -> jax.Array:
    return jp.arctan2(jp.sin(angle), jp.cos(angle))

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


Go2SeqJumpSampling = Go2Sampling
