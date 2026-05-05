"""Go2 sampled-reference APG imitation environment."""

from typing import Any, Dict, Optional, Union
import os

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mediapy as media
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2.TrotUtil import (
    cos_wave,
    dcos_wave,
    make_kinematic_ref,
    quaternion_to_matrix,
    quaternion_to_rotation_6d,
    rotate_inv,
)
from mujoco_playground._src.locomotion.go2.base import Go2Env


def default_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.Kp = 80.0
    cfg.Kd = 0.5
    cfg.sim_dt = 0.002
    cfg.ctrl_dt = 0.02
    cfg.episode_length = 240

    cfg.env = config_dict.ConfigDict()
    cfg.env.termination_height = 0.1
    cfg.env.step_k = 13
    cfg.env.err_threshold = 0.1
    cfg.env.action_scale = [0.2, 0.8, 0.8] * 4
    cfg.env.reset2ref = True
    cfg.env.reference_state_init = False
    cfg.env.reference_path = ""
    cfg.env.reference_loop = True
    cfg.env.impratio = 100

    cfg.pert_config = config_dict.ConfigDict()
    cfg.pert_config.enable = False
    cfg.pert_config.velocity_kick = [0.0, 3.0]
    cfg.pert_config.kick_durations = [0.05, 0.2]
    cfg.pert_config.kick_wait_times = [1.0, 3.0]

    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    cfg.rewards.scales.min_reference_tracking = -2.5 * 3e-3
    cfg.rewards.scales.reference_tracking = -10.0
    cfg.rewards.scales.feet_height = -10.0
    cfg.rewards.scales.base_tracking = -1.0

    cfg.impl = "jax"
    cfg.naconmax = 4 * 8192
    cfg.njmax = 40
    return cfg


class Go2SampleAPG(Go2Env):
    """Go2 sampled-reference imitation environment for APG training."""

    def __init__(
        self,
        task: str = None,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        del task
        super().__init__(
            xml_path=consts.MJX_XML_PATH.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self):
        self._dtype = jp.float64 if jax.config.jax_enable_x64 else jp.float32
        self._init_q = jp.array(
            self._mj_model.keyframe("home").qpos.copy(), dtype=self._dtype
        )
        self._default_ap_pose = jp.array(
            self._mj_model.keyframe("home").qpos[7:].copy(), dtype=self._dtype
        )

        self.lowers, self.uppers = self.mj_model.jnt_range[1:].T
        self.action_loc = jp.array(self._default_ap_pose, dtype=self._dtype)
        self.action_scale = jp.array(
            self._config.env.action_scale, dtype=self._dtype
        )

        self.termination_height = float(
            getattr(self._config.env, "termination_height", 0.1)
        )
        self.err_threshold = self._config.env.err_threshold
        self.reward_config = self._config.rewards
        self.feet_inds = jp.array([
            self._mj_model.geom(name).id for name in consts.FEET_GEOMS
        ])
        self.base_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.base_mass = self.mj_model.body("base").mass

        self.reference_loop = bool(getattr(self._config.env, "reference_loop", True))
        reference_path = str(getattr(self._config.env, "reference_path", ""))
        if reference_path:
            self._load_reference(reference_path)
        else:
            self._make_cosine_reference()

        self.reset2ref = self._config.env.reset2ref
        self.reference_state_init = self._config.env.reference_state_init

    def _make_cosine_reference(self) -> None:
        step_k = int(getattr(self._config.env, "step_k", 25))
        ref_joint_qpos = make_kinematic_ref(
            cos_wave, step_k, scale=0.3, dt=self.dt
        )
        ref_joint_qvel = make_kinematic_ref(
            dcos_wave, step_k, scale=0.3, dt=self.dt
        )
        self.l_cycle = int(ref_joint_qpos.shape[0])

        ref_joint_qpos = np.array(ref_joint_qpos) + np.array(self._default_ap_pose)
        ref_qpos = np.tile(
            np.asarray(self._init_q).reshape(1, self.mjx_model.nq),
            (self.l_cycle, 1),
        )
        ref_qpos[:, 7:] = ref_joint_qpos
        self.kinematic_ref_qpos = jp.array(ref_qpos, dtype=self._dtype)

        ref_qvel = np.zeros((self.l_cycle, self.mjx_model.nv))
        ref_qvel[:, 6:] = np.array(ref_joint_qvel)
        self.kinematic_ref_qvel = jp.array(ref_qvel, dtype=self._dtype)

    def _load_reference(self, reference_path: str) -> None:
        reference_path = os.path.expanduser(reference_path)
        with np.load(reference_path) as ref:
            qpos = np.asarray(ref["qpos"])
            qvel = np.asarray(ref["qvel"])

        if qpos.ndim != 2 or qpos.shape[1] != self.mjx_model.nq:
            raise ValueError(
                "Reference qpos must have shape "
                f"[T, {self.mjx_model.nq}], got {qpos.shape} from {reference_path}."
            )
        if qvel.ndim != 2 or qvel.shape[1] != self.mjx_model.nv:
            raise ValueError(
                "Reference qvel must have shape "
                f"[T, {self.mjx_model.nv}], got {qvel.shape} from {reference_path}."
            )
        if qpos.shape[0] != qvel.shape[0]:
            raise ValueError(
                "Reference qpos/qvel lengths must match, got "
                f"{qpos.shape[0]} and {qvel.shape[0]} from {reference_path}."
            )
        if qpos.shape[0] < 2:
            raise ValueError(
                f"Reference needs at least two frames, got {qpos.shape[0]}."
            )

        self.l_cycle = int(qpos.shape[0])
        self.kinematic_ref_qpos = jp.array(qpos, dtype=self._dtype)
        self.kinematic_ref_qvel = jp.array(qvel, dtype=self._dtype)

    def _reference_step_idx(self, step: jax.Array) -> jax.Array:
        step = jp.array(step, dtype=jp.int32)
        if self.reference_loop:
            return step % self.l_cycle
        return jp.minimum(step, self.l_cycle - 1)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self.reference_state_init:
            rng, step_rng = jax.random.split(rng)
            init_step = jax.random.randint(step_rng, (), 0, self.l_cycle)
            qpos = self.kinematic_ref_qpos[init_step]
            qvel = self.kinematic_ref_qvel[init_step]
        else:
            init_step = 0
            qpos = self._init_q
            qvel = jp.zeros(self.mjx_model.nv, dtype=self._dtype)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu, dtype=self._dtype),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        pen = jp.where(data.ncon > 0, jp.min(data._impl.contact.dist), 0.0)
        qpos = qpos.at[2].set(qpos[2] - pen)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu, dtype=self._dtype),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key1,
            minval=self._config.pert_config.kick_wait_times[0],
            maxval=self._config.pert_config.kick_wait_times[1],
        )
        steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
            jp.int32
        )
        pert_duration_seconds = jax.random.uniform(
            key2,
            minval=self._config.pert_config.kick_durations[0],
            maxval=self._config.pert_config.kick_durations[1],
        )
        pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
            jp.int32
        )
        pert_mag = jax.random.uniform(
            key3,
            minval=self._config.pert_config.velocity_kick[0],
            maxval=self._config.pert_config.velocity_kick[1],
        )

        state_info = {
            "rng": rng,
            "step": jp.array(init_step, dtype=jp.float32),
            "reward_tuple": {
                "reference_tracking": 0.0,
                "min_reference_tracking": 0.0,
                "feet_height": 0.0,
                "base_tracking": 0.0,
            },
            "last_action": jp.zeros(self.mjx_model.nu, dtype=self._dtype),
            "kinematic_ref": qpos,
            "steps_until_next_pert": steps_until_next_pert,
            "pert_duration_seconds": pert_duration_seconds,
            "pert_duration": pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3, dtype=self._dtype),
            "pert_mag": pert_mag,
        }

        obs = self._get_obs(data, state_info)
        reward, done = jp.zeros(2)
        metrics = {}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]
        state = mjx_env.State(data, obs, reward, done, metrics, state_info)
        return jax.lax.stop_gradient(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)

        action = jp.array(jp.clip(action, -1, 1), dtype=self._dtype)
        ctrl = jp.array(
            self.action_loc + (action * self.action_scale), dtype=self._dtype
        )

        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        step_idx = self._reference_step_idx(state.info["step"])
        ref_qpos = self.kinematic_ref_qpos[step_idx]
        ref_qvel = self.kinematic_ref_qvel[step_idx]
        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.mjx_model, ref_data)
        state.info["kinematic_ref"] = ref_qpos

        obs = self._get_obs(data, state.info)
        base_z = data.xpos[self.base_id, 2]
        done = jp.where(base_z < self.termination_height, 1.0, 0.0)
        base_z_axis_world = quaternion_to_matrix(data.xquat[1]) @ jp.array(
            [0.0, 0.0, 1.0]
        )
        done = jp.where(
            jp.dot(base_z_axis_world, jp.array([0.0, 0.0, 1.0])) < 0.0,
            1.0,
            done,
        )

        reward_tuple = dict(
            reference_tracking=self._reward_reference_tracking(data, ref_data)
            * self.reward_config.scales.reference_tracking,
            min_reference_tracking=self._reward_min_reference_tracking(
                ref_qpos, ref_qvel, data
            )
            * self.reward_config.scales.min_reference_tracking,
            feet_height=self._reward_feet_height(
                data.geom_xpos[self.feet_inds][:, 2],
                ref_data.geom_xpos[self.feet_inds][:, 2],
            )
            * self.reward_config.scales.feet_height,
            base_tracking=self._reward_base_tracking(data, ref_data)
            * self.reward_config.scales.base_tracking,
        )

        state.info["last_action"] = ctrl

        if self.reset2ref:
            err = (((data.xpos[1:] - ref_data.xpos[1:]) ** 2).sum(-1) ** 0.5).mean()
            to_ref = err > self.err_threshold
            reward_tuple["reference_tracking"] *= jp.where(to_ref, 10.0, 1.0)
            reward_tuple["base_tracking"] *= jp.where(to_ref, 10.0, 1.0)
            reward = sum(reward_tuple.values())
            for k in reward_tuple.keys():
                state.metrics[k] = reward_tuple[k]
            state.info["reward_tuple"] = reward_tuple
            data_blend = jax.tree_util.tree_map(
                lambda a, b: jp.where(to_ref, b, a), data, ref_data
            )
            obs = self._get_obs(data_blend, state.info)
            state.info["step"] = state.info["step"] + 1.0
            return state.replace(data=data_blend, obs=obs, reward=reward, done=done)

        reward = sum(reward_tuple.values())
        state.info["reward_tuple"] = reward_tuple
        for k in reward_tuple.keys():
            state.metrics[k] = reward_tuple[k]
        state.info["step"] = state.info["step"] + 1.0
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def play_ref_motion(self, render_every: int = 2, seed: int = 0):
        print("Playing reference motion...")
        rng = jax.random.PRNGKey(seed)
        state = self.reset(rng)
        data = state.data
        traj = []
        for i in range(0, self.l_cycle, render_every):
            data = data.replace(
                qpos=self.kinematic_ref_qpos[i],
                qvel=self.kinematic_ref_qvel[i],
            )
            data = mjx.forward(self.mjx_model, data)
            traj.append(
                state.replace(
                    data=data,
                    obs=None,
                    reward=0.0,
                    done=0.0,
                    info={"step": float(i)},
                )
            )

        fps = 1.0 / (self.dt * render_every)
        frames = self.render(traj, height=480, width=640)
        media.show_video(frames, fps=fps, loop=True)
        print(f"Rendered {len(frames)} frames at {fps:.1f} FPS.")
        return frames

    def _get_obs(self, data, state_info: Dict[str, Any]):
        yaw_rate = data.cvel[1, :3][2]
        g_local = rotate_inv(jp.array([0.0, 0.0, -1.0]), data.xquat[1])
        angles = data.qpos[7:19]
        step_idx = self._reference_step_idx(state_info["step"])
        kin_ref = self.kinematic_ref_qpos[step_idx][7:]
        obs = jp.concatenate([
            jp.array([yaw_rate]) * 0.25,
            g_local,
            angles - jp.array(self._default_ap_pose),
            state_info["last_action"],
            kin_ref,
        ])
        return jp.clip(obs, -100.0, 100.0)

    def _reward_reference_tracking(self, data, ref_data):
        f = lambda a, b: ((a - b) ** 2).sum(-1).mean()
        mse_pos = f(data.xpos[1:], ref_data.xpos[1:])
        mse_rot = f(
            quaternion_to_rotation_6d(data.xquat[1:]),
            quaternion_to_rotation_6d(ref_data.xquat[1:]),
        )
        mse_vel = f(data.cvel[1:, 3:], ref_data.cvel[1:, 3:])
        mse_ang = f(data.cvel[1:, :3], ref_data.cvel[1:, :3])
        return mse_pos + 0.1 * mse_rot + 0.01 * mse_vel + 0.001 * mse_ang

    def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, data):
        pos = jp.concatenate([data.qpos[:3], data.qpos[7:]])
        pos_targ = jp.concatenate([ref_qpos[:3], ref_qpos[7:]])
        return jp.linalg.norm(pos_targ - pos) + jp.linalg.norm(data.qvel - ref_qvel)

    def _reward_feet_height(self, feet_z, feet_z_ref):
        return jp.sum(jp.abs(feet_z - feet_z_ref))

    def _reward_base_tracking(self, data, ref_data):
        pos_err = jp.linalg.norm(data.xpos[1] - ref_data.xpos[1])
        q = data.xquat[1]
        q_ref = ref_data.xquat[1]
        dot = jp.clip(jp.abs(jp.dot(q, q_ref)), -1.0, 1.0)
        rot_err = jp.arccos(2 * dot**2 - 1)
        vel_err = jp.linalg.norm(data.cvel[1] - ref_data.cvel[1])
        return pos_err + 0.5 * rot_err + 0.1 * vel_err

    def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
            return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        def apply_pert(state: mjx_env.State) -> mjx_env.State:
            t = state.info["pert_steps"] * self.dt
            u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
            force = (
                u_t
                * self.base_mass
                * state.info["pert_mag"]
                / state.info["pert_duration_seconds"]
            )
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self.base_id, :3].set(
                force * state.info["pert_dir"]
            )
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state = state.replace(data=data)
            state.info["steps_since_last_pert"] = jp.where(
                state.info["pert_steps"] >= state.info["pert_duration"],
                0,
                state.info["steps_since_last_pert"],
            )
            state.info["pert_steps"] += 1
            return state

        def wait(state: mjx_env.State) -> mjx_env.State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_pert"] += 1
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state.info["pert_steps"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                0,
                state.info["pert_steps"],
            )
            state.info["pert_dir"] = jp.where(
                state.info["steps_since_last_pert"]
                >= state.info["steps_until_next_pert"],
                gen_dir(rng),
                state.info["pert_dir"],
            )
            return state.replace(data=data)

        return jax.lax.cond(
            state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"],
            apply_pert,
            wait,
            state,
        )

