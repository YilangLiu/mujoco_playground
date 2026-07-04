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


def _safe_norm(x: jax.Array) -> jax.Array:
    # jp.linalg.norm has a NaN gradient at exactly zero input; keep the
    # analytic gradient finite everywhere for APG.
    return jp.sqrt(jp.sum(x * x) + 1e-12)


def default_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.Kp = 80.0
    cfg.Kd = 0.5
    cfg.sim_dt = 0.002
    cfg.ctrl_dt = 0.02
    cfg.episode_length = 240

    cfg.env = config_dict.ConfigDict()
    # "flat" = collision-free scene (all prior runs/references); "rough" =
    # feet-only robot on the +-5 cm hfield (scene_mjx_sample_rough_terrain.xml,
    # same home keyframe as flat so action_loc and references carry over);
    # "crate" = dial-mpc's 0.6 m crate + torso collision box for the
    # belly-on-edge climbing reference (scene_mjx_sample_crate.xml).
    cfg.env.terrain = "flat"
    cfg.env.termination_height = 0.1
    cfg.env.step_k = 13
    cfg.env.err_threshold = 0.1
    # Abduction scale 0.4: the sampled references need up to ~0.36 rad of
    # abduction from home; at 0.2 the hard-clipped action saturates on ~5% of
    # reference targets and the clip has zero gradient exactly there.
    cfg.env.action_scale = [0.4, 0.8, 0.8] * 4
    cfg.env.reset2ref = True
    # DeepMimic-style alternative to reset2ref: END the episode (true
    # termination, critic bootstraps 0) when mean body tracking error
    # exceeds deviation_threshold, instead of snapping back. Removes
    # reward collection from off-reference attractors (e.g. parking in
    # front of the crate) without training a tube-dependent policy.
    cfg.env.terminate_on_deviation = False
    cfg.env.deviation_threshold = 0.4
    # Measure deviation against the NEAREST reference-ensemble member
    # instead of the per-episode tracked member: the ensemble defines a
    # demonstrated corridor, and a robot drifting from its tracked member
    # toward a neighboring demonstrated path should keep training, not
    # bootstrap 0. (With the tracked-member tube + injection noise, training
    # episodes died after ~13 steps — too short for credit to cross the
    # launch.) Reward still tracks the chosen member.
    cfg.env.deviation_nearest_member = False
    # Reverse-curriculum RSI window and tube-widening injection noise
    # (rsi_max_frame=-1 keeps the default [0, l_cycle - episode_length)).
    cfg.env.rsi_min_frame = 0
    cfg.env.rsi_max_frame = -1
    cfg.env.rsi_noise_qpos = 0.0
    cfg.env.rsi_noise_qvel = 0.0
    cfg.env.reference_state_init = False
    cfg.env.reference_path = ""
    cfg.env.reference_loop = True
    cfg.env.impratio = 100

    cfg.pert_config = config_dict.ConfigDict()
    cfg.pert_config.enable = False
    cfg.pert_config.velocity_kick = [0.0, 3.0]
    cfg.pert_config.kick_durations = [0.05, 0.2]
    cfg.pert_config.kick_wait_times = [1.0, 3.0]

    # Positive-form rewards: each term is in [0, scale] (exp(-error) shaping),
    # plus a constant alive bonus. Total per-step reward is bounded above by
    # ~13, so longer episodes always dominate dying early.
    cfg.rewards = config_dict.ConfigDict()
    cfg.rewards.scales = config_dict.ConfigDict()
    cfg.rewards.scales.reference_tracking = 5.0
    cfg.rewards.scales.feet_height = 2.0
    cfg.rewards.scales.base_tracking = 5.0
    # Body-frame base linear velocity tracking: a per-step signal that does
    # not require integration through contact, so APG gets a clean 1-step
    # gradient pulling the policy off the "stand still" attractor.
    cfg.rewards.scales.base_linvel = 5.0
    cfg.rewards.scales.alive = 1.0

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
        terrain = config.env.get("terrain", "flat")
        if config_overrides and "env.terrain" in config_overrides:
            terrain = config_overrides["env.terrain"]
        try:
            xml_path = {
                "flat": consts.MJX_XML_PATH,
                "rough": consts.SAMPLE_ROUGH_TERRAIN_XML,
                "crate": consts.SAMPLE_CRATE_XML,
            }[terrain]
        except KeyError as e:
            raise ValueError(
                f"Unknown env.terrain={terrain!r}; expected 'flat', 'rough',"
                f" or 'crate'"
            ) from e
        super().__init__(
            xml_path=xml_path.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._terrain = terrain
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
        self.terminate_on_deviation = bool(
            getattr(self._config.env, "terminate_on_deviation", False)
        )
        self.deviation_threshold = float(
            getattr(self._config.env, "deviation_threshold", 0.4)
        )
        self.deviation_nearest_member = bool(
            getattr(self._config.env, "deviation_nearest_member", False)
        )
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

        # RSI start-frame range: for non-looping references, restrict starts
        # so every episode has full reference coverage — otherwise episodes
        # starting near the end clamp on the final frame and spend most steps
        # tracking a frozen pose.
        episode_cap = int(self._config.episode_length)
        self._rsi_max_step = (
            self.l_cycle
            if self.reference_loop
            else max(1, self.l_cycle - episode_cap)
        )

        self.reset2ref = self._config.env.reset2ref
        self.reference_state_init = self._config.env.reference_state_init
        self._rsi_min_frame = int(
            getattr(self._config.env, "rsi_min_frame", 0)
        )
        self._rsi_max_frame = int(
            getattr(self._config.env, "rsi_max_frame", -1)
        )
        self._rsi_noise_qpos = float(
            getattr(self._config.env, "rsi_noise_qpos", 0.0)
        )
        self._rsi_noise_qvel = float(
            getattr(self._config.env, "rsi_noise_qvel", 0.0)
        )

        # Task-success instrumentation for the crate terrain: feet-site
        # positions tested against the crate-top window (crate geom spans
        # x [0.99, 1.61], y [-0.46, 0.46], top z = 0.6; site centers sit one
        # foot radius above the surface). Tracking reward has masked task
        # failure before (a policy parked in front of the crate collects
        # ~73% of the reward cap), so success is a first-class metric.
        self._track_crate = getattr(self, "_terrain", "flat") == "crate"
        if self._track_crate:
            self._crate_feet_sites = jp.array([
                self._mj_model.site(f"{nm}_foot").id
                for nm in ("FL", "FR", "RL", "RR")
            ])
            self._crate_lo = jp.array([0.99, -0.46, 0.55], dtype=self._dtype)
            self._crate_hi = jp.array(
                [1.61, 0.46, jp.inf], dtype=self._dtype
            )

        # Precompute every member's body positions at every frame (one-time
        # CPU kinematics) so nearest-member deviation is a lookup, not M
        # forward passes per step.
        self._ref_xpos_all = None
        if self.terminate_on_deviation and self.deviation_nearest_member:
            mjd = mujoco.MjData(self._mj_model)
            ref_qpos_np = np.asarray(self._ref_qpos)
            n_members, n_frames = ref_qpos_np.shape[:2]
            xpos_all = np.zeros(
                (n_members, n_frames, self._mj_model.nbody - 1, 3)
            )
            for m in range(n_members):
                for t in range(n_frames):
                    mjd.qpos[:] = ref_qpos_np[m, t]
                    mujoco.mj_kinematics(self._mj_model, mjd)
                    xpos_all[m, t] = mjd.xpos[1:]
            self._ref_xpos_all = jp.array(xpos_all, dtype=self._dtype)

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

        self.num_refs = 1
        self._ref_qpos = self.kinematic_ref_qpos[None]
        self._ref_qvel = self.kinematic_ref_qvel[None]

    def _load_reference(self, reference_path: str) -> None:
        """Loads a kinematic reference (or reference ensemble) from an npz.

        qpos/qvel are either [T, nq]/[T, nv] (a single reference) or
        [M, T, nq]/[M, T, nv] (an ensemble — e.g. DIAL-MPC recovery
        demonstrations grafted onto the nominal prefix, produced by
        learning/harvest_crate_recoveries.py). Each episode tracks one
        member, chosen uniformly at reset; all members share one length.

        References must be in *playground* joint order (leg blocks FL, FR,
        RL, RR). Files recorded by learning/play_go2_sampling.py on the
        dial-mpc model order leg blocks differently — convert them first
        with learning/fix_reference_leg_order.py.
        """
        reference_path = os.path.expanduser(reference_path)
        with np.load(reference_path) as ref:
            qpos = np.asarray(ref["qpos"])
            qvel = np.asarray(ref["qvel"])

        if qpos.ndim == 2:
            qpos = qpos[None]
        if qvel.ndim == 2:
            qvel = qvel[None]
        if qpos.ndim != 3 or qpos.shape[-1] != self.mjx_model.nq:
            raise ValueError(
                "Reference qpos must have shape [T, nq] or [M, T, nq] with "
                f"nq={self.mjx_model.nq}, got {qpos.shape} from {reference_path}."
            )
        if qvel.ndim != 3 or qvel.shape[-1] != self.mjx_model.nv:
            raise ValueError(
                "Reference qvel must have shape [T, nv] or [M, T, nv] with "
                f"nv={self.mjx_model.nv}, got {qvel.shape} from {reference_path}."
            )
        if qpos.shape[:2] != qvel.shape[:2]:
            raise ValueError(
                "Reference qpos/qvel leading shapes must match, got "
                f"{qpos.shape} and {qvel.shape} from {reference_path}."
            )
        if qpos.shape[1] < 2:
            raise ValueError(
                f"Reference needs at least two frames, got {qpos.shape[1]}."
            )

        self.num_refs = int(qpos.shape[0])
        self.l_cycle = int(qpos.shape[1])
        self._ref_qpos = jp.array(qpos, dtype=self._dtype)
        self._ref_qvel = jp.array(qvel, dtype=self._dtype)
        # 2D aliases exposing the nominal member (index 0) for external
        # tools (BC warm start, curriculum eval, reference video validation).
        self.kinematic_ref_qpos = self._ref_qpos[0]
        self.kinematic_ref_qvel = self._ref_qvel[0]

    def _reference_step_idx(self, step: jax.Array) -> jax.Array:
        step = jp.array(step, dtype=jp.int32)
        if self.reference_loop:
            return step % self.l_cycle
        return jp.minimum(step, self.l_cycle - 1)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self.reference_state_init:
            rng, step_rng, ref_rng = jax.random.split(rng, 3)
            # Reverse-curriculum window: rsi_min/max_frame override the
            # default [0, _rsi_max_step) range (rsi_max_frame=-1 keeps it).
            rsi_lo = self._rsi_min_frame
            rsi_hi = (
                self._rsi_max_frame
                if self._rsi_max_frame > 0
                else self._rsi_max_step
            )
            rsi_hi = max(rsi_lo + 1, rsi_hi)  # non-degenerate randint bounds
            init_step = jax.random.randint(step_rng, (), rsi_lo, rsi_hi)
            ref_idx = jax.random.randint(ref_rng, (), 0, self.num_refs)
            qpos = self._ref_qpos[ref_idx, init_step]
            qvel = self._ref_qvel[ref_idx, init_step]
            if self._rsi_noise_qpos > 0.0 or self._rsi_noise_qvel > 0.0:
                # Widen the reference tube to nonzero measure so
                # flight/landing feedback is learned on a neighborhood, not
                # a single knife-edge trajectory (applied BEFORE the
                # penetration fix below).
                rng, nq_rng, nv_rng = jax.random.split(rng, 3)
                qpos = qpos.at[7:].add(
                    self._rsi_noise_qpos
                    * jax.random.normal(nq_rng, (12,), dtype=self._dtype)
                )
                qvel = qvel + self._rsi_noise_qvel * jax.random.normal(
                    nv_rng, (self.mjx_model.nv,), dtype=self._dtype
                )
        else:
            init_step = 0
            ref_idx = jp.zeros((), dtype=jp.int32)
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

        # Lift out of actual penetration only (dist < 0). In the MJX JAX
        # impl ncon is a STATIC candidate-pair count (always > 0 here) and
        # contact.dist holds signed distances for all candidate pairs, so an
        # unclamped min(dist) is the POSITIVE gap to the nearest surface for
        # airborne states — subtracting it teleported flight-phase RSI
        # starts 9-26 cm down onto the ground/crate.
        pen = jp.where(
            data.ncon > 0,
            jp.minimum(jp.min(data._impl.contact.dist), 0.0),
            0.0,
        )
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
            # Frame this episode's reset pose was built from; used to resync
            # "step" after wrapper-level autoresets (which restore the cached
            # reset pose without resetting env info).
            "init_step": jp.array(init_step, dtype=jp.float32),
            # Reference-ensemble member this episode tracks. Constant within
            # an episode, so it stays consistent with the cached reset pose
            # across BraxAutoResetWrapper restores (like init_step).
            "ref_idx": jp.array(ref_idx, dtype=jp.int32),
            "reward_tuple": {
                "reference_tracking": 0.0,
                "feet_height": 0.0,
                "base_tracking": 0.0,
                "base_linvel": 0.0,
                "alive": 0.0,
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
        if self._track_crate:
            # Native env dtype on BOTH reset and step sides: a mismatch is a
            # lax.scan carry TypeError, and f32 doesn't survive brax's
            # EvalWrapper either (it multiplies by f64 active_episodes,
            # promoting the accumulator mid-scan).
            metrics["task/feet_on_crate"] = jp.zeros((), dtype=self._dtype)
            metrics["task/success"] = jp.zeros((), dtype=self._dtype)
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

        # Advance the reference index before computing obs / reward: the
        # action applied at time t produces data at time t+1, so we score and
        # condition the policy against the reference frame at t+1.
        #
        # BraxAutoResetWrapper (full_reset=False) restores the cached reset
        # pose on done but never resets env info, so this counter would
        # otherwise drift away from the restored pose (by a random offset,
        # thanks to brax APG's scramble_times). The wrapper does zero the
        # episode "steps" counter on done, so steps==0 marks the first step
        # after any (auto)reset: resync to the frame the cached pose holds.
        step_count = state.info["step"]
        if "steps" in state.info:
            step_count = jp.where(
                state.info["steps"] == 0, state.info["init_step"], step_count
            )
        state.info["step"] = step_count + 1.0
        step_idx = self._reference_step_idx(state.info["step"])
        ref_idx = state.info["ref_idx"]
        ref_qpos = self._ref_qpos[ref_idx, step_idx]
        ref_qvel = self._ref_qvel[ref_idx, step_idx]
        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.mjx_model, ref_data)
        state.info["kinematic_ref"] = ref_qpos

        obs = self._get_obs(data, state.info)
        # Only terminate on truly unrecoverable states (base upside-down).
        # The previous base_z < termination_height check let the policy game
        # APG by diving into the ground to truncate negative-reward unrolls.
        base_z_axis_world = quaternion_to_matrix(data.xquat[1]) @ jp.array(
            [0.0, 0.0, 1.0]
        )
        done = jp.where(
            jp.dot(base_z_axis_world, jp.array([0.0, 0.0, 1.0])) < 0.0,
            1.0,
            0.0,
        )
        if self.terminate_on_deviation:
            if self.deviation_nearest_member:
                # Distance to the demonstrated corridor: mean body-position
                # error against EACH ensemble member at this frame, min over
                # members. Reward still tracks the episode's chosen member.
                dev_err = (
                    (
                        (
                            data.xpos[1:][None]
                            - self._ref_xpos_all[:, step_idx]
                        )
                        ** 2
                    ).sum(-1)
                    ** 0.5
                ).mean(-1).min()
            else:
                dev_err = (
                    ((data.xpos[1:] - ref_data.xpos[1:]) ** 2).sum(-1) ** 0.5
                ).mean()
            done = jp.maximum(
                done,
                (dev_err > self.deviation_threshold).astype(done.dtype),
            )

        reward_tuple = dict(
            reference_tracking=self._reward_reference_tracking(data, ref_data)
            * self.reward_config.scales.reference_tracking,
            feet_height=self._reward_feet_height(
                data.geom_xpos[self.feet_inds][:, 2],
                ref_data.geom_xpos[self.feet_inds][:, 2],
            )
            * self.reward_config.scales.feet_height,
            base_tracking=self._reward_base_tracking(data, ref_data)
            * self.reward_config.scales.base_tracking,
            base_linvel=self._reward_base_linvel(data, ref_qvel)
            * self.reward_config.scales.base_linvel,
            alive=jp.array(self.reward_config.scales.alive, dtype=self._dtype),
        )

        state.info["last_action"] = ctrl

        if self._track_crate:
            feet = data.site_xpos[self._crate_feet_sites]
            feet_on = jp.sum(
                jp.all(
                    (feet > self._crate_lo) & (feet < self._crate_hi), axis=-1
                )
            ).astype(self._dtype)
            state.metrics["task/feet_on_crate"] = feet_on
            state.metrics["task/success"] = (feet_on >= 3.0).astype(
                self._dtype
            )

        if self.reset2ref:
            err = (((data.xpos[1:] - ref_data.xpos[1:]) ** 2).sum(-1) ** 0.5).mean()
            to_ref = err > self.err_threshold
            reward = sum(reward_tuple.values())
            for k in reward_tuple.keys():
                state.metrics[k] = reward_tuple[k]
            state.info["reward_tuple"] = reward_tuple
            data_blend = jax.tree_util.tree_map(
                lambda a, b: jp.where(to_ref, b, a), data, ref_data
            )
            obs = self._get_obs(data_blend, state.info)
            return state.replace(data=data_blend, obs=obs, reward=reward, done=done)

        reward = sum(reward_tuple.values())
        state.info["reward_tuple"] = reward_tuple
        for k in reward_tuple.keys():
            state.metrics[k] = reward_tuple[k]
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
        base_quat = data.xquat[1]
        g_local = rotate_inv(jp.array([0.0, 0.0, -1.0]), base_quat)
        angles = data.qpos[7:19]
        step_idx = self._reference_step_idx(state_info["step"])
        # External callers (BC dataset building, curriculum eval) may pass
        # hand-built info dicts; default to the nominal member for them.
        ref_idx = state_info.get("ref_idx", jp.zeros((), dtype=jp.int32))
        ref_frame_qpos = self._ref_qpos[ref_idx, step_idx]
        kin_ref = ref_frame_qpos[7:]

        # Base tracking signals, expressed in the base body frame so the
        # policy doesn't have to learn an implicit yaw rotation between
        # world-frame quantities.
        ref_base_pos = ref_frame_qpos[:3]
        base_pos_err_local = rotate_inv(ref_base_pos - data.xpos[1], base_quat)
        # qvel[:3] (free-joint origin velocity, world axes) matches the
        # convention of the reference qvel it is paired with; cvel measures
        # at the subtree COM and differs by omega x r.
        base_linvel_local = rotate_inv(data.qvel[:3], base_quat)
        base_angvel_local = rotate_inv(data.cvel[1, :3], base_quat)
        ref_base_linvel_local = rotate_inv(
            self._ref_qvel[ref_idx, step_idx][:3], base_quat
        )
        # Cyclic phase encoding built from the wrapped reference index, so it
        # stays bounded and consistent with the frame actually being tracked
        # (the raw counter is unbounded across autoresets). Looping references
        # get a full-circle sin/cos; non-looping ones a half circle so cos is
        # monotonic in progress and the end doesn't alias with the start.
        frac = step_idx.astype(self._dtype) / self.l_cycle
        angle = (2.0 if self.reference_loop else 1.0) * jp.pi * frac
        phase = jp.array([jp.sin(angle), jp.cos(angle)], dtype=self._dtype)

        obs = jp.concatenate([
            g_local,
            angles - jp.array(self._default_ap_pose),
            state_info["last_action"],
            kin_ref,
            base_pos_err_local,
            base_linvel_local,
            ref_base_linvel_local,
            base_angvel_local,
            phase,
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
        err = mse_pos + 0.1 * mse_rot + 0.01 * mse_vel + 0.001 * mse_ang
        return jp.exp(-err)

    def _reward_feet_height(self, feet_z, feet_z_ref):
        return jp.exp(-10.0 * jp.sum(jp.abs(feet_z - feet_z_ref)))

    def _reward_base_tracking(self, data, ref_data):
        pos_err = _safe_norm(data.xpos[1] - ref_data.xpos[1])
        q = data.xquat[1]
        q_ref = ref_data.xquat[1]
        # Smooth rotation error: 1 - <q, q_ref>^2 is double-cover invariant
        # and ~theta^2/4 near alignment. The previous arccos geodesic has
        # unbounded slope as the quaternions align, which amplifies gradient
        # noise exactly when tracking is good (and NaNs at exact alignment).
        rot_err = 1.0 - jp.dot(q, q_ref) ** 2
        vel_err = _safe_norm(data.cvel[1] - ref_data.cvel[1])
        err = 2.0 * pos_err + 0.5 * rot_err + 0.1 * vel_err
        return jp.exp(-err)

    def _reward_base_linvel(self, data, ref_qvel):
        # Body-frame linear velocity tracking. Both vectors are rotated into
        # the current base frame so the gradient signal aligns with the
        # ref_base_linvel_local observation the policy sees. qvel[:3] matches
        # the reference's free-joint-origin convention (cvel is COM-based).
        base_quat = data.xquat[1]
        data_linvel_local = rotate_inv(data.qvel[:3], base_quat)
        ref_linvel_local = rotate_inv(ref_qvel[:3], base_quat)
        err = _safe_norm(ref_linvel_local - data_linvel_local)
        return jp.exp(-err)

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

