"""Standalone evaluation script: loads a PPO checkpoint, runs rollouts, saves video and trajectory."""

import datetime
import functools
import json
import os
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import numpy as np


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path", None, "Path to checkpoint directory or specific step dir. Required."
)
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides", None, "JSON string of env config overrides."
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_NUM_EPISODES = flags.DEFINE_integer("num_episodes", 1, "Number of episodes to evaluate")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_RENDER_EVERY = flags.DEFINE_integer("render_every", 2, "Render every N steps")
_HEIGHT = flags.DEFINE_integer("height", 480, "Video frame height")
_WIDTH = flags.DEFINE_integer("width", 640, "Video frame width")
_OUTDIR = flags.DEFINE_string("outdir", None, "Output directory. Defaults to checkpoint dir.")
_DETERMINISTIC = flags.DEFINE_boolean("deterministic", True, "Use deterministic policy")
_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", None, "Number of steps per episode. Defaults to the training episode length."
)


def _quat_to_euler_zyx(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternions (w, x, y, z) to roll/pitch/yaw in radians.

    Args:
        quat: (T, 4) array of (w, x, y, z).
    Returns:
        (T, 3) array of (roll, pitch, yaw) in radians.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def _plot_go2_episode(
    qpos: np.ndarray,
    torque: np.ndarray,
    time: np.ndarray,
    save_path: str,
) -> None:
    """Save a 4-subplot figure of base orientation + joint torques for a Go2 episode.

    Torque is taken directly from mjx `data.actuator_force`. Actuator order
    matches the go2_mjx.xml `<actuator>` block:
    [FL_abd, FL_hip, FL_knee, FR_abd, FR_hip, FR_knee,
     RL_abd, RL_hip, RL_knee, RR_abd, RR_hip, RR_knee].
    """
    # Drop samples where the brax auto-reset wrapper has already replaced
    # state with the reset pose (torque snaps to the home-pose hold value),
    # which collapses all legs to the same value at the last sample.
    if qpos.shape[0] > 1:
        qpos = qpos[:-1]
        torque = torque[:-1]
        time = time[:-1]

    rpy_deg = np.degrees(_quat_to_euler_zyx(qpos[:, 3:7]))

    leg_names = ["FL", "FR", "RL", "RR"]
    abd_idx = [0, 3, 6, 9]
    hip_idx = [1, 4, 7, 10]
    knee_idx = [2, 5, 8, 11]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(time, rpy_deg[:, 2], label="yaw")
    axes[0].plot(time, rpy_deg[:, 1], label="pitch")
    axes[0].plot(time, rpy_deg[:, 0], label="roll")
    axes[0].set_ylabel("angle (deg)")
    axes[0].set_title("Base orientation (yaw / pitch / roll)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    for ax, idx, title in (
        (axes[1], abd_idx, "Abduction torques"),
        (axes[2], hip_idx, "Hip torques"),
        (axes[3], knee_idx, "Knee torques"),
    ):
        for leg, j in zip(leg_names, idx):
            ax.plot(time, torque[:, j], label=leg)
        ax.set_ylabel("torque (N·m)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


_LEG_JOINT_IDX = {
    "FL": (0, 1, 2),
    "FR": (3, 4, 5),
    "RL": (6, 7, 8),
    "RR": (9, 10, 11),
}


def _plot_go2_single_support_torque_vs_pitch(
    qpos: np.ndarray,
    torque: np.ndarray,
    support_leg: str,
    save_path: str,
) -> None:
    """Scatter support-leg torque norm vs. base pitch for Single{Hand,Foot}stand.

    `config.support_leg` is fixed, so the support leg is known a priori
    (no contact inference needed). Each timestep contributes one scatter point.
    """
    if support_leg not in _LEG_JOINT_IDX:
        raise ValueError(
            f"support_leg must be one of {list(_LEG_JOINT_IDX)}, got '{support_leg}'"
        )

    if qpos.shape[0] > 1:
        qpos = qpos[:-1]
        torque = torque[:-1]

    pitch_deg = np.degrees(_quat_to_euler_zyx(qpos[:, 3:7])[:, 1])
    a, h, k = _LEG_JOINT_IDX[support_leg]
    torque_norm = np.linalg.norm(torque[:, [a, h, k]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pitch_deg,
        torque_norm,
        c=np.arange(pitch_deg.shape[0]),
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("step index")
    ax.set_xlabel("base pitch (deg)")
    ax.set_ylabel(f"‖τ_abd, τ_hip, τ_knee‖ of {support_leg} (N·m)")
    ax.set_title(f"Support-leg ({support_leg}) torque norm vs. base pitch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _resolve_checkpoint(path_str: str) -> epath.Path:
    """Return the most recent numeric step directory, or the path itself."""
    p = epath.Path(path_str).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")
    if p.is_dir():
        numeric_dirs = [d for d in p.glob("*") if d.is_dir() and d.name.isdigit()]
        if numeric_dirs:
            numeric_dirs.sort(key=lambda x: int(x.name))
            chosen = numeric_dirs[-1]
            print(f"Found step directories; using latest: {chosen}")
            return chosen
    return p


def _get_rl_config(env_name: str, impl: str, vision: bool):
    if env_name in mujoco_playground.manipulation._envs:
        if vision:
            return manipulation_params.brax_vision_ppo_config(env_name, impl)
        return manipulation_params.brax_ppo_config(env_name, impl)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, impl)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        if vision:
            return dm_control_suite_params.brax_vision_ppo_config(env_name, impl)
        return dm_control_suite_params.brax_ppo_config(env_name, impl)
    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def main(argv):
    del argv

    if _CHECKPOINT_PATH.value is None:
        raise ValueError("--checkpoint_path is required.")

    env_name = _ENV_NAME.value
    impl = _IMPL.value
    vision = _VISION.value
    num_episodes = _NUM_EPISODES.value
    seed = _SEED.value

    restore_path = _resolve_checkpoint(_CHECKPOINT_PATH.value)
    print(f"Restoring from: {restore_path}")

    outdir = epath.Path(_OUTDIR.value).resolve() if _OUTDIR.value else restore_path.parent.parent
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = outdir / f"eval-{env_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {run_dir}")

    env_cfg = registry.get_default_config(env_name)
    ppo_params = _get_rl_config(env_name, impl, vision)
    num_steps = _NUM_STEPS.value if _NUM_STEPS.value is not None else num_steps

    env_cfg_overrides = {"impl": impl}
    if vision:
        env_cfg_overrides["vision"] = True
        env_cfg_overrides["vision_config.nworld"] = num_episodes
    if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
        env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

    print(f"Environment Config:\n{env_cfg}")
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")

    # Build network factory (must match training config).
    network_fn = (
        ppo_networks_vision.make_ppo_networks_vision if vision else ppo_networks.make_ppo_networks
    )
    network_factory = (
        functools.partial(network_fn, **ppo_params.network_factory)
        if hasattr(ppo_params, "network_factory")
        else network_fn
    )

    # Load the model by running ppo.train with num_timesteps=0.
    training_params = dict(ppo_params)
    training_params["num_timesteps"] = 0
    if "network_factory" in training_params:
        del training_params["network_factory"]
    num_eval_envs = training_params.pop("num_eval_envs", 128)

    infer_env_overrides = dict(env_cfg_overrides)
    if vision:
        infer_env_overrides["vision_config.nworld"] = num_episodes

    infer_env = registry.load(env_name, config=registry.get_default_config(env_name), config_overrides=infer_env_overrides)
    train_env = registry.load(env_name, config=env_cfg, config_overrides=env_cfg_overrides)

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=seed,
        restore_checkpoint_path=restore_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        vision=vision,
    )

    make_inference_fn, params, _ = train_fn(
        environment=train_env,
        progress_fn=lambda *_: None,
        eval_env=infer_env,
    )
    print("Model loaded.")

    inference_fn = make_inference_fn(params, deterministic=_DETERMINISTIC.value)
    jit_inference_fn = jax.jit(inference_fn)

    wrapped_env = wrapper.wrap_for_brax_training(
        infer_env,
        episode_length=num_steps,
        action_repeat=ppo_params.get("action_repeat", 1),
    )

    rng = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    reset_states = jax.jit(wrapped_env.reset)(rng)

    # Build empty trajectory template.
    empty_data = reset_states.data.__class__(
        **{k: None for k in reset_states.data.__annotations__}
    )
    empty_traj = reset_states.__class__(**{k: None for k in reset_states.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        act_keys = jax.random.split(act_key, num_episodes)
        act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
        state = wrapped_env.step(state, act)
        traj_data = empty_traj.tree_replace({
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.actuator_force": state.data.actuator_force,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        return (state, rng), traj_data

    @jax.jit
    def do_rollout(state, rng):
        _, traj = jax.lax.scan(step, (state, rng), None, length=num_steps)
        return traj

    print("Running rollouts...")
    traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(seed + 1))
    # Shape: (time, num_episodes, ...) -> (num_episodes, time, ...)
    traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)

    trajectories = []
    for i in range(num_episodes):
        t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        traj_steps = [jax.tree.map(lambda x, j=j: x[j], t) for j in range(num_steps)]
        trajectories.append(traj_steps)

    # Save trajectories as npz files.
    is_go2 = env_name.lower().startswith("go2")
    for i, rollout in enumerate(trajectories):
        qpos_arr = np.array([s.data.qpos for s in rollout])
        qvel_arr = np.array([s.data.qvel for s in rollout])
        ctrl_arr = np.array([s.data.ctrl for s in rollout])
        time_arr = np.array([s.data.time for s in rollout])
        actuator_force_arr = np.array([s.data.actuator_force for s in rollout])
        npz_path = run_dir / f"trajectory_{i}.npz"
        np.savez(
            str(npz_path),
            qpos=qpos_arr,
            qvel=qvel_arr,
            ctrl=ctrl_arr,
            time=time_arr,
            actuator_force=actuator_force_arr,
        )
        print(f"Trajectory saved: {npz_path}")

        if is_go2:
            plot_path = run_dir / f"go2_diagnostics_{i}.png"
            # Monotonic rollout clock: state.data.time resets to 0 whenever the
            # wrapped env auto-resets mid-rollout, which shows up as a diagonal
            # line connecting segments in the plot.
            plot_time = np.arange(qpos_arr.shape[0]) * float(infer_env.dt)
            _plot_go2_episode(
                qpos=qpos_arr,
                torque=actuator_force_arr,
                time=plot_time,
                save_path=str(plot_path),
            )
            print(f"Go2 diagnostics plot saved: {plot_path}")

            if env_name in ("Go2SingleHandstand", "Go2SingleFootstand"):
                support_leg = str(env_cfg.support_leg)
                support_plot_path = (
                    run_dir / f"go2_support_torque_vs_pitch_{i}.png"
                )
                _plot_go2_single_support_torque_vs_pitch(
                    qpos=qpos_arr,
                    torque=actuator_force_arr,
                    support_leg=support_leg,
                    save_path=str(support_plot_path),
                )
                print(
                    "Go2 support-leg torque-vs-pitch plot saved:"
                    f" {support_plot_path}"
                )

    # Render and save videos.
    render_every = _RENDER_EVERY.value
    fps = 1.0 / infer_env.dt / render_every
    print(f"Rendering at {fps:.1f} FPS...")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    for i, rollout in enumerate(trajectories):
        frames = infer_env.render(
            rollout[::render_every],
            height=_HEIGHT.value,
            width=_WIDTH.value,
            scene_option=scene_option,
            camera="track",
        )
        video_path = run_dir / f"rollout_{i}.mp4"
        media.write_video(str(video_path), frames, fps=fps)
        print(f"Video saved: {video_path}")

    print("Done.")


def run():
    app.run(main)


if __name__ == "__main__":
    run()
