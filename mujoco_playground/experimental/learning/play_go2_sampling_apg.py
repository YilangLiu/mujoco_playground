import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["MUJOCO_GL"] = "egl"
os.environ["JAX_LOG_COMPILES"] = "0"

import functools
from pathlib import Path

import jax
import jax.numpy as jp
from jax import config
import mediapy as media
import numpy as np

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "high")

print("JAX Device:", jax.devices())
print("jax.devices():", jax.devices())
print("local_device_count:", jax.local_device_count())

from brax.envs.wrappers import training as brax_training
from brax.io import model
from brax.training.agents.apg import networks as apg_networks

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


env_name = "Go2SampleAPG"
REFERENCE_PATH = Path(
    os.environ.get(
        "GO2_SAMPLING_REFERENCE_PATH",
        "references/go2_seqjump_sampling_ref.npz",
    )
)
LOGDIR = Path(os.environ.get("GO2_SAMPLING_LOGDIR", "logs"))
RUN_DIR_OVERRIDE = os.environ.get("GO2_SAMPLING_RUN_DIR")
MODEL_PATH_OVERRIDE = os.environ.get("GO2_SAMPLING_MODEL_PATH")
VIDEO_PATH_OVERRIDE = os.environ.get("GO2_SAMPLING_EVAL_VIDEO_PATH")


def _latest_numeric_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoints = [p for p in checkpoint_dir.iterdir() if p.name.isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"No numeric checkpoints found in: {checkpoint_dir}")
    return sorted(checkpoints, key=lambda p: int(p.name))[-1]


def _latest_run_dir() -> Path:
    if RUN_DIR_OVERRIDE is not None:
        run_dir = Path(RUN_DIR_OVERRIDE)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    run_dirs = [
        p
        for p in LOGDIR.glob(f"{env_name}-*")
        if p.is_dir() and (p / "checkpoints").is_dir()
    ]
    if not run_dirs:
        raise FileNotFoundError(
            f"No {env_name} run directories found under {LOGDIR}. "
            "Set GO2_SAMPLING_MODEL_PATH or GO2_SAMPLING_RUN_DIR explicitly."
        )
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime)[-1]


def resolve_output_paths() -> tuple[Path, Path]:
    if MODEL_PATH_OVERRIDE is not None:
        model_path = Path(MODEL_PATH_OVERRIDE)
        run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
    else:
        run_dir = _latest_run_dir()
        model_path = _latest_numeric_checkpoint(run_dir / "checkpoints")

    video_path = (
        Path(VIDEO_PATH_OVERRIDE)
        if VIDEO_PATH_OVERRIDE is not None
        else run_dir / "eval_motion.mp4"
    )
    return model_path, video_path


def make_env():
    cfg = registry.get_default_config(env_name)
    cfg["env"]["reset2ref"] = False
    cfg["env"]["reference_state_init"] = False
    cfg["env"]["reference_path"] = REFERENCE_PATH.as_posix()
    cfg["env"]["reference_loop"] = False
    cfg["pert_config"]["enable"] = False
    cfg["env"]["impratio"] = 100
    if REFERENCE_PATH.exists():
        with np.load(REFERENCE_PATH) as ref:
            cfg["episode_length"] = int(ref["qpos"].shape[0] - 1)
    else:
        raise FileNotFoundError(
            f"Sampling reference not found: {REFERENCE_PATH}. Generate it with "
            "`python learning/play_go2_sampling.py "
            "--reference_output_path references/go2_seqjump_sampling_ref.npz`."
        )
    return registry.load(env_name, cfg)


def make_normalize_fn(mean, std, max_abs_value=None):
    def normalize_fn(batch, _unused_processor_params=None):
        def normalize_leaf(data, m, s):
            if not jp.issubdtype(data.dtype, jp.inexact):
                return data
            data = (data - m) / s
            if max_abs_value is not None:
                data = jp.clip(data, -max_abs_value, +max_abs_value)
            return data

        return jax.tree_util.tree_map(normalize_leaf, batch, mean, std)

    return normalize_fn


def render_rollout(
    reset_fn,
    step_fn,
    env,
    inference_fn,
    video_path,
    n_step,
    render_every,
    seed=42,
):
    rng = jax.random.PRNGKey(seed)
    state = reset_fn(jax.random.split(rng, 1))
    rollout = [state]
    rewards = 0.0
    print("Starting rollout...")
    for i in range(n_step):
        act_rng, rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)
        rollout.append(state)
        rewards += state.reward
        if bool(state.done[0]):
            print(f"Early termination at step {i + 1}")
            break

    print(f"Total reward: {float(rewards[0]):.1f}")
    traj = rollout[::render_every]
    fps = 1.0 / (env.dt * render_every)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames = env.render(traj, height=480, width=640, camera="track")
    media.write_video(video_path.as_posix(), frames, fps=fps)
    print(f"Saved evaluation video to {video_path}.")
    return state


if __name__ == "__main__":
    model_path, video_path = resolve_output_paths()
    print(f"Loading APG policy params from: {model_path}")
    print(f"Evaluation video will be saved to: {video_path}")

    demo_env = make_env()
    demo_env = brax_training.VmapWrapper(demo_env)
    demo_step_fn = jax.jit(demo_env.step)
    demo_reset_fn = jax.jit(demo_env.reset)

    params = model.load_params(model_path.as_posix())
    params = (params[0], params[1])

    apg_params = locomotion_params.brax_apg_config(env_name)
    apg_training_params = dict(apg_params)
    network_factory = apg_networks.make_apg_networks
    if "network_factory" in apg_params:
        del apg_training_params["network_factory"]
        network_factory = functools.partial(
            apg_networks.make_apg_networks,
            **apg_params.network_factory,
        )

    normalize = lambda x, y: x
    if apg_params["normalize_observations"]:
        mean, std = params[0].mean, params[0].std
        normalize = make_normalize_fn(mean, std)

    apg_network = network_factory(
        demo_env.observation_size,
        demo_env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_inference_fn = apg_networks.make_inference_fn(apg_network)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    render_rollout(
        demo_reset_fn,
        demo_step_fn,
        demo_env,
        jit_inference_fn,
        video_path,
        n_step=1000,
        render_every=1,
        seed=42,
    )
