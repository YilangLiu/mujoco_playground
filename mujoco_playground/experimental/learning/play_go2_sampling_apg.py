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

# Keep these defaults in sync with go2_sampling_apg.py so train/eval
# agree on what each task means (reference file, looping behaviour, PD).
TASK_DEFAULTS = {
    # *_pg.npz files are the playground-joint-order conversions produced by
    # learning/fix_reference_leg_order.py (keep in sync with
    # go2_sampling_apg.py).
    "seq_jump": dict(
        reference_path="references/go2_seqjump_sampling_ref_pg.npz",
        reference_loop=False,
        eval_video="eval_seqjump.mp4",
        env_overrides={},
    ),
    "trot": dict(
        reference_path="references/go2_trot_sampling_ref_pg.npz",
        reference_loop=True,
        eval_video="eval_trot.mp4",
        env_overrides={
            # Match dial-mpc UnitreeGo2EnvConfig + unitree_go2_trot.yaml so
            # the eval physics is identical to training.
            "Kp": 30.0,
            "Kd": 0.0,
            "ctrl_dt": 0.02,
        },
    ),
    # Online-reference runs store their generated reference inside the run
    # directory; reference_path resolves to <run_dir>/generated_reference.npz.
    "trot_online": dict(
        reference_path=None,
        reference_loop=False,
        eval_video="eval_trot_online.mp4",
        env_overrides={
            # Keep in sync with go2_sampling_apg.py: Kd=0.65 dof_damping
            # (faithful dial-mpc plant).
            "Kp": 30.0,
            "Kd": 0.65,
            "ctrl_dt": 0.02,
        },
    ),
}


def _apply_env_overrides(cfg, overrides):
    for dotted_key, value in overrides.items():
        target = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
TASK = os.environ.get("GO2_SAMPLING_TASK", "seq_jump")
if TASK not in TASK_DEFAULTS:
    raise ValueError(
        f"Unknown GO2_SAMPLING_TASK='{TASK}'. "
        f"Supported: {sorted(TASK_DEFAULTS)}"
    )
_task_cfg = TASK_DEFAULTS[TASK]

_ref_override = os.environ.get("GO2_SAMPLING_REFERENCE_PATH")
_ref_default = _task_cfg["reference_path"]
REFERENCE_PATH = (
    Path(_ref_override or _ref_default)
    if (_ref_override or _ref_default)
    else None
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

    # New runs are named "Go2SampleAPG-{task}-{timestamp}"; pre-task runs are
    # "Go2SampleAPG-{timestamp}". Filter to the current task, and for seq_jump
    # also accept the untagged legacy layout.
    candidates = [
        p
        for p in LOGDIR.glob(f"{env_name}-*")
        if p.is_dir()
        and (p / "checkpoints").is_dir()
        # Skip runs with no saved params (e.g. generate-only or aborted runs).
        and any(c.name.isdigit() for c in (p / "checkpoints").iterdir())
    ]
    run_dirs = [p for p in candidates if p.name.startswith(f"{env_name}-{TASK}-")]
    if not run_dirs and TASK == "seq_jump":
        untagged = []
        for p in candidates:
            tail = p.name[len(env_name) + 1 :]
            head = tail.split("-", 1)[0]
            if head not in TASK_DEFAULTS:
                untagged.append(p)
        run_dirs = untagged
    if not run_dirs:
        raise FileNotFoundError(
            f"No {env_name}-{TASK}-* run directories found under {LOGDIR}. "
            "Set GO2_SAMPLING_MODEL_PATH or GO2_SAMPLING_RUN_DIR explicitly."
        )
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime)[-1]


def resolve_output_paths() -> tuple[Path, Path, Path]:
    if MODEL_PATH_OVERRIDE is not None:
        model_path = Path(MODEL_PATH_OVERRIDE)
        run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
    else:
        run_dir = _latest_run_dir()
        model_path = _latest_numeric_checkpoint(run_dir / "checkpoints")

    video_path = (
        Path(VIDEO_PATH_OVERRIDE)
        if VIDEO_PATH_OVERRIDE is not None
        else run_dir / _task_cfg["eval_video"]
    )
    return model_path, video_path, run_dir


def make_env():
    cfg = registry.get_default_config(env_name)
    cfg["env"]["reset2ref"] = False
    cfg["env"]["reference_state_init"] = False
    cfg["env"]["reference_path"] = REFERENCE_PATH.as_posix()
    cfg["env"]["reference_loop"] = _task_cfg["reference_loop"]
    cfg["pert_config"]["enable"] = False
    cfg["env"]["impratio"] = 100
    _apply_env_overrides(cfg, _task_cfg["env_overrides"])
    if REFERENCE_PATH.exists():
        with np.load(REFERENCE_PATH) as ref:
            cfg["episode_length"] = int(ref["qpos"].shape[0] - 1)
    else:
        if _task_cfg["reference_path"] is None:
            raise FileNotFoundError(
                f"Sampling reference not found: {REFERENCE_PATH}. Online-"
                f"reference runs generate it at training time; expected "
                f"generated_reference.npz inside the run directory (or set "
                f"GO2_SAMPLING_REFERENCE_PATH to one)."
            )
        raise FileNotFoundError(
            f"Sampling reference not found: {REFERENCE_PATH}. Generate the raw "
            f"rollout with `python learning/play_go2_sampling.py --task {TASK}` "
            f"and convert it to playground joint order with "
            f"`python learning/fix_reference_leg_order.py`."
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
    model_path, video_path, run_dir = resolve_output_paths()
    if REFERENCE_PATH is None:
        # Online-reference runs keep their generated reference in the run dir.
        REFERENCE_PATH = run_dir / "generated_reference.npz"
    print(f"Evaluating Go2SampleAPG task='{TASK}', reference={REFERENCE_PATH}.")
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
