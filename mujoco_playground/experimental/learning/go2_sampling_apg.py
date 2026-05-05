import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ['MUJOCO_GL'] = 'egl'

os.environ['JAX_LOG_COMPILES'] = '0'

import json
import time

import functools
from pathlib import Path

import jax.numpy as jp
import numpy as np
import jax
from mujoco import mjx
print("JAX Device:", jax.devices())
from jax import config # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')

print("jax.devices():", jax.devices())
print("local_device_count:", jax.local_device_count())

from absl import logging
logging.set_verbosity(logging.DEBUG)

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from brax.training.agents.apg import train as apg
# from apg_alg.algorithm import apg  # Local modified APG version # type: ignore
from brax.training.agents.apg import networks as apg_networks

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from brax.envs.wrappers import training as brax_training

from brax.training import acting

from brax.io import model

from brax import envs

import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output
from datetime import datetime
import mediapy as media
from tqdm import tqdm

import wandb

env_name = "Go2SampleAPG"
env_cfg = registry.get_default_config(env_name)
randomizer = registry.get_domain_randomizer(env_name)
REFERENCE_PATH = Path(
    os.environ.get(
        "GO2_SAMPLING_REFERENCE_PATH",
        "references/go2_seqjump_sampling_ref.npz",
    )
)
VALIDATE_REFERENCE = os.environ.get("GO2_VALIDATE_REFERENCE", "0") == "1"
REFERENCE_VIDEO_PATH = Path(
    os.environ.get("GO2_REFERENCE_VIDEO_PATH", "go2_seqjump_apg_reference.mp4")
)
LOGDIR = Path(os.environ.get("GO2_SAMPLING_LOGDIR", "logs"))
RUN_SUFFIX = os.environ.get("GO2_SAMPLING_RUN_SUFFIX")
MODEL_PATH_OVERRIDE = os.environ.get("GO2_SAMPLING_MODEL_PATH")

def render_rollout(reset_fn, step_fn, env, batch_size, inference_fn, n_step, render_every, seed=0):
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, batch_size) 
    state = reset_fn(rngs)
    rollout_list = [state]
    rewards = 0.0
    print("Starting rollout...")
    for i in range(n_step):
        act_rng, rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)
        # state.info['steps'] += 1
        if state.done:
            print(f"Early termination at step {i+1}")
            print(state.info)
            break
        rollout_list.append(state)
        rewards += state.reward
    print(f"Total reward: {float(rewards[0]):.1f}")
    final_state = state
    traj = rollout_list[::render_every]
    fps = 1.0 / (env.dt * render_every)
    frames = env.render(traj, height=480, width=640)
    media.show_video(frames, fps=fps, loop=True)

    return final_state


def validate_reference_motion(env, output_path: Path, render_every: int = 2) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = env.reset(jax.random.PRNGKey(0))
    data = state.data
    trajectory = []
    ref_steps = range(0, env.l_cycle, render_every)
    for i in tqdm(ref_steps, desc="Building reference video trajectory"):
        data = data.replace(
            qpos=env.kinematic_ref_qpos[i],
            qvel=env.kinematic_ref_qvel[i],
        )
        data = mjx.forward(env.mjx_model, data)
        trajectory.append(state.replace(data=data, reward=0.0, done=0.0))

    fps = 1.0 / (env.dt * render_every)
    print(f"Rendering and writing reference video to {output_path}...")
    frames = env.render(trajectory, height=480, width=640, camera="track")
    media.write_video(output_path.as_posix(), frames, fps=fps)
    print(
        f"Saved loaded reference validation video to {output_path} "
        f"({len(frames)} frames at {fps:.1f} FPS)."
    )

if __name__ == "__main__":
    apg_env_cfg = registry.get_default_config(env_name)
    apg_env_cfg['env']['reset2ref'] = False
    apg_env_cfg['env']['reference_state_init'] = True
    apg_env_cfg['env']['reference_path'] = REFERENCE_PATH.as_posix()
    apg_env_cfg['env']['reference_loop'] = False
    apg_env_cfg['pert_config']['enable'] = True
    apg_env_cfg['pert_config']['velocity_kick'] = [0.0, 3.0]
    apg_env_cfg['env']['impratio'] = 100
    if REFERENCE_PATH.exists():
        with np.load(REFERENCE_PATH) as ref:
            apg_env_cfg['episode_length'] = int(ref["qpos"].shape[0] - 1)
    else:
        raise FileNotFoundError(
            f"Sampling reference not found: {REFERENCE_PATH}. Generate it with "
            "`python learning/play_go2_sampling.py "
            "--reference_output_path references/go2_seqjump_sampling_ref.npz`."
        )

    env = registry.load(env_name, apg_env_cfg)
    eval_env = registry.load(env_name, apg_env_cfg)
    if VALIDATE_REFERENCE:
        validate_reference_motion(env, REFERENCE_VIDEO_PATH)

    # frames = env.play_ref_motion(render_every=1)
    # media.write_video("go2_trot_ref_motion.mp4", frames, fps=1.0 / (env.dt * 1))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{timestamp}"
    if RUN_SUFFIX is not None:
        exp_name += f"-{RUN_SUFFIX}"
    logdir = LOGDIR.resolve() / exp_name
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment name: {exp_name}")
    print(f"Logs are being stored in: {logdir}")
    print(f"Checkpoint path: {ckpt_path}")

    apg_params = locomotion_params.brax_apg_config(env_name)
    print(apg_params)

    # apg_params["policy_updates"] = 1500
    # # apg_params["num_envs"] = 512
    # apg_params["deterministic_eval"] = False
    # apg_params["num_envs"] = 32
    # apg_params["num_eval_envs"] = 1
    # apg_params["horizon_length"] = 20

    # print(apg_params)

    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(apg_env_cfg.to_dict(), fp, indent=4)

    apg_training_params = dict(apg_params)
    training_params_for_json = dict(apg_training_params)
    if "network_factory" in training_params_for_json:
        del training_params_for_json["network_factory"]
    with open(ckpt_path / "training_params.json", "w", encoding="utf-8") as fp:
        json.dump(training_params_for_json, fp, indent=4)

    network_factory = apg_networks.make_apg_networks
    if "network_factory" in apg_params:
        del apg_training_params["network_factory"]
        network_factory = functools.partial(
            apg_networks.make_apg_networks, 
            **apg_params.network_factory)

    train_fn = functools.partial(
        apg.train, **dict(apg_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer
        )

    wandb.init(project="mujoco-playground-apg-go2", name=exp_name)

    from datetime import datetime
    x_data_apg = []
    y_data_apg = []
    ydataerr_apg = []
    times_apg = [datetime.now()]

    num_timesteps = apg_params["policy_updates"] * apg_params["horizon_length"] * apg_params["num_envs"] * apg_params["action_repeat"]
    updates_per_epoch = round(apg_params["policy_updates"] / max(apg_params["num_evals"] - 1, 1))
    scale_it = updates_per_epoch * apg_params["horizon_length"] * apg_params["num_envs"] * apg_params["action_repeat"]

    def progress_apg(num_steps, metrics):

        times_apg.append(datetime.now())
        x_data_apg.append(num_steps * scale_it)
        y_data_apg.append(metrics["eval/episode_reward"])
        ydataerr_apg.append(metrics["eval/episode_reward_std"])
        wandb.log(metrics,step=num_steps * scale_it)

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress_apg
    )

    wandb.finish()

    num_timesteps = apg_params["policy_updates"] * apg_params["horizon_length"] * apg_params["num_envs"] * apg_params["action_repeat"]
    model_path = (
        Path(MODEL_PATH_OVERRIDE)
        if MODEL_PATH_OVERRIDE is not None
        else ckpt_path / str(num_timesteps)
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_params(model_path.as_posix(), params)
    print(f"Saved APG policy params to {model_path}.")