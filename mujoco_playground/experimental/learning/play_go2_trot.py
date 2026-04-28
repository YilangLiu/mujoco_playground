import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ['MUJOCO_GL'] = 'egl'

os.environ['JAX_LOG_COMPILES'] = '0'

import time

import functools

import jax.numpy as jp
import numpy as np
import jax
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

env_name = "Go2Trot"
demo_cfg = registry.get_default_config(env_name)
demo_cfg['env']['reset2ref'] = False
demo_cfg['env']['reference_state_init'] = False
demo_cfg['pert_config']['enable'] = False
demo_cfg['env']['impratio'] = 100
demo_env = registry.load(env_name, demo_cfg)
demo_env = brax_training.VmapWrapper(demo_env)
demo_step_fn = jax.jit(demo_env.step)
demo_reset_fn = jax.jit(demo_env.reset)

apg_params = locomotion_params.brax_apg_config(env_name)
params = model.load_params('/tmp/trotting_apg_2hz_policy')
params = (params[0], params[1])
network_factory = apg_networks.make_apg_networks
apg_training_params = dict(apg_params)
network_factory = apg_networks.make_apg_networks
if "network_factory" in apg_params:
    del apg_training_params["network_factory"]
    network_factory = functools.partial(
        apg_networks.make_apg_networks, 
        **apg_params.network_factory)

# from brax.training.acme import running_statistics
normalize = lambda x, y: x

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
    media.write_video("go2_eval_motion.mp4", frames, fps=fps)

    return final_state

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

if apg_params['normalize_observations']:
    # normalize = running_statistics.normalize
    mean, std = params[0].mean, params[0].std
    normalize = make_normalize_fn(mean, std)
apg_network = network_factory(
    demo_env.observation_size, demo_env.action_size, preprocess_observations_fn=normalize
)
make_inference_fn = apg_networks.make_inference_fn(apg_network)

jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
final_state = render_rollout(
    demo_reset_fn, demo_step_fn, demo_env, 1,
    jit_inference_fn, n_step=1000, render_every=1, seed=42)