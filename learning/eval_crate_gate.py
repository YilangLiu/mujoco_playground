"""Free-rollout task-success gate for Go2 crate-climb policies.

The established acceptance test for this task: deterministic policy rollouts
from frame 0 (standing in front of the crate), NO deviation termination, NO
reset-to-reference — count rollouts that end with >=3 feet on the crate top.
Tracking reward has repeatedly masked task failure (parked policies collect
~73% of the reward cap), so this gate, not reward, decides success.

Seed 0 starts exactly on the reference initial state; the remaining seeds
add RSI-tier start noise (qpos 0.02 / qvel 0.05).

Run:
  python learning/eval_crate_gate.py \
      --checkpoint logs/<run>/checkpoints/<steps> \
      --reference references/go2_crate_climb_recovery_refs.npz \
      --video go2_crate_gate.mp4
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import argparse

import jax
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "high")

import jax.numpy as jp
import mediapy as media
import numpy as np
from brax.io import model

from mujoco_playground import registry
from mujoco_playground.experimental.learning import dva_train

ACTION_SCALE = [0.4, 1.9, 0.9] * 4


def make_cfg(reference: str, noisy: bool):
  cfg = registry.get_default_config("Go2SampleAPG")
  cfg["env"]["terrain"] = "crate"
  cfg["env"]["reset2ref"] = False
  cfg["env"]["reference_state_init"] = True
  cfg["env"]["reference_loop"] = False
  cfg["env"]["reference_path"] = reference
  cfg["env"]["impratio"] = 100
  cfg["Kp"] = 30.0
  cfg["Kd"] = 0.65
  cfg["ctrl_dt"] = 0.02
  cfg["episode_length"] = 130
  cfg["env"]["action_scale"] = ACTION_SCALE
  # The gate measures the free task: no deviation termination, no kicks.
  cfg["env"]["terminate_on_deviation"] = False
  cfg["pert_config"]["enable"] = False
  # Pin episode starts to frame 0.
  cfg["env"]["rsi_min_frame"] = 0
  cfg["env"]["rsi_max_frame"] = 1
  cfg["env"]["rsi_noise_qpos"] = 0.02 if noisy else 0.0
  cfg["env"]["rsi_noise_qvel"] = 0.05 if noisy else 0.0
  return cfg


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=str, required=True)
  p.add_argument(
      "--reference",
      type=str,
      default="references/go2_crate_climb_recovery_refs.npz",
  )
  p.add_argument("--seeds", type=int, default=8)
  p.add_argument("--steps", type=int, default=130)
  p.add_argument("--video", type=str, default="")
  args = p.parse_args()

  envs = {
      False: registry.load("Go2SampleAPG", make_cfg(args.reference, False)),
      True: registry.load("Go2SampleAPG", make_cfg(args.reference, True)),
  }
  env0 = envs[False]

  params = model.load_params(args.checkpoint)
  norm_actor = tuple(params[:2])  # accepts (norm, actor) or full 4-tuple
  actor = dva_train.ActorMLP(action_size=env0.action_size)
  policy = jax.jit(
      dva_train.make_policy_factory(actor, "state", True)(
          norm_actor, deterministic=True
      )
  )
  steps = {k: jax.jit(v.step) for k, v in envs.items()}
  resets = {k: jax.jit(v.reset) for k, v in envs.items()}

  feet_ids = jp.array([
      env0._mj_model.site(f"{nm}_foot").id for nm in ("FL", "FR", "RL", "RR")
  ])

  def feet_on(state) -> int:
    feet = np.asarray(state.data.site_xpos[feet_ids])
    return int(
        (
            (feet[:, 0] > 0.99)
            & (feet[:, 0] < 1.61)
            & (np.abs(feet[:, 1]) < 0.46)
            & (feet[:, 2] > 0.55)
        ).sum()
    )

  successes = 0
  video_traj = None
  for seed in range(args.seeds):
    noisy = seed > 0
    state = resets[noisy](jax.random.PRNGKey(seed))
    # All ensemble members share the nominal prefix, so frame-0 states are
    # identical across members; pin the tracked member for determinism.
    state.info["ref_idx"] = jp.zeros((), dtype=jp.int32)
    traj = [state]
    survived = args.steps
    for t in range(args.steps):
      action, _ = policy(state.obs, jax.random.PRNGKey(0))
      state = steps[noisy](state, action)
      traj.append(state)
      if bool(state.done):
        survived = t + 1
        break
    n_on = feet_on(state)
    ok = survived == args.steps and n_on >= 3
    successes += int(ok)
    torso = np.round(np.asarray(state.data.qpos[:3]), 2).tolist()
    print(
        f"seed {seed} ({'noisy' if noisy else 'exact'}): "
        f"{'SUCCESS' if ok else 'fail'} | survived {survived}/{args.steps} "
        f"| feet_on {n_on}/4 | final torso {torso}"
    )
    if seed == 0:
      video_traj = traj

  print(f"GATE: {successes}/{args.seeds} free rollouts succeed.")

  if args.video and video_traj is not None:
    frames = env0.render(
        video_traj[::2], height=480, width=640, camera="track"
    )
    media.write_video(args.video, frames, fps=1.0 / (env0.dt * 2))
    print(f"Saved seed-0 rollout video to {args.video}")


if __name__ == "__main__":
  main()
