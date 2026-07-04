"""Reverse-RSI curriculum for the Go2 crate climb with the DVA trainer.

Top-ranked option of the 2026-07-04 design study: grow the mastered suffix of
the reference BACKWARD from the goal. Each stage restricts reference-state
init to a window [lo, lo+16) of the 131-frame training-plant reference and
carries the full training state (actor + critic + target critic + normalizer)
into the next-earlier stage, so the target-critic terminal bootstrap
gamma^H * V(s_H) is already truthful in the region every earlier stage lands
in. The reference tube is widened to nonzero measure with RSI injection
noise. Deviation termination stays on throughout (it keeps V honest).

Reference phases (from go2_crate_climb_pg_sampling_ref.npz):
  t 0-12 approach/crouch, t 13-23 ballistic flight, t 24 front-feet catch,
  t 24-58 mantle, t 59-130 all four feet on the crate.

Run:
  python mujoco_playground/experimental/learning/dva_crate_curriculum.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["MUJOCO_GL"] = "egl"

import json
import time
from datetime import datetime
from pathlib import Path

import jax
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "high")

import jax.numpy as jp
import numpy as np
from brax.io import model
from mujoco import mjx

from mujoco_playground import registry
from mujoco_playground.experimental.learning import dva_train

ENV_NAME = "Go2SampleAPG"
REF_PATH = "references/go2_crate_climb_pg_sampling_ref.npz"
ACTION_SCALE = [0.4, 1.9, 0.9] * 4
EPISODE_LENGTH = 130
RSI_WINDOW = 16
RSI_NOISE_QPOS = 0.02
RSI_NOISE_QVEL = 0.05
# Round 2: resume from the round-1 stage-1 checkpoint (lo=59 was mastered —
# in-stage evals ran full-length; round 1's promo failures were a stale-obs
# eval bug, now fixed). Dense 3-4 frame steps across the launch (13-24).
STAGES = [44, 34, 27, 24, 21, 18, 15, 12, 8, 4, 0]
EPOCHS_PER_STAGE = 400
PROMO_SEEDS = 8
PROMO_PASS = 6
MAX_STAGE_REPEATS = 2
INIT_ACTOR = os.environ.get(
    "CRATE_CURRICULUM_INIT",
    "logs/Go2SampleAPG-crate_climb-dva-20260704-010157-curriculum/"
    "checkpoints/stage01_lo59",
)

DVA_OVERRIDES = dict(
    episode_length=EPISODE_LENGTH,
    max_epochs=EPOCHS_PER_STAGE,
    num_evals=2,
    num_eval_envs=64,
    actor_lr=5e-4,
    actor_logstd_init=-2.5,
)


def make_cfg(rsi_lo: int, rsi_hi: int):
  cfg = registry.get_default_config(ENV_NAME)
  cfg["env"]["terrain"] = "crate"
  cfg["env"]["reset2ref"] = False
  cfg["env"]["reference_state_init"] = True
  cfg["env"]["reference_loop"] = False
  cfg["env"]["reference_path"] = REF_PATH
  cfg["env"]["impratio"] = 100
  cfg["Kp"] = 30.0
  cfg["Kd"] = 0.65
  cfg["ctrl_dt"] = 0.02
  cfg["episode_length"] = EPISODE_LENGTH
  cfg["env"]["action_scale"] = ACTION_SCALE
  cfg["env"]["terminate_on_deviation"] = True
  cfg["env"]["rsi_min_frame"] = int(rsi_lo)
  cfg["env"]["rsi_max_frame"] = int(rsi_hi)
  cfg["env"]["rsi_noise_qpos"] = RSI_NOISE_QPOS
  cfg["env"]["rsi_noise_qvel"] = RSI_NOISE_QVEL
  cfg["pert_config"]["enable"] = False
  return cfg


def pinned_success(eval_env, step_fn, policy, lo: int, n_seeds: int) -> int:
  """Deterministic rollouts injected at frame lo (+ tube noise); count
  rollouts that reach the reference end with >=3 feet in the crate window."""
  feet_ids = jp.array([
      eval_env._mj_model.site(f"{nm}_foot").id
      for nm in ("FL", "FR", "RL", "RR")
  ])
  horizon = eval_env.l_cycle - 1 - lo
  passes = 0
  for seed in range(n_seeds):
    rng = jax.random.PRNGKey(1000 + seed)
    state = jax.jit(eval_env.reset)(rng)
    state.info["step"] = jp.asarray(float(lo))
    state.info["init_step"] = jp.asarray(float(lo))
    nq_rng, nv_rng = jax.random.split(jax.random.fold_in(rng, 7))
    qpos = eval_env.kinematic_ref_qpos[lo]
    qpos = qpos.at[7:].add(
        RSI_NOISE_QPOS * jax.random.normal(nq_rng, (12,), dtype=qpos.dtype)
    )
    qvel = eval_env.kinematic_ref_qvel[lo] + RSI_NOISE_QVEL * (
        jax.random.normal(nv_rng, (18,), dtype=qpos.dtype)
    )
    data = mjx.forward(
        eval_env.mjx_model, state.data.replace(qpos=qpos, qvel=qvel)
    )
    # Mirror the env reset's penetration fix: noisy joint angles can start
    # feet a couple of cm inside the crate, and under impratio=100 that
    # first-contact impulse throws the robot off the box. Clamped to
    # dist < 0 — the unclamped min is a positive gap for airborne frames
    # and teleported flight-phase starts down to the ground.
    pen = jp.where(
        data.ncon > 0,
        jp.minimum(jp.min(data._impl.contact.dist), 0.0),
        0.0,
    )
    qpos = qpos.at[2].set(qpos[2] - pen)
    data = mjx.forward(
        eval_env.mjx_model, state.data.replace(qpos=qpos, qvel=qvel)
    )
    # Recompute the observation for the injected state — replacing data
    # alone leaves the reset-state obs, and the first action from a stale
    # obs is fatal at delicate flight/landing starts (this bug produced
    # false 0/8 promotions in curriculum round 1).
    obs = eval_env._get_obs(data, state.info)
    state = state.replace(data=data, obs=obs)
    dead = False
    for _ in range(int(horizon)):
      a, _ = policy(state.obs, jax.random.PRNGKey(0))
      state = step_fn(state, a)
      if bool(state.done):
        dead = True
        break
    feet = np.asarray(state.data.site_xpos[feet_ids])
    on = (
        (feet[:, 0] > 0.99)
        & (feet[:, 0] < 1.61)
        & (np.abs(feet[:, 1]) < 0.46)
        & (feet[:, 2] > 0.55)
    ).sum()
    if not dead and on >= 3:
      passes += 1
  return passes


def main():
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir = Path("logs") / f"Go2SampleAPG-crate_climb-dva-{timestamp}-curriculum"
  ckpt_dir = logdir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  print(f"Curriculum run dir: {logdir}")
  with open(logdir / "curriculum.json", "w", encoding="utf-8") as fp:
    json.dump(
        dict(stages=STAGES, epochs=EPOCHS_PER_STAGE, rsi_window=RSI_WINDOW,
             noise_qpos=RSI_NOISE_QPOS, noise_qvel=RSI_NOISE_QVEL,
             dva=DVA_OVERRIDES, init=INIT_ACTOR),
        fp, indent=2,
    )

  # Shared eval env (deviation termination ON so promo detects divergence).
  eval_cfg = make_cfg(0, 1)
  eval_env = registry.load(ENV_NAME, eval_cfg)
  eval_step = jax.jit(eval_env.step)

  prev_state = model.load_params(INIT_ACTOR)  # (norm, actor) 2-tuple
  actor = dva_train.ActorMLP(action_size=eval_env.action_size)

  history = []
  for stage_i, lo in enumerate(STAGES):
    hi = min(lo + RSI_WINDOW, eval_env.l_cycle - 2)
    for attempt in range(1 + MAX_STAGE_REPEATS):
      t0 = time.time()
      print(
          f"=== stage {stage_i} (lo={lo}, window [{lo},{hi}), "
          f"attempt {attempt + 1}) ==="
      )
      env = registry.load(ENV_NAME, make_cfg(lo, hi))

      def progress(it, metrics, _lo=lo, _st=stage_i):
        print(
            f"  [stage {_st} lo={_lo} eval {it}] "
            f"episode_reward={metrics['eval/episode_reward']:.1f} "
            f"len={metrics.get('eval/avg_episode_length', float('nan')):.1f}"
        )

      _, params, _, full_state = dva_train.train(
          environment=env,
          eval_env=env,
          progress_fn=progress,
          restore_params=prev_state,
          return_training_state=True,
          **DVA_OVERRIDES,
      )
      prev_state = full_state
      policy = jax.jit(
          dva_train.make_policy_factory(actor, "state", True)(
              params, deterministic=True
          )
      )
      passes = pinned_success(eval_env, eval_step, policy, lo, PROMO_SEEDS)
      dt_min = (time.time() - t0) / 60
      print(
          f"  stage {stage_i} lo={lo}: promo {passes}/{PROMO_SEEDS} "
          f"({dt_min:.1f} min)"
      )
      history.append(dict(stage=stage_i, lo=lo, attempt=attempt + 1,
                          promo=f"{passes}/{PROMO_SEEDS}"))
      model.save_params(
          (ckpt_dir / f"stage{stage_i:02d}_lo{lo}").as_posix(), full_state
      )
      if passes >= PROMO_PASS:
        break
      if attempt < MAX_STAGE_REPEATS:
        print(f"  stage {stage_i} below promotion bar — repeating once")
    else:
      print(f"  WARNING stage {stage_i} never promoted; continuing anyway")

  with open(logdir / "history.json", "w", encoding="utf-8") as fp:
    json.dump(history, fp, indent=2)
  print("Curriculum finished. History:")
  for h in history:
    print(" ", h)
  print(f"Final full state: {ckpt_dir}/stage{len(STAGES)-1:02d}_lo0")


if __name__ == "__main__":
  main()
