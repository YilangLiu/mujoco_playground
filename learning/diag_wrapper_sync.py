"""Regression test: reference index must stay synced across autoresets.

BraxAutoResetWrapper (full_reset=False) restores the cached reset pose on
done but does not reset env info. Go2SampleAPG.step therefore resyncs its
private reference counter from info["init_step"] whenever the wrapper's
episode "steps" counter reads 0 (the first step after any autoreset).

Invariant checked after every wrapped step:
    info["step"] == info["init_step"] + info["steps"]
(EpisodeWrapper increments "steps" after env.step, so both counters advance
in lockstep within an episode; on autoreset both must restart together.)

Run on CPU: JAX_PLATFORMS=cpu python learning/diag_wrapper_sync.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jp
import numpy as np

from mujoco_playground import registry
from mujoco_playground._src import wrapper

EPISODE_LENGTH = 10
NUM_ENVS = 2
NUM_STEPS = 35  # crosses three truncation boundaries


def main():
  cfg = registry.get_default_config("Go2SampleAPG")
  cfg["env"]["reset2ref"] = False
  cfg["env"]["reference_state_init"] = True
  cfg["env"]["reference_path"] = "references/go2_trot_sampling_ref_pg.npz"
  cfg["env"]["reference_loop"] = True
  cfg["pert_config"]["enable"] = False

  env = registry.load("Go2SampleAPG", cfg)
  wrapped = wrapper.wrap_for_brax_training(
      env, episode_length=EPISODE_LENGTH, action_repeat=1
  )
  reset_fn = jax.jit(wrapped.reset)
  step_fn = jax.jit(wrapped.step)

  rng = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
  state = reset_fn(rng)
  init_step = np.asarray(state.info["init_step"])
  print(f"init_step per env: {init_step}")
  assert not np.all(init_step == 0), "RSI should give nonzero init frames"

  actions = jp.zeros((NUM_ENVS, env.action_size))
  n_resets = 0
  for t in range(NUM_STEPS):
    state = step_fn(state, actions)
    step = np.asarray(state.info["step"])
    steps = np.asarray(state.info["steps"])
    init = np.asarray(state.info["init_step"])
    ok = np.allclose(step, init + steps)
    if not ok:
      raise AssertionError(
          f"DESYNC at t={t}: step={step}, init_step={init}, steps={steps} "
          f"(expected step == init_step + steps)"
      )
    if np.any(np.asarray(state.done) > 0) or np.any(steps == EPISODE_LENGTH):
      n_resets += 1
    # Phase obs (last two dims) must be a unit-circle embedding.
    phase = np.asarray(state.obs[:, -2:])
    assert np.all(np.abs(phase) <= 1.0 + 1e-9), f"phase out of range: {phase}"
    assert np.allclose((phase**2).sum(-1), 1.0, atol=1e-6), (
        f"phase not on unit circle: {phase}"
    )

  assert n_resets >= 3, f"test never crossed a truncation ({n_resets})"
  obs_dim = state.obs.shape[-1]
  print(f"obs dim: {obs_dim}")
  print(f"PASS: {NUM_STEPS} steps, {n_resets} truncation boundaries, "
        "reference index stayed synced; phase bounded on unit circle.")


if __name__ == "__main__":
  main()
