"""Behavior-clone the sampled crate-climb expert into a DVA actor warm start.

Builds (obs, action) pairs along the training-plant reference
(go2_crate_climb_pg_sampling_ref.npz): obs come from Go2SampleAPG._get_obs
evaluated at the reference states with the correct step index and
last_action; targets are the expert actions converted from the sampler's
act2joint ranges into the training policy's action units
(a = (target - action_loc) / action_scale, then arctanh for the pre-tanh mu).

Output: <out>.warmstart file containing (RunningMeanStd, actor_params) in the
dva_train checkpoint format, usable via GO2_SAMPLING_DVA_RESTORE.

Run: JAX_ENABLE_X64=1 python learning/bc_crate_warmstart.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jp
import numpy as np
import optax
from brax.io import model
from mujoco import mjx

from mujoco_playground import registry
from mujoco_playground.experimental.learning import dva_train

REF_PATH = "references/go2_crate_climb_pg_sampling_ref.npz"
OUT_PATH = "references/go2_crate_climb_bc.warmstart"
ACTION_SCALE = [0.4, 1.9, 0.9] * 4
# Sampler act2joint ranges (playground leg order FL, FR, RL, RR).
SAMPLER_LO = np.array([-0.25, -1.0, -2.7] * 2 + [-0.25, 0.0, -2.7] * 2)
SAMPLER_HI = np.array([0.25, 1.4, -1.0] * 2 + [0.25, 1.8, -1.0] * 2)
BC_STEPS = 3000
BC_LR = 1e-3


def main():
  ref = dict(np.load(REF_PATH))
  n = ref["actions"].shape[0]  # 130 actions for 131 states

  cfg = registry.get_default_config("Go2SampleAPG")
  cfg["env"]["terrain"] = "crate"
  cfg["env"]["reset2ref"] = False
  cfg["env"]["reference_state_init"] = True
  cfg["env"]["reference_loop"] = False
  cfg["env"]["reference_path"] = REF_PATH
  cfg["env"]["impratio"] = 100
  cfg["Kp"] = 30.0
  cfg["Kd"] = 0.65
  cfg["ctrl_dt"] = 0.02
  cfg["env"]["action_scale"] = ACTION_SCALE
  cfg["pert_config"]["enable"] = False
  env = registry.load("Go2SampleAPG", cfg)

  loc = np.asarray(env.action_loc)
  scale = np.asarray(env.action_scale)

  # Expert joint targets -> policy action units (within +-1 by construction).
  targets = SAMPLER_LO + (ref["actions"] + 1.0) / 2.0 * (SAMPLER_HI - SAMPLER_LO)
  a_pol = np.clip((targets - loc) / scale, -0.999, 0.999)
  mu_tgt = np.arctanh(a_pol)  # pre-tanh regression targets

  # Observations at reference states t = 0..n-1. last_action in the obs is
  # the ctrl (motor targets) of the PREVIOUS step, matching env.step.
  state = jax.jit(env.reset)(jax.random.PRNGKey(0))
  obs_list = []
  data = state.data
  for t in range(n):
    data = data.replace(
        qpos=env.kinematic_ref_qpos[t], qvel=env.kinematic_ref_qvel[t]
    )
    data = mjx.forward(env.mjx_model, data)
    info = dict(state.info)
    info["step"] = jp.asarray(float(t))
    prev_ctrl = (
        loc + a_pol[t - 1] * scale if t > 0 else np.asarray(env.action_loc)
    )
    info["last_action"] = jp.asarray(prev_ctrl, dtype=env._dtype)
    obs_list.append(np.asarray(env._get_obs(data, info)))
  obs = np.stack(obs_list)  # (n, 53)
  print(f"BC dataset: obs {obs.shape}, targets {mu_tgt.shape}")

  # Normalizer fit on the BC obs (count scaled up so early training rollouts
  # don't immediately wash it out).
  norm = dva_train.RunningMeanStd(
      mean=jp.asarray(obs.mean(0), dtype=jp.float32),
      var=jp.asarray(obs.var(0) + 1e-4, dtype=jp.float32),
      count=jp.asarray(float(n * 100), dtype=jp.float32),
  )
  obs_n = np.asarray(dva_train.rms_normalize(norm, jp.asarray(obs))).astype(
      np.float32
  )

  actor = dva_train.ActorMLP(action_size=env.action_size)
  params = actor.init(jax.random.PRNGKey(1), jp.zeros((53,), jp.float32))

  def loss_fn(p):
    mu, _ = actor.apply(p, jp.asarray(obs_n))
    return jp.mean((mu - jp.asarray(mu_tgt, dtype=jp.float32)) ** 2)

  opt = optax.adam(BC_LR)
  opt_state = opt.init(params)

  @jax.jit
  def train_step(p, s):
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, s = opt.update(grads, s)
    return optax.apply_updates(p, updates), s, loss

  for i in range(BC_STEPS):
    params, opt_state, loss = train_step(params, opt_state)
    if i % 500 == 0 or i == BC_STEPS - 1:
      print(f"  bc step {i}: mse {float(loss):.5f}")

  model.save_params(OUT_PATH, (norm, jax.device_get(params)))
  print(f"Saved BC warm start to {OUT_PATH}")

  # Closed-loop sanity: how far does the BC policy get on its own?
  policy = jax.jit(
      dva_train.make_policy_factory(actor, "state", True)(
          (norm, params), deterministic=True
      )
  )
  step = jax.jit(env.step)
  s = jax.jit(env.reset)(jax.random.PRNGKey(0))
  s.info["step"] = jp.zeros_like(s.info["step"])
  s.info["init_step"] = jp.zeros_like(s.info["init_step"])
  d0 = s.data.replace(qpos=env.kinematic_ref_qpos[0], qvel=env.kinematic_ref_qvel[0])
  s = s.replace(data=mjx.forward(env.mjx_model, d0))
  for t in range(n):
    a, _ = policy(s.obs, jax.random.PRNGKey(0))
    s = step(s, a)
  feet = np.asarray(
      s.data.site_xpos[
          jp.array([env._mj_model.site(f"{nm}_foot").id for nm in ("FL", "FR", "RL", "RR")])
      ]
  )
  on = (
      (feet[:, 0] > 0.99)
      & (feet[:, 0] < 1.61)
      & (np.abs(feet[:, 1]) < 0.46)
      & (feet[:, 2] > 0.55)
  ).sum()
  print(
      f"BC closed-loop rollout: final torso "
      f"{np.round(np.asarray(s.data.qpos[:3]), 2).tolist()}, "
      f"feet_on_crate {int(on)}/4"
  )


if __name__ == "__main__":
  main()
