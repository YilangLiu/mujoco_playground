"""Run sampling-based inference for the Go2 sequential-jump environment."""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path
from typing import Dict, Sequence

# os.environ.setdefault("MUJOCO_GL", "egl")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
# os.environ.setdefault("JAX_LOG_COMPILES", "0")

import jax
import jax.numpy as jp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import mediapy as media
import numpy as np
from tqdm import tqdm

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_sampling


def rollout_us(
    step_env,
    state: mjx_env.State,
    us: jax.Array,
) -> jax.Array:
  def step_fn(carry: mjx_env.State, action: jax.Array):
    next_state = step_env(carry, action)
    return next_state, next_state.reward

  _, rewards = jax.lax.scan(step_fn, state, us)
  return rewards


class DialNodeController:
  """DIAL-MPC-style node trajectory controller for Playground states."""

  def __init__(
      self,
      env: go2_sampling.Go2Sampling,
      num_samples: int,
      hsample: int,
      hnode: int,
      temperature: float,
      horizon_diffuse_factor: float,
      traj_diffuse_factor: float,
  ):
    self.env = env
    self.nu = env.action_size
    self.num_samples = num_samples
    self.hsample = hsample
    self.hnode = hnode
    self.temperature = temperature
    self.traj_diffuse_factor = traj_diffuse_factor
    self.node_spline_order = 2 if hnode >= 2 else 1
    self.u_spline_order = 2 if hsample >= 2 else 1

    self.sigma_control = (
        horizon_diffuse_factor ** jp.arange(hnode + 1, dtype=jp.float32)[::-1]
    )
    self.step_us = jp.linspace(0.0, env.dt * hsample, hsample + 1)
    self.step_nodes = jp.linspace(0.0, env.dt * hsample, hnode + 1)

    self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
    self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
    self.node2u_vmap = jax.jit(jax.vmap(self.node2u, in_axes=1, out_axes=1))
    self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=1, out_axes=1))
    self.node2u_vvmap = jax.jit(jax.vmap(self.node2u_vmap, in_axes=0))

  @functools.partial(jax.jit, static_argnums=(0,))
  def node2u(self, nodes: jax.Array) -> jax.Array:
    spline = InterpolatedUnivariateSpline(
        self.step_nodes, nodes, k=self.node_spline_order
    )
    return spline(self.step_us)

  @functools.partial(jax.jit, static_argnums=(0,))
  def u2node(self, us: jax.Array) -> jax.Array:
    spline = InterpolatedUnivariateSpline(
        self.step_us, us, k=self.u_spline_order
    )
    return spline(self.step_nodes)

  @functools.partial(jax.jit, static_argnums=(0,))
  def reverse_once(
      self,
      state: mjx_env.State,
      rng: jax.Array,
      ybar: jax.Array,
      noise_scale: jax.Array,
  ) -> tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    rng, sample_rng = jax.random.split(rng)
    eps = jax.random.normal(
        sample_rng, (self.num_samples, self.hnode + 1, self.nu)
    )
    y0s = eps * noise_scale[None, :, None] + ybar
    y0s = y0s.at[:, 0].set(ybar[0])
    y0s = jp.concatenate([y0s, ybar[None]], axis=0)
    y0s = jp.clip(y0s, -1.0, 1.0)

    us = self.node2u_vvmap(y0s)
    rewardss = self.rollout_us_vmap(state, us)
    scores = rewardss.mean(axis=-1)
    nominal_score = scores[-1]
    score_std = jp.maximum(scores.std(), 1e-6)
    logp = (scores - nominal_score) / score_std / self.temperature
    weights = jax.nn.softmax(logp)
    ybar = jp.einsum("n,nij->ij", weights, y0s)
    info = {
        "scores": scores,
        "best_score": jp.max(scores),
        "mean_score": jp.mean(scores),
        "nominal_score": nominal_score,
    }
    return rng, ybar, info

  @functools.partial(jax.jit, static_argnums=(0,))
  def shift(self, ybar: jax.Array) -> jax.Array:
    us = self.node2u_vmap(ybar)
    us = jp.roll(us, -1, axis=0)
    us = us.at[-1].set(jp.zeros(self.nu))
    return self.u2node_vmap(us)


def render_rollout(
    env: go2_sampling.Go2Sampling,
    states: Sequence[mjx_env.State],
    output_path: Path,
    render_every: int,
    width: int,
    height: int,
) -> None:
  traj = list(states)[::render_every]
  fps = 1.0 / (env.dt * render_every)
  frames = env.render(traj, height=height, width=width, camera="track")
  media.write_video(output_path.as_posix(), frames, fps=fps)
  print(f"Saved rollout video to {output_path} ({len(frames)} frames at {fps:.1f} FPS).")


def save_reference_rollout(
    env: go2_sampling.Go2Sampling,
    states: Sequence[mjx_env.State],
    actions: Sequence[jax.Array],
    rewards: Sequence[jax.Array],
    output_path: Path,
) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  qpos = np.asarray(jax.device_get(jp.stack([state.data.qpos for state in states])))
  qvel = np.asarray(jax.device_get(jp.stack([state.data.qvel for state in states])))
  base_xpos = np.asarray(
      jax.device_get(jp.stack([state.data.xpos[env._torso_body_id] for state in states]))
  )
  feet_xpos = np.asarray(
      jax.device_get(jp.stack([state.data.site_xpos[env._feet_site_ids] for state in states]))
  )
  action_arr = (
      np.asarray(jax.device_get(jp.stack(actions)))
      if actions
      else np.zeros((0, env.action_size), dtype=np.float32)
  )
  reward_arr = (
      np.asarray(jax.device_get(jp.stack(rewards)))
      if rewards
      else np.zeros((0,), dtype=np.float32)
  )
  np.savez(
      output_path,
      qpos=qpos,
      qvel=qvel,
      actions=action_arr,
      rewards=reward_arr,
      base_xpos=base_xpos,
      feet_xpos=feet_xpos,
      dt=np.array(env.dt),
  )
  print(
      "Saved APG reference rollout to "
      f"{output_path} (qpos={qpos.shape}, qvel={qvel.shape})."
  )


def run(args: argparse.Namespace) -> mjx_env.State:
  env_cfg = go2_sampling.default_config()
  env_cfg.impl = args.impl
  env_cfg.episode_length = args.num_steps + 1
  env = go2_sampling.Go2Sampling(config=env_cfg)

  reset_env = jax.jit(env.reset)
  step_env = jax.jit(env.step)

  rng = jax.random.PRNGKey(args.seed)
  rng, reset_rng = jax.random.split(rng)
  state = reset_env(reset_rng)
  controller = DialNodeController(
      env=env,
      num_samples=args.num_samples,
      hsample=args.hsample,
      hnode=args.hnode,
      temperature=args.temperature,
      horizon_diffuse_factor=args.horizon_diffuse_factor,
      traj_diffuse_factor=args.traj_diffuse_factor,
  )

  def reverse_scan(carry, factor):
    rng, ybar, state = carry
    rng, ybar, info = controller.reverse_once(state, rng, ybar, factor)
    return (rng, ybar, state), info

  ybar = jp.zeros((args.hnode + 1, env.action_size))
  rollout = [state]
  rews = []
  actions = []
  print(
      "Starting Go2 sequential-jump sampling rollout "
      f"(steps={args.num_steps}, samples={args.num_samples}, "
      f"Hsample={args.hsample}, Hnode={args.hnode})."
  )
  with tqdm(range(args.num_steps), desc="Rollout") as pbar:
    for t in pbar:
      action = ybar[0]
      state = step_env(state, action)
      rollout.append(state)
      rews.append(state.reward)
      actions.append(action)

      ybar = controller.shift(ybar)

      n_diffuse = args.num_diffuse_init if t == 0 else args.num_diffuse
      if t == 0:
        print("Performing JIT on DIAL-MPC")

      t0 = time.time()
      factors = controller.sigma_control * (
          args.traj_diffuse_factor
          ** jp.arange(n_diffuse, dtype=jp.float32)[:, None]
      )
      (rng, ybar, _), _ = jax.lax.scan(
          reverse_scan, (rng, ybar, state), factors
      )
      freq = 1.0 / (time.time() - t0)
      pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

  total_reward = float(jp.stack(rews).sum())
  print(f"Total rollout reward: {total_reward:.3f}")
  render_rollout(
      env=env,
      states=rollout,
      output_path=Path(args.output_path),
      render_every=args.render_every,
      width=args.width,
      height=args.height,
  )
  if args.reference_output_path:
    save_reference_rollout(
        env=env,
        states=rollout,
        actions=actions,
        rewards=rews,
        output_path=Path(args.reference_output_path),
    )
  return state


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_steps", type=int, default=300)
  parser.add_argument("--num_samples", type=int, default=2048)
  parser.add_argument("--hsample", type=int, default=20)
  parser.add_argument("--hnode", type=int, default=5)
  parser.add_argument("--num_diffuse", type=int, default=2)
  parser.add_argument("--num_diffuse_init", type=int, default=10)
  parser.add_argument("--temperature", type=float, default=0.05)
  parser.add_argument("--horizon_diffuse_factor", type=float, default=0.9)
  parser.add_argument("--traj_diffuse_factor", type=float, default=0.5)
  parser.add_argument("--render_every", type=int, default=2)
  parser.add_argument("--width", type=int, default=1920)
  parser.add_argument("--height", type=int, default=1080)
  parser.add_argument("--impl", type=str, default="jax", choices=("jax", "warp"))
  parser.add_argument("--output_path", type=str, default="go2_seq_jump_sampling.mp4")
  parser.add_argument(
      "--reference_output_path",
      type=str,
      default="",
      help="Optional .npz path for exporting qpos/qvel as an APG reference.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  final_state = run(parse_args())
  print("Final metrics:", {k: float(v) for k, v in final_state.metrics.items()})
