"""Harvest DIAL-MPC recovery demonstrations around the nominal crate climb.

The single sampled reference (go2_crate_climb_pg_sampling_ref.npz) contains
zero information about recovering once the robot drifts off it — the expert
(DIAL-MPC) only survives the ballistic launch by replanning every step. This
script materializes that feedback as data: for each start frame f in a
launch-dense schedule, perturb the nominal reference state with RSI-style
Gaussian noise, then run the DIAL-MPC node controller CLOSED-LOOP on the
training plant (Go2CrateClimbPG) from that state to the end of the
reference, warm-started from the nominal action plan. Rollouts that end on
top of the crate are grafted onto the nominal prefix (frames 0..f-1) to
form full-length alternative references, giving the reference tube nonzero
measure with demonstrated corrections.

Output npz (multi-reference format understood by Go2SampleAPG):
  qpos (M, T, nq), qvel (M, T, nv), actions (M, T-1, nu) [sampler act2joint
  units], start_frame (M,), noise_qpos (M,), noise_qvel (M,),
  final_feet_on (M,), final_head_err (M,), dt.
Member 0 is the nominal reference itself. References from this plant are in
native playground joint order — never run fix_reference_leg_order on them.

Run:
  python learning/harvest_crate_recoveries.py --smoke   # plumbing + timing
  python learning/harvest_crate_recoveries.py           # full harvest
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import argparse
import json
import sys
import time
from pathlib import Path

import jax
from jax import config

config.update("jax_enable_x64", True)  # f32 NaNs under impratio=100

import jax.numpy as jp
import mediapy as media
import numpy as np
from mujoco import mjx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_go2_sampling import DialNodeController  # noqa: E402

from mujoco_playground._src.locomotion.go2 import go2_sampling  # noqa: E402

NOMINAL_REF = "references/go2_crate_climb_pg_sampling_ref.npz"
OUT_PATH = "references/go2_crate_climb_recovery_refs.npz"

# (sigma_qpos_joints, sigma_qvel) start-noise tiers, interleaved across each
# batch. Tier 0 matches the training env's RSI injection noise; tier 1 is
# wider so the demonstrated funnel extends beyond the training tube.
NOISE_TIERS = ((0.02, 0.05), (0.05, 0.15))


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument(
      "--frames",
      type=int,
      nargs="+",
      default=[4, 8, 12, 15, 18, 21, 24, 28],
      help=(
          "Start frames to harvest from (reference phases: 0-12 approach,"
          " 13-23 flight, 24 catch, 24-58 mantle)."
      ),
  )
  p.add_argument(
      "--batch",
      type=int,
      default=16,
      help="Recovery attempts per frame (noise tiers interleaved).",
  )
  p.add_argument(
      "--num_samples",
      type=int,
      default=127,
      help=(
          "DIAL-MPC samples per attempt per reverse iteration. batch *"
          " (num_samples + 1) parallel rollouts must fit on the GPU; the"
          " nominal reference run used 2049."
      ),
  )
  p.add_argument("--hsample", type=int, default=25)
  p.add_argument("--hnode", type=int, default=5)
  p.add_argument("--num_diffuse", type=int, default=2)
  p.add_argument("--num_diffuse_init", type=int, default=6)
  p.add_argument("--temperature", type=float, default=0.05)
  p.add_argument(
      "--sigma_scale",
      type=float,
      default=1.0,
      help="Multiplier on the reverse-iteration noise schedule.",
  )
  p.add_argument("--traj_diffuse_factor", type=float, default=0.5)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", type=str, default=OUT_PATH)
  p.add_argument(
      "--render_n",
      type=int,
      default=2,
      help="Render this many kept recoveries to mp4 for inspection.",
  )
  p.add_argument(
      "--smoke",
      action="store_true",
      help="Tiny run (one frame, batch 4, N 63): checks plumbing + timing.",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  if args.smoke:
    args.frames = [15]
    args.batch = 4
    args.num_samples = 63
    args.num_diffuse_init = 4
    args.render_n = 0
    args.out = "references/go2_crate_climb_recovery_refs_smoke.npz"

  ref = dict(np.load(NOMINAL_REF))
  nom_qpos = np.asarray(ref["qpos"], dtype=np.float64)  # (T, nq)
  nom_qvel = np.asarray(ref["qvel"], dtype=np.float64)  # (T, nv)
  nom_actions = np.asarray(ref["actions"], dtype=np.float64)  # (T-1, nu)
  T = nom_qpos.shape[0]
  assert nom_actions.shape[0] == T - 1, (
      f"actions/state length mismatch: {nom_actions.shape[0]} vs {T - 1}"
  )
  if max(args.frames) + args.hsample + 1 > nom_actions.shape[0]:
    print("Note: late start frames pad the warm-start plan with the last "
          "nominal action.")

  cfg = go2_sampling.default_crate_climb_pg_config()
  cfg.episode_length = 4 * T  # done is unused here; keep it out of the way
  env = go2_sampling.Go2CrateClimbPG(config=cfg)

  controller = DialNodeController(
      env=env,
      num_samples=args.num_samples,
      hsample=args.hsample,
      hnode=args.hnode,
      temperature=args.temperature,
      horizon_diffuse_factor=1.0,  # crate yaml setting
      traj_diffuse_factor=args.traj_diffuse_factor,
  )

  template = jax.jit(env.reset)(jax.random.PRNGKey(0))
  # env.reset emits f64 done/metrics under x64 while env.step emits f32 for
  # some leaves; lax.scan inside the planner needs carry dtypes to match, so
  # cast start states to step's exact output dtype tree.
  _step_shapes = jax.eval_shape(
      env.step, template, jp.zeros(env.action_size, dtype=jp.float64)
  )

  def cast_like_step(state):
    return jax.tree_util.tree_map(
        lambda x, s: jp.asarray(x, s.dtype), state, _step_shapes
    )

  def make_start(qpos, qvel, step):
    data = template.data.replace(qpos=qpos, qvel=qvel)
    data = mjx.forward(env.mjx_model, data)
    # Mirror the training env's reset penetration fix: noisy joints can put
    # feet a couple of cm inside the ground/crate, and under impratio=100
    # the first-contact impulse throws the robot. Clamped to dist < 0: MJX's
    # contact.dist is signed for ALL candidate pairs, so the unclamped min
    # is a positive gap for airborne states and would teleport flight-frame
    # starts down to contact.
    pen = jp.where(
        data.ncon > 0,
        jp.minimum(jp.min(data._impl.contact.dist), 0.0),
        0.0,
    )
    qpos = qpos.at[2].add(-pen)
    data = mjx.forward(
        env.mjx_model, template.data.replace(qpos=qpos, qvel=qvel)
    )
    info = dict(template.info)
    info["step"] = jp.asarray(step, dtype=template.info["step"].dtype)
    obs = env._get_obs(data, info)
    return cast_like_step(template.replace(data=data, obs=obs, info=info))

  vmake_start = jax.jit(jax.vmap(make_start, in_axes=(0, 0, None)))
  vstep = jax.jit(jax.vmap(env.step))
  vshift = jax.jit(jax.vmap(controller.shift))

  def plan(state, rng, ybar, n_diffuse):
    factors = args.sigma_scale * controller.sigma_control * (
        args.traj_diffuse_factor
        ** jp.arange(n_diffuse, dtype=jp.float32)[:, None]
    )

    def body(carry, factor):
      rng, ybar = carry
      rng, ybar, _ = controller.reverse_once(state, rng, ybar, factor)
      return (rng, ybar), None

    (rng, ybar), _ = jax.lax.scan(body, (rng, ybar), factors)
    return rng, ybar

  vplan = jax.jit(
      jax.vmap(plan, in_axes=(0, 0, 0, None)), static_argnums=(3,)
  )

  feet_ids = env._feet_site_ids  # FL, FR, RL, RR (playground order)

  def gate_stats(state):
    feet = state.data.site_xpos[feet_ids]
    on = jp.sum(
        (feet[:, 0] > 0.99)
        & (feet[:, 0] < 1.61)
        & (jp.abs(feet[:, 1]) < 0.46)
        & (feet[:, 2] > 0.55)
    )
    upright = env._upvector(state.data)[2]
    head_err = jp.linalg.norm(env._head_pos(state.data) - env._pos_tar)
    return on, upright, head_err

  vgate = jax.jit(jax.vmap(gate_stats))

  def warm_ybar(f: int) -> jax.Array:
    us = nom_actions[f : f + args.hsample + 1]
    if us.shape[0] < args.hsample + 1:
      pad = np.repeat(us[-1:], args.hsample + 1 - us.shape[0], axis=0)
      us = np.concatenate([us, pad], axis=0)
    return controller.u2node_vmap(jp.asarray(us))

  nB = args.batch
  tier_idx = np.arange(nB) % len(NOISE_TIERS)
  sig_q = np.array([NOISE_TIERS[i][0] for i in tier_idx])
  sig_v = np.array([NOISE_TIERS[i][1] for i in tier_idx])

  rng = jax.random.PRNGKey(args.seed)
  kept = []
  stats = []
  print(
      f"Harvesting recoveries: frames={args.frames}, batch={nB}, "
      f"N={args.num_samples}, Ndiffuse={args.num_diffuse} "
      f"(init {args.num_diffuse_init}), tiers={NOISE_TIERS}"
  )
  for f in args.frames:
    t0 = time.time()
    rng, nq_rng, nv_rng, plan_rng = jax.random.split(rng, 4)
    qpos0 = jp.tile(jp.asarray(nom_qpos[f])[None], (nB, 1))
    qvel0 = jp.tile(jp.asarray(nom_qvel[f])[None], (nB, 1))
    qpos0 = qpos0.at[:, 7:].add(
        jp.asarray(sig_q)[:, None]
        * jax.random.normal(nq_rng, (nB, 12), dtype=qpos0.dtype)
    )
    qvel0 = qvel0 + jp.asarray(sig_v)[:, None] * jax.random.normal(
        nv_rng, qvel0.shape, dtype=qvel0.dtype
    )
    states = vmake_start(qpos0, qvel0, f)

    ybar = jp.tile(warm_ybar(f)[None], (nB, 1, 1))
    rngs = jax.random.split(plan_rng, nB)
    rngs, ybar = vplan(states, rngs, ybar, args.num_diffuse_init)

    qpos_traj = [np.asarray(states.data.qpos)]
    qvel_traj = [np.asarray(states.data.qvel)]
    act_traj = []
    for t in range(f, T - 1):
      action = ybar[:, 0]
      states = vstep(states, action)
      qpos_traj.append(np.asarray(states.data.qpos))
      qvel_traj.append(np.asarray(states.data.qvel))
      act_traj.append(np.asarray(action))
      if t < T - 2:
        ybar = vshift(ybar)
        rngs, ybar = vplan(states, rngs, ybar, args.num_diffuse)

    on, upright, head_err = jax.device_get(vgate(states))
    on = np.asarray(on)
    upright = np.asarray(upright)
    head_err = np.asarray(head_err)
    success = (on >= 3) & (upright > 0.5) & (head_err < 0.35)

    qpos_rec = np.stack(qpos_traj, axis=1)  # (B, T - f, nq)
    qvel_rec = np.stack(qvel_traj, axis=1)
    act_rec = np.stack(act_traj, axis=1)  # (B, T - 1 - f, nu)

    dt_min = (time.time() - t0) / 60.0
    print(
        f"frame {f:3d}: kept {int(success.sum())}/{nB} in {dt_min:.1f} min | "
        f"feet_on {on.tolist()} | "
        f"head_err {[round(float(h), 3) for h in head_err]}"
    )
    for b in range(nB):
      stats.append(
          dict(
              frame=f,
              tier=int(tier_idx[b]),
              feet_on=int(on[b]),
              upright=float(upright[b]),
              head_err=float(head_err[b]),
              kept=bool(success[b]),
          )
      )
      if success[b]:
        kept.append(
            dict(
                qpos=np.concatenate([nom_qpos[:f], qpos_rec[b]], axis=0),
                qvel=np.concatenate([nom_qvel[:f], qvel_rec[b]], axis=0),
                actions=np.concatenate([nom_actions[:f], act_rec[b]], axis=0),
                start_frame=f,
                noise_qpos=float(sig_q[b]),
                noise_qvel=float(sig_v[b]),
                feet_on=int(on[b]),
                head_err=float(head_err[b]),
            )
        )

  total_attempts = len(args.frames) * nB
  print(
      f"Harvest complete: kept {len(kept)}/{total_attempts} recoveries "
      f"({100.0 * len(kept) / max(total_attempts, 1):.0f}% keep rate)."
  )

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      out_path,
      qpos=np.stack([nom_qpos] + [k["qpos"] for k in kept]),
      qvel=np.stack([nom_qvel] + [k["qvel"] for k in kept]),
      actions=np.stack([nom_actions] + [k["actions"] for k in kept]),
      start_frame=np.array([0] + [k["start_frame"] for k in kept]),
      noise_qpos=np.array([0.0] + [k["noise_qpos"] for k in kept]),
      noise_qvel=np.array([0.0] + [k["noise_qvel"] for k in kept]),
      final_feet_on=np.array([4] + [k["feet_on"] for k in kept]),
      final_head_err=np.array([0.0] + [k["head_err"] for k in kept]),
      dt=np.array(env.dt),
  )
  stats_path = out_path.with_suffix(".stats.json")
  with open(stats_path, "w", encoding="utf-8") as fp:
    json.dump(dict(args=vars(args), attempts=stats), fp, indent=2)
  print(
      f"Saved {len(kept) + 1} references (member 0 = nominal) to {out_path}; "
      f"per-attempt stats in {stats_path}."
  )

  if args.render_n > 0 and kept:
    # Render the earliest-start (hardest) recoveries for eyeballing.
    order = np.argsort([k["start_frame"] for k in kept])
    for rank in range(min(args.render_n, len(kept))):
      k = kept[order[rank]]
      video_path = out_path.with_name(
          f"{out_path.stem}_f{k['start_frame']}_{rank}.mp4"
      )
      traj = []
      data = template.data
      for i in range(0, T, 2):
        data = data.replace(
            qpos=jp.asarray(k["qpos"][i]), qvel=jp.asarray(k["qvel"][i])
        )
        data = mjx.forward(env.mjx_model, data)
        traj.append(template.replace(data=data))
      frames = env.render(traj, height=480, width=640, camera="track")
      media.write_video(
          video_path.as_posix(), frames, fps=1.0 / (env.dt * 2)
      )
      print(f"Rendered recovery (start frame {k['start_frame']}) to {video_path}")


if __name__ == "__main__":
  main()
