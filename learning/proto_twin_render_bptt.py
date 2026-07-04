"""Stage-0 prototype: twin-data warp rendering inside a differentiated MJX rollout.

Validates the load-bearing assumption of the D.VA-on-playground architecture:
physics rolls out on impl='jax' float64 (differentiable, BPTT), while each
step renders an egocentric depth image by pushing stop_gradient(qpos) into a
parallel impl='warp' model (non-differentiable renderer) — and the composed
program still yields finite, nonzero gradients of the loss w.r.t. a CNN
actor's parameters through the action -> dynamics -> reward path.

Gates:
  1. jax.grad through the 8-step scan does not error (no missing-VJP on warp ops).
  2. Actor-param gradients are finite and nonzero.
  3. Depth images are sane (finite, in [0,1], not constant).
  4. Timing at 64 envs: per-step cost with vs without render.

Run: python learning/proto_twin_render_bptt.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("MUJOCO_GL", "egl")

import time

import jax
from jax import config

config.update("jax_enable_x64", True)

import flax.linen as nn
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import go2_constants as consts

NUM_ENVS = 64
HORIZON = 8
N_SUBSTEPS = 10
CAM_RES = (64, 64)
DEPTH_SCALE = 3.0  # meters mapped to [0, 1]


def build_model_with_camera() -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(consts.MJX_XML_PATH.as_posix())
  base = None
  for body in spec.bodies:
    if body.name == "base":
      base = body
      break
  assert base is not None, "base body not found"
  cam = base.add_camera()
  cam.name = "egocentric"
  # Front of the robot, looking forward (+x) and slightly down. MuJoCo
  # cameras look along -z with +y up: x=(0,-1,0), y~(0.25,0,1) => -z points
  # forward with a mild downward pitch.
  cam.pos = [0.28, 0.0, -0.02]
  cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
  cam.alt.xyaxes = [0.0, -1.0, 0.0, 0.25, 0.0, 1.0]
  cam.fovy = 80.0
  model = spec.compile()
  model.opt.timestep = 0.002
  return model


class DepthProprioActor(nn.Module):
  act_dim: int = 12

  @nn.compact
  def __call__(self, depth, proprio):
    x = depth[..., None]  # (H, W, 1)
    x = nn.Conv(16, (5, 5), strides=(2, 2))(x)
    x = nn.relu(x)
    x = nn.Conv(32, (3, 3), strides=(2, 2))(x)
    x = nn.relu(x)
    x = x.reshape((-1,))
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    x = jp.concatenate([x, proprio])
    x = nn.Dense(64)(x)
    x = nn.relu(x)
    return nn.tanh(nn.Dense(self.act_dim)(x))


def main():
  mj_model = build_model_with_camera()
  ncam = mj_model.ncam
  ego_cam_id = mujoco.mj_name2id(
      mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "egocentric"
  )
  print(f"model compiled: ncam={ncam}, egocentric cam id={ego_cam_id}")

  jax_model = mjx.put_model(mj_model, impl="jax")
  # The warp FFI requires exact float32 pytree leaves; build all warp-side
  # objects with x64 disabled so their leaves are f32 despite global x64.
  config.update("jax_enable_x64", False)
  warp_model = mjx.put_model(mj_model, impl="warp")

  rc = mjx.create_render_context(
      mjm=mj_model,
      nworld=NUM_ENVS,
      cam_res=[CAM_RES for _ in range(ncam)],
      render_rgb=[False for _ in range(ncam)],
      render_depth=[i == ego_cam_id for i in range(ncam)],
      enabled_geom_groups=[0, 1, 2],
  )
  rc_pytree = rc.pytree()
  print("render context created")

  home_qpos = jp.array(mj_model.keyframe("home").qpos)
  home_ctrl = jp.array(mj_model.keyframe("home").qpos[7:])
  action_scale = jp.array([0.4, 0.8, 0.8] * 4)

  def make_jax_data(rng):
    qpos = home_qpos + 0.0 * jax.random.normal(rng, (mj_model.nq,))
    data = mjx_env.make_data(
        mj_model,
        qpos=qpos.astype(jp.float64),
        qvel=jp.zeros(mj_model.nv, dtype=jp.float64),
        ctrl=home_ctrl.astype(jp.float64),
        impl="jax",
    )
    return mjx.forward(jax_model, data)

  def make_warp_data(rng):
    data = mjx_env.make_data(
        mj_model,
        qpos=home_qpos.astype(jp.float32),
        qvel=jp.zeros(mj_model.nv, dtype=jp.float32),
        ctrl=home_ctrl.astype(jp.float32),
        impl="warp",
    )
    return data

  rngs32 = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
  wd0 = jax.jit(jax.vmap(make_warp_data))(rngs32)
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), wd0)
  config.update("jax_enable_x64", True)

  rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
  jd0 = jax.jit(jax.vmap(make_jax_data))(rngs)
  print("twin data created (jax f64 + warp f32)")

  actor = DepthProprioActor()
  dummy_depth = jp.zeros(CAM_RES, dtype=jp.float32)
  dummy_proprio = jp.zeros(15, dtype=jp.float32)
  params = actor.init(jax.random.PRNGKey(1), dummy_depth, dummy_proprio)
  n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
  print(f"actor params: {n_params}")

  def render_depth_batch(warp_data, qpos_f64, qvel_f64):
    """Push stop_gradient'd state into warp data and render depth."""
    qpos32 = jax.lax.stop_gradient(qpos_f64).astype(jp.float32)
    qvel32 = jax.lax.stop_gradient(qvel_f64).astype(jp.float32)
    warp_data = warp_data.replace(qpos=qpos32, qvel=qvel32)
    warp_data = jax.vmap(lambda d: mjx.forward(warp_model, d))(warp_data)
    render_data = jax.vmap(
        lambda d: mjx.refit_bvh(warp_model, d, rc_pytree)
    )(warp_data)
    out = jax.vmap(lambda d: mjx.render(warp_model, d, rc_pytree))(render_data)
    depth = jax.vmap(
        lambda dd: mjx.get_depth(rc_pytree, ego_cam_id, dd, DEPTH_SCALE)
    )(out[1])
    # get_depth returns (H, W, 1); normalize to (H, W) per env.
    depth = depth.reshape((depth.shape[0],) + CAM_RES)
    return warp_data, jax.lax.stop_gradient(depth)

  def proprio_of(jd):
    # Minimal proprio slice: gravity-frame z axis (3) + joint angles (12).
    return jp.concatenate(
        [jd.qpos[..., 2:5], jd.qpos[..., 7:19]], axis=-1
    ).astype(jp.float32)

  def loss_fn(actor_params, jd, wd, with_render=True):
    def step(carry, _):
      jd, wd = carry
      if with_render:
        wd, depth = render_depth_batch(wd, jd.qpos, jd.qvel)
      else:
        depth = jp.zeros((NUM_ENVS,) + CAM_RES, dtype=jp.float32)
      proprio = proprio_of(jd)
      action = jax.vmap(
          lambda dep, pro: actor.apply(actor_params, dep, pro)
      )(depth, proprio)
      ctrl = home_ctrl + action.astype(jp.float64) * action_scale
      jd = jax.vmap(
          lambda d, c: mjx_env.step(jax_model, d, c, N_SUBSTEPS)
      )(jd, ctrl)
      # Differentiable reward through jax physics: track 0.6 m/s forward,
      # stay at height 0.30.
      vx = jd.qvel[..., 0]
      height = jd.qpos[..., 2]
      reward = -((vx - 0.6) ** 2) - ((height - 0.30) ** 2)
      return (jd, wd), reward

    (_, _), rewards = jax.lax.scan(step, (jd, wd), None, length=HORIZON)
    return -rewards.mean()

  # ---- Gate 3 first: sane depth images (forward only) ----
  wd_probe, depth_probe = jax.jit(
      lambda wd, jd: render_depth_batch(wd, jd.qpos, jd.qvel)
  )(wd0, jd0)
  depth_np = np.asarray(depth_probe)
  print(
      f"depth: shape={depth_np.shape}, finite={np.isfinite(depth_np).all()}, "
      f"min={depth_np.min():.3f}, max={depth_np.max():.3f}, "
      f"std={depth_np.std():.4f}"
  )
  assert np.isfinite(depth_np).all(), "depth has NaN/inf"
  assert depth_np.std() > 1e-4, "depth image is constant — camera broken"
  np.save(
      "/tmp/claude-1000/-home-yilang-research-mujoco-playground/"
      "87ed5527-51b7-4d2a-a215-fa7d4ed6b341/scratchpad/proto_depth.npy",
      depth_np[:4],
  )

  # ---- Gates 1+2: gradient through the composed program ----
  grad_fn = jax.jit(jax.grad(lambda p, jd, wd: loss_fn(p, jd, wd, True)))
  t0 = time.time()
  grads = grad_fn(params, jd0, wd0)
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
  compile_time = time.time() - t0
  gnorm = jp.sqrt(
      sum(jp.sum(x.astype(jp.float64) ** 2) for x in jax.tree_util.tree_leaves(grads))
  )
  leaves = jax.tree_util.tree_leaves(grads)
  all_finite = all(bool(jp.isfinite(x).all()) for x in leaves)
  print(
      f"GRADIENT GATE: finite={all_finite}, norm={float(gnorm):.6e}, "
      f"compile+first-call={compile_time:.1f}s"
  )
  assert all_finite, "gradients contain NaN/inf"
  assert float(gnorm) > 1e-12, "gradients are all zero — no path to actor"

  # ---- Gate 4: timing with vs without render ----
  for reps in range(2):  # second rep = warm
    t0 = time.time()
    g = grad_fn(params, jd0, wd0)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
    t_with = time.time() - t0
  grad_fn_norender = jax.jit(
      jax.grad(lambda p, jd, wd: loss_fn(p, jd, wd, False))
  )
  g = grad_fn_norender(params, jd0, wd0)  # compile
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
  t0 = time.time()
  g = grad_fn_norender(params, jd0, wd0)
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
  t_without = time.time() - t0
  print(
      f"TIMING (grad of {HORIZON}-step window, {NUM_ENVS} envs): "
      f"with render {t_with*1e3:.1f} ms, without {t_without*1e3:.1f} ms, "
      f"render overhead {(t_with-t_without)*1e3:.1f} ms "
      f"({(t_with/max(t_without,1e-9)-1)*100:.0f}%)"
  )
  print("ALL GATES PASSED")


if __name__ == "__main__":
  main()
