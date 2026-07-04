"""JAX port of D.VA (decoupled visual analytic policy gradient) for playground envs.

Faithful port of D.VA's training loop (`D.VA/algorithms/dva.py`, a SHAC fork)
onto the mujoco_playground + brax wrapper stack:

  * Actor loss: H-step BPTT window through differentiable MJX physics with a
    terminal value bootstrap from a target critic. The terminal value V(s) is
    differentiated THROUGH its state input (dva.py:243 computes next_values
    inside the graph) — this is SHAC's credit path beyond the window.
  * The D.VA decoupling: the actor consumes stop_gradient(obs) even in state
    mode (dva.py:209), so no gradient ever flows through the observation
    path. This is what later admits a non-differentiable (warp) renderer for
    vision observations.
  * Stochastic actor: MLP mu head + state-independent learned logstd;
    reparameterized Normal sample, tanh applied at env-step time
    (dva.py:211), not inside the distribution.
  * Asymmetric critic: always trained on the *state* observation with
    TD(lambda) targets (dva.py:420-434, exact recursion) computed from a
    target critic; Polyak target update target = a*target + (1-a)*online.
  * Critic fitting: `critic_iterations` sweeps over `num_batch` fixed
    contiguous minibatches of the flattened window buffer (CriticDataset is
    constructed with shuffle=False in dva.py).
  * Termination vs truncation: envs done by early termination bootstrap 0;
    envs done by episode truncation bootstrap V(obs_before_reset) — the
    pre-autoreset observation (dva.py:245-257), guarded against non-finite
    values.
  * Observation normalization: running mean/std updated every step with raw
    obs, while a copy frozen at window start is used to normalize
    (dva.py:178-196).

Vision-readiness: observations may be a flat vector or a dict;
`actor_obs_key` / `critic_obs_key` select the actor and critic views. In
state mode both are "state". The critic always reads the state view.

Vision mode (`vision=True`): the env stays pixel-free; the trainer renders an
egocentric depth image per step with the MJX-Warp batch renderer using the
twin-data trick validated by the Stage-0 prototype — physics rolls on
impl='jax' float64 (differentiable BPTT), and each step pushes
stop_gradient(qpos/qvel) as float32 into a parallel impl='warp' model, then
forward -> refit_bvh -> render -> get_depth. The renderer has no JAX
gradients, which is admissible because D.VA never differentiates the actor's
observation path. The vision actor consumes a frame-stacked depth image plus
the (normalized, detached) proprio vector; the critic still reads the full
state. Additional deliberate deviation in vision mode: D.VA's visual actor is
pixels-only, ours concatenates proprio after the encoder — depth of flat
ground cannot convey the reference/phase tracking target.

Deliberate deviations from dva.py (flagged, not silent):
  * Actor mu-net weight init is flax default (lecun normal) instead of
    torch's kaiming uniform; critic keeps dva's orthogonal(sqrt(2)) init.
  * The running-stats update covers each observation exactly once, but the
    per-window boundary obs is credited to the following window (a pure
    reindexing of dva.py's update schedule; same stats over training).
  * Env time staggering: like brax APG we scramble the episode-step clocks at
    init so truncations are spread across the batch (dflex envs stagger via
    stochastic init instead).
  * An env whose own termination fires on exactly the timeout step is treated
    as terminated (bootstrap 0) via the EpisodeWrapper truncation flag;
    dva.py's length-only rule (dva.py:250) would bootstrap V(obs_before_reset)
    there. The port's rule is the more correct MDP semantics.

Single-device implementation (no pmap).
"""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from brax import envs as brax_envs
from brax.envs.wrappers import training as brax_training
from brax.training import acting
import flax
import flax.linen as nn
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import optax

from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as pg_wrapper

Metrics = dict
_OBS_BEFORE_RESET = 'obs_before_reset'


# ---------------------------------------------------------------------------
# Running mean/std (port of D.VA utils/running_mean_std.py, parallel-variance
# update, count init 1e-4, normalize eps 1e-5).
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class RunningMeanStd:
  mean: jax.Array
  var: jax.Array
  count: jax.Array


def rms_init(shape: Sequence[int]) -> RunningMeanStd:
  return RunningMeanStd(
      mean=jp.zeros(shape, dtype=jp.float32),
      var=jp.ones(shape, dtype=jp.float32),
      count=jp.asarray(1e-4, dtype=jp.float32),
  )


def rms_update(rms: RunningMeanStd, batch: jax.Array) -> RunningMeanStd:
  batch = jax.lax.stop_gradient(batch).astype(jp.float32)
  batch_mean = jp.mean(batch, axis=0)
  batch_var = jp.var(batch, axis=0)
  batch_count = jp.asarray(batch.shape[0], dtype=jp.float32)
  delta = batch_mean - rms.mean
  tot_count = rms.count + batch_count
  new_mean = rms.mean + delta * batch_count / tot_count
  m_a = rms.var * rms.count
  m_b = batch_var * batch_count
  m_2 = m_a + m_b + jp.square(delta) * rms.count * batch_count / tot_count
  return RunningMeanStd(mean=new_mean, var=m_2 / tot_count, count=tot_count)


def rms_normalize(rms: RunningMeanStd, x: jax.Array) -> jax.Array:
  return (x - rms.mean) / jp.sqrt(rms.var + 1e-5)


# ---------------------------------------------------------------------------
# Networks (port of D.VA models/actor.py ActorStochasticMLP and
# models/critic.py CriticMLP: Linear -> ELU -> LayerNorm hidden blocks).
# ---------------------------------------------------------------------------


class ActorMLP(nn.Module):
  """Stochastic-actor mu net + learned state-independent logstd."""

  action_size: int
  hidden: Sequence[int] = (128, 64, 32)
  logstd_init: float = -1.0

  @nn.compact
  def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    x = obs
    for size in self.hidden:
      x = nn.Dense(size)(x)
      x = nn.elu(x)
      x = nn.LayerNorm(epsilon=1e-5)(x)
    mu = nn.Dense(self.action_size)(x)
    logstd = self.param(
        'logstd',
        lambda _, shape: jp.full(shape, self.logstd_init, dtype=jp.float32),
        (self.action_size,),
    )
    return mu, logstd


class CriticMLP(nn.Module):
  hidden: Sequence[int] = (64, 64)

  @nn.compact
  def __call__(self, obs: jax.Array) -> jax.Array:
    x = obs
    kernel_init = nn.initializers.orthogonal(scale=jp.sqrt(2.0))
    for size in self.hidden:
      x = nn.Dense(size, kernel_init=kernel_init)(x)
      x = nn.elu(x)
      x = nn.LayerNorm(epsilon=1e-5)(x)
    return nn.Dense(1, kernel_init=kernel_init)(x)


class VisionActor(nn.Module):
  """Depth-stack + proprio stochastic actor.

  Encoder is D.VA's DrQv2-style Encoder (models/encoder.py): four
  Conv(32, 3x3) layers (strides 2,1,1,1, VALID padding, ReLU, orthogonal
  sqrt(2) init) then Linear -> LayerNorm -> Tanh trunk. Depth input is in
  [0, 1] and gets centered by -0.5 (the analogue of D.VA's /255 - 0.5).
  The trunk output is concatenated with the proprio vector and fed through
  the same mu-net blocks as ActorMLP. Inputs are unbatched (vmap outside).
  """

  action_size: int
  encoder_dim: int = 128
  hidden: Sequence[int] = (128, 64, 32)
  logstd_init: float = -1.0

  @nn.compact
  def __call__(
      self, depth_stack: jax.Array, proprio: jax.Array
  ) -> Tuple[jax.Array, jax.Array]:
    conv_init = nn.initializers.orthogonal(scale=jp.sqrt(2.0))
    x = depth_stack - 0.5  # (H, W, C) in [0,1]
    for stride in (2, 1, 1, 1):
      x = nn.Conv(
          32,
          (3, 3),
          strides=(stride, stride),
          padding='VALID',
          kernel_init=conv_init,
      )(x)
      x = nn.relu(x)
    x = x.reshape(-1)
    x = nn.Dense(
        self.encoder_dim, kernel_init=nn.initializers.orthogonal(scale=1.0)
    )(x)
    x = nn.LayerNorm(epsilon=1e-5)(x)
    x = nn.tanh(x)
    x = jp.concatenate([x, proprio])
    for size in self.hidden:
      x = nn.Dense(size)(x)
      x = nn.elu(x)
      x = nn.LayerNorm(epsilon=1e-5)(x)
    mu = nn.Dense(self.action_size)(x)
    logstd = self.param(
        'logstd',
        lambda _, shape: jp.full(shape, self.logstd_init, dtype=jp.float32),
        (self.action_size,),
    )
    return mu, logstd


# ---------------------------------------------------------------------------
# Vision rendering: MJX-Warp twin-data batch renderer (Stage-0 design).
# ---------------------------------------------------------------------------


class VisionAssets:
  """Warp-side model/data/render-context twins for depth rendering.

  Not a pytree: holds the warp model, the live RenderContext (kept alive on
  purpose — its buffers back the pytree handle), and per-batch warp Data.

  cam_res follows the render-context convention (width, height); rendered
  depth images come back row-major (height, width).
  """

  def __init__(self, mj_model: mujoco.MjModel, num_worlds: int,
               camera: str, cam_res: Tuple[int, int]):
    self.cam_res = tuple(cam_res)
    self.cam_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
    )
    if self.cam_id < 0:
      raise ValueError(f'camera {camera!r} not found in model')
    ncam = mj_model.ncam
    # The warp FFI requires exact float32 pytree leaves: build every
    # warp-side object with x64 temporarily disabled.
    x64_was = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)
    try:
      self.warp_model = mjx.put_model(mj_model, impl='warp')
      self.render_context = mjx.create_render_context(
          mjm=mj_model,
          nworld=num_worlds,
          cam_res=[self.cam_res for _ in range(ncam)],
          render_rgb=[False for _ in range(ncam)],
          render_depth=[i == self.cam_id for i in range(ncam)],
          enabled_geom_groups=[0, 1, 2],
      )
      self.rc_pytree = self.render_context.pytree()
      home_qpos = jp.asarray(
          mj_model.keyframe('home').qpos, dtype=jp.float32
      )

      def _make_wd(_):
        return mjx_env.make_data(
            mj_model,
            qpos=home_qpos,
            qvel=jp.zeros(mj_model.nv, dtype=jp.float32),
            ctrl=home_qpos[7:],
            impl='warp',
        )

      self.warp_data = jax.jit(jax.vmap(_make_wd))(jp.arange(num_worlds))
      jax.block_until_ready(self.warp_data)
    finally:
      jax.config.update('jax_enable_x64', x64_was)


def make_depth_render_fn(assets: VisionAssets, depth_scale: float):
  """Returns render(warp_data, qpos, qvel) -> (warp_data, depth (B, H, W)).

  Detaches state before handing it to warp (no gradients exist through the
  renderer) and detaches the output for symmetry.
  """
  warp_model = assets.warp_model
  rc_p = assets.rc_pytree
  cam_id = assets.cam_id

  def render(warp_data, qpos, qvel):
    qpos32 = jax.lax.stop_gradient(qpos).astype(jp.float32)
    qvel32 = jax.lax.stop_gradient(qvel).astype(jp.float32)
    warp_data = warp_data.replace(qpos=qpos32, qvel=qvel32)
    warp_data = jax.vmap(lambda d: mjx.forward(warp_model, d))(warp_data)
    render_data = jax.vmap(
        lambda d: mjx.refit_bvh(warp_model, d, rc_p)
    )(warp_data)
    out = jax.vmap(lambda d: mjx.render(warp_model, d, rc_p))(render_data)
    depth = jax.vmap(
        lambda dd: mjx.get_depth(rc_p, cam_id, dd, depth_scale)
    )(out[1])
    # get_depth returns (H, W, 1) per world — already row-major (H, W); just
    # drop the channel axis. Do NOT reshape with cam_res: cam_res is
    # (width, height) in the render-context convention and a reshape would
    # silently row-scramble non-square images.
    depth = depth[..., 0]
    # Degenerate poses (post-kick physics blowups) can render non-finite
    # depth; sanitize so one bad frame cannot NaN the whole update.
    depth = jp.clip(
        jp.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )
    return warp_data, jax.lax.stop_gradient(depth)

  return render


def _update_frame_stack(
    stack: jax.Array, frame: jax.Array, reset_mask: jax.Array
) -> jax.Array:
  """Shift the newest frame in; on reset, fill the stack with the new frame.

  stack: (B, H, W, C), frame: (B, H, W), reset_mask: (B,) 0/1.
  """
  shifted = jp.concatenate([stack[..., 1:], frame[..., None]], axis=-1)
  tiled = jp.repeat(frame[..., None], stack.shape[-1], axis=-1)
  reset = jp.reshape(reset_mask > 0.5, (-1, 1, 1, 1))
  return jp.where(reset, tiled, shifted)


# ---------------------------------------------------------------------------
# Env wrapping: standard playground stack, with an AutoReset variant that
# exposes the pre-reset observation for the truncation bootstrap.
# ---------------------------------------------------------------------------


class AutoResetWithObsBeforeReset(pg_wrapper.Wrapper):
  """Playground BraxAutoResetWrapper (full_reset=False path) that additionally
  stores the post-step, pre-reset observation in info['obs_before_reset']."""

  def __init__(self, env: Any):
    super().__init__(env)
    self._info_key = 'AutoResetWrapper'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self.env.reset(key)
    state.info[f'{self._info_key}_first_data'] = state.data
    state.info[f'{self._info_key}_first_obs'] = state.obs
    state.info[f'{self._info_key}_rng'] = rng
    state.info[_OBS_BEFORE_RESET] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    reset_data = state.info[f'{self._info_key}_first_data']
    reset_obs = state.info[f'{self._info_key}_first_obs']
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng = rng_key[..., 0]

    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)

    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_data, state.data)
    obs = jax.tree.map(where_done, reset_obs, state.obs)

    next_info = state.info
    next_info[_OBS_BEFORE_RESET] = state.obs
    next_info[f'{self._info_key}_rng'] = reset_rng
    return state.replace(data=data, obs=obs, info=next_info)


def wrap_for_dva_training(
    env: mjx_env.MjxEnv,
    episode_length: int,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[..., Any]] = None,
) -> pg_wrapper.Wrapper:
  if randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = pg_wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = AutoResetWithObsBeforeReset(env)
  return env


# ---------------------------------------------------------------------------
# Training state and policy factory.
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class TrainingState:
  actor_params: Any
  actor_opt_state: optax.OptState
  critic_params: Any
  critic_opt_state: optax.OptState
  target_critic_params: Any
  normalizer: RunningMeanStd
  # Per-env accumulators for completed-episode logging (raw, unscaled reward).
  episode_return: jax.Array
  episode_len: jax.Array


def _select_obs(obs: Any, key: str) -> jax.Array:
  if isinstance(obs, Mapping):
    return obs[key]
  return obs


def make_policy_factory(
    actor: nn.Module,
    actor_obs_key: str,
    normalize_observations: bool = True,
    vision: bool = False,
    proprio_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> Callable[..., Callable]:
  """Returns make_policy(params, deterministic) compatible with brax acting.

  params = (RunningMeanStd, actor_params). In state mode the policy takes the
  env observation (batched or not — Dense broadcasts). In vision mode it
  takes a dict {'depth_stack': (H, W, C), actor_obs_key: state vector},
  UNBATCHED (vmap outside — the conv encoder does not broadcast).
  """

  def make_policy(params, deterministic: bool = False):
    normalizer, actor_params = params

    def policy(obs, key):
      if vision:
        x = _select_obs(obs, actor_obs_key)
        if normalize_observations:
          x = rms_normalize(normalizer, x)
        proprio = proprio_fn(x.astype(jp.float32))
        mu, logstd = actor.apply(
            actor_params, obs['depth_stack'], proprio
        )
      else:
        x = _select_obs(obs, actor_obs_key)
        if normalize_observations:
          x = rms_normalize(normalizer, x)
        x = x.astype(jp.float32)
        mu, logstd = actor.apply(actor_params, x)
      if deterministic:
        raw = mu
      else:
        raw = mu + jp.exp(logstd) * jax.random.normal(
            key, mu.shape, dtype=mu.dtype
        )
      return jp.tanh(raw), {}

    return policy

  return make_policy


# ---------------------------------------------------------------------------
# TD(lambda) targets — exact port of dva.py compute_target_values.
# ---------------------------------------------------------------------------


def compute_td_lambda_targets(
    rew_buf: jax.Array,  # (H, N)
    done_mask: jax.Array,  # (H, N), last row forced to 1
    next_values: jax.Array,  # (H, N)
    gamma: float,
    lam: float,
) -> jax.Array:
  num_envs = rew_buf.shape[1]

  def body(carry, xs):
    ai, bi, lam_acc = carry
    rew, dm, nv = xs
    lam_acc = lam_acc * lam * (1.0 - dm) + dm
    ai = (1.0 - dm) * (
        lam * gamma * ai + gamma * nv + (1.0 - lam_acc) / (1.0 - lam) * rew
    )
    bi = gamma * (nv * dm + bi * (1.0 - dm)) + rew
    target = (1.0 - lam) * ai + lam_acc * bi
    return (ai, bi, lam_acc), target

  init = (
      jp.zeros(num_envs, dtype=jp.float32),
      jp.zeros(num_envs, dtype=jp.float32),
      jp.ones(num_envs, dtype=jp.float32),
  )
  _, targets = jax.lax.scan(
      body, init, (rew_buf, done_mask, next_values), reverse=True
  )
  return targets  # (H, N)


# ---------------------------------------------------------------------------
# Trainer.
# ---------------------------------------------------------------------------


def train(
    environment: mjx_env.MjxEnv,
    episode_length: int,
    max_epochs: int = 1000,
    horizon_length: int = 32,
    num_envs: int = 64,
    num_evals: int = 11,
    num_eval_envs: int = 64,
    action_repeat: int = 1,
    actor_lr: float = 2e-3,
    critic_lr: float = 2e-3,
    lr_schedule: str = 'linear',
    betas: Tuple[float, float] = (0.7, 0.95),
    gamma: float = 0.99,
    td_lambda: float = 0.95,
    target_critic_alpha: float = 0.2,
    critic_iterations: int = 16,
    num_batch: int = 4,
    rew_scale: float = 1.0,
    grad_norm: float = 1.0,
    truncate_grads: bool = True,
    normalize_observations: bool = True,
    actor_hidden: Sequence[int] = (128, 64, 32),
    critic_hidden: Sequence[int] = (64, 64),
    actor_logstd_init: float = -1.0,
    deterministic_eval: bool = True,
    scramble_time: bool = True,
    # Redraw all per-episode reset randomness (RSI frame, reference-ensemble
    # member, injection noise) at every eval block boundary. The autoreset
    # wrapper (full_reset=False) restores each env slot's cached FIRST reset
    # pose bit-for-bit on done, so without this a run only ever sees
    # num_envs distinct reset tuples.
    reset_envs_per_block: bool = False,
    seed: int = 0,
    actor_obs_key: str = 'state',
    critic_obs_key: str = 'state',
    vision: bool = False,
    cam_res: Tuple[int, int] = (64, 64),
    frame_stack: int = 3,
    depth_scale: float = 3.0,
    vision_camera: str = 'egocentric',
    vision_proprio: str = 'full',
    encoder_dim: int = 128,
    randomization_fn: Optional[Callable[..., Any]] = None,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_fn: Optional[Callable[[int, Any], None]] = None,
    eval_env: Optional[mjx_env.MjxEnv] = None,
    restore_params: Optional[Any] = None,
    return_training_state: bool = False,
):
  """D.VA training. Returns (make_policy, params, metrics) like brax trainers.

  params = (RunningMeanStd normalizer, actor_params). The critic is internal
  to training (asymmetric; not needed for inference).
  """
  num_evals_after_init = max(num_evals - 1, 1)
  epochs_per_eval = max_epochs // num_evals_after_init
  if (horizon_length * num_envs) % num_batch != 0:
    # dva.py's CriticDataset(drop_last=False) trains on a ragged tail batch;
    # the fixed-shape scan here cannot, so demand equal batches up front
    # rather than silently dropping samples.
    raise ValueError(
        f'horizon_length*num_envs ({horizon_length * num_envs}) must be '
        f'divisible by num_batch ({num_batch})'
    )

  key = jax.random.PRNGKey(seed)
  key, net_key, env_key, scramble_key, eval_key = jax.random.split(key, 5)

  # --- env setup -----------------------------------------------------------
  rand_train = None
  rand_eval = None
  if randomization_fn is not None:
    import functools  # pylint: disable=g-import-not-at-top

    key, rand_train_key = jax.random.split(key)
    rand_train = functools.partial(
        randomization_fn, rng=jax.random.split(rand_train_key, num_envs)
    )
    rand_eval = functools.partial(
        randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
    )

  env = wrap_for_dva_training(
      environment,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=rand_train,
  )
  obs_size = env.observation_size
  if isinstance(obs_size, Mapping):
    state_obs_size = obs_size[critic_obs_key]
    actor_obs_size = obs_size[actor_obs_key]
  else:
    state_obs_size = actor_obs_size = obs_size
  action_size = env.action_size

  # --- vision: proprio slice + renderer twins --------------------------------
  if vision_proprio == 'full':
    _proprio_of = lambda x: x
    proprio_size = actor_obs_size
  elif vision_proprio == 'no_linvel':
    # Drop base_linvel_local (Go2SampleAPG obs indices 42:45) — the classic
    # hard-to-estimate quantity on hardware. Note on flat featureless ground
    # depth carries no substitute signal for it (partial observability).
    _proprio_of = lambda x: jp.concatenate([x[..., :42], x[..., 45:]], -1)
    proprio_size = actor_obs_size - 3
  else:
    raise ValueError(f'Unknown vision_proprio={vision_proprio!r}')

  assets = None
  render_depth = None
  if vision:
    if num_eval_envs != num_envs:
      # The warp render context hardcodes nworld; one shared context keeps
      # things simple. Lift this by building a second context if ever needed.
      raise ValueError(
          f'vision mode requires num_eval_envs ({num_eval_envs}) == '
          f'num_envs ({num_envs})'
      )
    assets = VisionAssets(
        environment.mj_model, num_envs, vision_camera, cam_res
    )
    render_depth = make_depth_render_fn(assets, depth_scale)

  # --- networks and optimizers ----------------------------------------------
  if vision:
    actor = VisionActor(
        action_size=action_size,
        encoder_dim=encoder_dim,
        hidden=tuple(actor_hidden),
        logstd_init=actor_logstd_init,
    )
  else:
    actor = ActorMLP(
        action_size=action_size,
        hidden=tuple(actor_hidden),
        logstd_init=actor_logstd_init,
    )
  critic = CriticMLP(hidden=tuple(critic_hidden))

  dummy_state_obs = jp.zeros((state_obs_size,), dtype=jp.float32)
  actor_key, critic_key = jax.random.split(net_key)
  if vision:
    # Depth images are (height, width); cam_res is (width, height).
    dummy_depth = jp.zeros(
        (cam_res[1], cam_res[0], frame_stack), dtype=jp.float32
    )
    dummy_proprio = jp.zeros((proprio_size,), dtype=jp.float32)
    actor_params = actor.init(actor_key, dummy_depth, dummy_proprio)
  else:
    dummy_actor_obs = jp.zeros((actor_obs_size,), dtype=jp.float32)
    actor_params = actor.init(actor_key, dummy_actor_obs)
  critic_params = critic.init(critic_key, dummy_state_obs)
  init_target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

  updates_per_critic_epoch = critic_iterations * num_batch

  def _linear_lr(base_lr, count_to_epoch):
    if lr_schedule == 'linear':
      def sched(count):
        epoch = count_to_epoch(count)
        frac = jp.minimum(epoch / max_epochs, 1.0)
        return base_lr + (1e-5 - base_lr) * frac
      return sched
    return base_lr

  clip = (
      optax.clip_by_global_norm(grad_norm)
      if truncate_grads
      else optax.identity()
  )

  def _nan_to_zero():
    def update_fn(updates, state, params=None):
      del params
      return (
          jax.tree.map(
              lambda g: jp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
              updates,
          ),
          state,
      )

    return optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn)

  # apply_if_finite: a transient sim/render NaN (dva.py's "ugly fix for
  # simulation nan" class of event — the env self-heals at the next
  # truncation via the autoreset swap) skips the poisoned update instead of
  # corrupting params. Persistent NaN still fails loudly: after
  # max_consecutive_errors the wrapper stops protecting, params go
  # non-finite, and the host-side check below raises.
  actor_opt = optax.apply_if_finite(
      optax.chain(
          clip,
          optax.adam(
              learning_rate=_linear_lr(actor_lr, lambda c: c),
              b1=betas[0],
              b2=betas[1],
          ),
      ),
      max_consecutive_errors=100,
  )
  critic_opt = optax.chain(
      _nan_to_zero(),
      clip,
      optax.adam(
          learning_rate=_linear_lr(
              critic_lr, lambda c: c // updates_per_critic_epoch
          ),
          b1=betas[0],
          b2=betas[1],
      ),
  )

  training_state = TrainingState(
      actor_params=actor_params,
      actor_opt_state=actor_opt.init(actor_params),
      critic_params=critic_params,
      critic_opt_state=critic_opt.init(critic_params),
      target_critic_params=init_target_critic_params,
      normalizer=rms_init((state_obs_size,)),
      episode_return=jp.zeros(num_envs, dtype=jp.float32),
      episode_len=jp.zeros(num_envs, dtype=jp.float32),
  )

  if restore_params is not None:
    # Warm start: either (RunningMeanStd, actor_params) — e.g. a
    # behavior-cloned actor, fresh critic — or the 4-tuple
    # (RunningMeanStd, actor_params, critic_params, target_critic_params)
    # returned with return_training_state=True, for curriculum stages that
    # must carry the value function forward (the whole point of a reverse
    # curriculum is an increasingly truthful V). Optimizer state is fresh
    # either way.
    restored = jax.device_put(restore_params)
    if len(restored) == 2:
      restore_norm, restore_actor = restored
      training_state = training_state.replace(
          actor_params=restore_actor, normalizer=restore_norm
      )
    elif len(restored) == 4:
      restore_norm, restore_actor, restore_critic, restore_target = restored
      training_state = training_state.replace(
          actor_params=restore_actor,
          normalizer=restore_norm,
          critic_params=restore_critic,
          target_critic_params=restore_target,
      )
    else:
      raise ValueError(
          f'restore_params must have 2 or 4 elements, got {len(restored)}'
      )

  make_policy = make_policy_factory(
      actor,
      actor_obs_key,
      normalize_observations,
      vision=vision,
      proprio_fn=_proprio_of,
  )

  def _normalize(rms, x):
    if normalize_observations:
      return rms_normalize(rms, x)
    return x

  # --- actor loss: H-step BPTT window (port of compute_actor_loss) ----------

  def actor_loss_fn(actor_params, carry_in, frozen_rms, target_critic_params, rng):
    # target_critic_params MUST be threaded through here (not closed over):
    # a closure would bake the initial random critic into the jitted epoch
    # and the Polyak updates in TrainingState would never be seen.
    env_state, warp_data, stack, rms, ep_ret, ep_len = carry_in

    critic_state_value = lambda p, o: critic.apply(
        p, _normalize(frozen_rms, o).astype(jp.float32)
    ).squeeze(-1)

    def step(carry, step_idx):
      (
          env_state,
          warp_data,
          stack,
          rms,
          rew_acc,
          disc,
          loss_sum,
          ep_ret,
          ep_len,
          rng,
      ) = carry

      obs_actor = _select_obs(env_state.obs, actor_obs_key)
      obs_state = _select_obs(env_state.obs, critic_obs_key)
      # Update running stats with the raw state obs (each obs exactly once),
      # normalize with the window-frozen copy. Like dva.py's disabled
      # state_obs_rms, no stats are collected when normalization is off.
      if normalize_observations:
        rms = rms_update(rms, obs_state)
      norm_actor_obs = _normalize(frozen_rms, obs_actor)
      # The D.VA decoupling: the actor never differentiates through obs.
      norm_actor_obs = jax.lax.stop_gradient(norm_actor_obs).astype(jp.float32)

      rng, sample_key = jax.random.split(rng)
      if vision:
        # Render the current (post-autoreset-swap) state; a done flag on the
        # carried state marks a fresh reset, so the stack re-tiles with the
        # newly rendered frame instead of carrying stale-episode history.
        warp_data, frame = render_depth(
            warp_data, env_state.data.qpos, env_state.data.qvel
        )
        stack = _update_frame_stack(stack, frame, env_state.done)
        proprio = _proprio_of(norm_actor_obs)
        mu, logstd = jax.vmap(
            lambda d, p: actor.apply(actor_params, d, p)
        )(stack, proprio)
      else:
        mu, logstd = actor.apply(actor_params, norm_actor_obs)
      eps = jax.random.normal(sample_key, mu.shape, dtype=mu.dtype)
      raw_action = mu + jp.exp(logstd) * eps

      next_state = env.step(env_state, jp.tanh(raw_action))
      raw_rew = next_state.reward
      rew = raw_rew * rew_scale
      done = next_state.done  # float 0/1
      trunc = next_state.info['truncation']

      # Terminal value: differentiable through the state input (SHAC path).
      next_obs_state = _select_obs(next_state.obs, critic_obs_key)
      obs_br = _select_obs(next_state.info[_OBS_BEFORE_RESET], critic_obs_key)
      # Guard non-finite/huge pre-reset obs BEFORE the critic so the untaken
      # jp.where branch cannot poison gradients (dva.py:246-249).
      br_bad = jp.logical_or(
          jp.logical_not(jp.all(jp.isfinite(obs_br), axis=-1)),
          jp.any(jp.abs(obs_br) > 1e6, axis=-1),
      )
      obs_br_safe = jp.where(br_bad[..., None], 0.0, obs_br)
      v_next = critic_state_value(target_critic_params, next_obs_state)
      v_br = critic_state_value(target_critic_params, obs_br_safe)
      v_br = jp.where(br_bad, 0.0, v_br)
      # done & truncated -> bootstrap pre-reset obs; done & terminated -> 0;
      # not done -> bootstrap the (differentiable) next obs.
      nv = jp.where(done > 0.5, jp.where(trunc > 0.5, v_br, 0.0), v_next)

      rew_acc_next = rew_acc + disc * rew
      is_last = step_idx == horizon_length - 1
      # Mid-window: only done envs close out a loss term; final step: all.
      mask = jp.where(is_last, jp.ones_like(done), done)
      loss_sum = loss_sum + jp.sum(mask * (-rew_acc_next - gamma * disc * nv))

      disc_next = disc * gamma
      disc_next = jp.where(done > 0.5, 1.0, disc_next)
      rew_acc_next = jp.where(done > 0.5, 0.0, rew_acc_next)

      # Completed-episode logging (raw reward, resets on done).
      ep_ret_next = ep_ret + raw_rew.astype(jp.float32)
      ep_len_next = ep_len + 1.0
      done32 = done.astype(jp.float32)
      stats = (
          jp.sum(done32 * ep_ret_next),
          jp.sum(done32 * ep_len_next),
          jp.sum(done32),
      )
      ep_ret_next = jp.where(done > 0.5, 0.0, ep_ret_next)
      ep_len_next = jp.where(done > 0.5, 0.0, ep_len_next)

      # Critic dataset entries for this step (frozen-normalized obs at time i).
      buf = (
          jax.lax.stop_gradient(
              _normalize(frozen_rms, obs_state).astype(jp.float32)
          ),
          jax.lax.stop_gradient(rew.astype(jp.float32)),
          done.astype(jp.float32),
          jax.lax.stop_gradient(nv.astype(jp.float32)),
      )

      carry = (
          next_state,
          warp_data,
          stack,
          rms,
          rew_acc_next,
          disc_next,
          loss_sum,
          ep_ret_next,
          ep_len_next,
          rng,
      )
      return carry, (buf, stats)

    init = (
        env_state,
        warp_data,
        stack,
        rms,
        jp.zeros(num_envs),
        jp.ones(num_envs),
        jp.asarray(0.0),
        ep_ret,
        ep_len,
        rng,
    )
    carry, (bufs, stats) = jax.lax.scan(
        step, init, jp.arange(horizon_length)
    )
    env_state, warp_data, stack, rms, _, _, loss_sum, ep_ret, ep_len, _ = carry
    actor_loss = loss_sum / (horizon_length * num_envs)

    obs_buf, rew_buf, done_buf, nv_buf = bufs
    done_mask = done_buf.at[-1, :].set(1.0)  # dva.py:284
    ep_stats = jax.tree.map(jp.sum, stats)
    aux = (
        (env_state, warp_data, stack, rms, ep_ret, ep_len),
        (obs_buf, rew_buf, done_mask, nv_buf),
        ep_stats,
    )
    return actor_loss, aux

  actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)

  # --- critic fitting (port of dva.py critic training loop) -----------------

  def critic_update(critic_params, critic_opt_state, obs_buf, targets):
    # Flatten (H, N) -> fixed contiguous minibatches, no shuffle
    # (CriticDataset default). H*N must be divisible by num_batch.
    flat_obs = obs_buf.reshape(-1, obs_buf.shape[-1])
    flat_tgt = targets.reshape(-1)
    batch_size = flat_obs.shape[0] // num_batch
    b_obs = flat_obs[: batch_size * num_batch].reshape(
        num_batch, batch_size, -1
    )
    b_tgt = flat_tgt[: batch_size * num_batch].reshape(num_batch, batch_size)

    def mse_loss(params, obs_b, tgt_b):
      pred = critic.apply(params, obs_b).squeeze(-1)
      return jp.mean((pred - tgt_b) ** 2)

    critic_loss_grad = jax.value_and_grad(mse_loss)

    def one_batch(carry, xs):
      params, opt_state = carry
      obs_b, tgt_b = xs
      loss, grads = critic_loss_grad(params, obs_b, tgt_b)
      updates, opt_state = critic_opt.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return (params, opt_state), loss

    def one_iteration(carry, _):
      carry, losses = jax.lax.scan(one_batch, carry, (b_obs, b_tgt))
      return carry, jp.mean(losses)

    (critic_params, critic_opt_state), iter_losses = jax.lax.scan(
        one_iteration,
        (critic_params, critic_opt_state),
        None,
        length=critic_iterations,
    )
    return critic_params, critic_opt_state, iter_losses[-1]

  # --- one epoch = one actor update + critic fitting + target polyak --------

  def training_epoch(carry, _):
    ts, env_carry, rng = carry
    env_state, warp_data, stack = env_carry
    rng, rollout_key = jax.random.split(rng)
    frozen_rms = ts.normalizer

    (actor_loss, aux), grads = actor_grad_fn(
        ts.actor_params,
        (
            env_state,
            warp_data,
            stack,
            ts.normalizer,
            ts.episode_return,
            ts.episode_len,
        ),
        frozen_rms,
        ts.target_critic_params,
        rollout_key,
    )
    (env_state, warp_data, stack, rms, ep_ret, ep_len), bufs, ep_stats = aux
    obs_buf, rew_buf, done_mask, nv_buf = bufs

    grad_norm_before = optax.global_norm(grads)
    updates, actor_opt_state = actor_opt.update(
        grads, ts.actor_opt_state, ts.actor_params
    )
    actor_params = optax.apply_updates(ts.actor_params, updates)

    targets = compute_td_lambda_targets(
        rew_buf, done_mask, nv_buf, gamma, td_lambda
    )
    critic_params, critic_opt_state, value_loss = critic_update(
        ts.critic_params, ts.critic_opt_state, obs_buf, targets
    )

    alpha = target_critic_alpha
    target_critic_params = jax.tree.map(
        lambda t, c: alpha * t + (1.0 - alpha) * c,
        ts.target_critic_params,
        critic_params,
    )

    ts = TrainingState(
        actor_params=actor_params,
        actor_opt_state=actor_opt_state,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        target_critic_params=target_critic_params,
        normalizer=rms,
        episode_return=ep_ret,
        episode_len=ep_len,
    )
    metrics = {
        'actor_loss': actor_loss,
        'value_loss': value_loss,
        'grad_norm_before_clip': grad_norm_before,
        'grad_norm_after_clip': jp.minimum(
            grad_norm_before, grad_norm if truncate_grads else jp.inf
        ),
        'actor_notfinite_total': actor_opt_state.total_notfinite,
        'episode_return_sum': ep_stats[0],
        'episode_len_sum': ep_stats[1],
        'episode_count': ep_stats[2],
    }
    return (ts, (env_state, warp_data, stack), rng), metrics

  @jax.jit
  def training_block(ts, env_carry, rng):
    (ts, env_carry, rng), metrics = jax.lax.scan(
        training_epoch, (ts, env_carry, rng), None, length=epochs_per_eval
    )
    return ts, env_carry, rng, metrics

  # --- eval ------------------------------------------------------------------

  if eval_env is None:
    eval_env = environment
  eval_env_wrapped = wrap_for_dva_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=rand_eval,
  )
  import functools  # pylint: disable=g-import-not-at-top

  if vision:

    class _VisionEvaluator:
      """brax acting.Evaluator lookalike that renders depth per step.

      Episode reward = masked sum of rewards until each env's first done
      over an episode_length unroll — the same quantity brax's EvalWrapper
      aggregates for full-length episodes.
      """

      def __init__(self, key):
        self._key = key

        def run(params, warp_data, reset_keys, act_key):
          normalizer, actor_params = params
          state = eval_env_wrapped.reset(reset_keys)

          warp_data, frame = render_depth(
              warp_data, state.data.qpos, state.data.qvel
          )
          stack = jp.repeat(frame[..., None], frame_stack, axis=-1)

          def estep(carry, _):
            state, warp_data, stack, ep_rew, alive, length, act_key = carry
            obs_actor = _select_obs(state.obs, actor_obs_key)
            if normalize_observations:
              obs_actor = rms_normalize(normalizer, obs_actor)
            proprio = _proprio_of(obs_actor.astype(jp.float32))
            mu, logstd = jax.vmap(
                lambda d, p: actor.apply(actor_params, d, p)
            )(stack, proprio)
            if deterministic_eval:
              raw = mu
            else:
              act_key, sample_key = jax.random.split(act_key)
              raw = mu + jp.exp(logstd) * jax.random.normal(
                  sample_key, mu.shape, dtype=mu.dtype
              )
            state = eval_env_wrapped.step(state, jp.tanh(raw))
            ep_rew = ep_rew + state.reward * alive
            length = length + alive
            alive = alive * (1.0 - state.done)
            warp_data, frame = render_depth(
                warp_data, state.data.qpos, state.data.qvel
            )
            stack = _update_frame_stack(stack, frame, state.done)
            return (
                state,
                warp_data,
                stack,
                ep_rew,
                alive,
                length,
                act_key,
            ), None

          init = (
              state,
              warp_data,
              stack,
              jp.zeros(num_eval_envs),
              jp.ones(num_eval_envs),
              jp.zeros(num_eval_envs),
              act_key,
          )
          carry, _ = jax.lax.scan(
              estep, init, None, length=episode_length // action_repeat
          )
          return carry[3], carry[5]  # ep_rew, length

        self._run = jax.jit(run)

      def run_evaluation(self, params, training_metrics):
        self._key, reset_key, act_key = jax.random.split(self._key, 3)
        reset_keys = jax.random.split(reset_key, num_eval_envs)
        ep_rew, ep_len = jax.device_get(
            self._run(params, assets.warp_data, reset_keys, act_key)
        )
        metrics = {
            'eval/episode_reward': float(np.mean(ep_rew)),
            'eval/episode_reward_std': float(np.std(ep_rew)),
            # ep_len counts env steps; report physics control steps like
            # brax's EvalWrapper does.
            'eval/avg_episode_length': float(np.mean(ep_len)) * action_repeat,
            **training_metrics,
        }
        return metrics

    evaluator = _VisionEvaluator(eval_key)
  else:
    evaluator = acting.Evaluator(
        eval_env_wrapped,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

  def _params(ts):
    return (ts.normalizer, ts.actor_params)

  # --- init env state (mirrors brax APG: scramble + one throwaway step) -----

  _jit_env_reset = jax.jit(env.reset)
  _jit_env_step = jax.jit(env.step)
  _jit_render_depth = jax.jit(render_depth) if vision else None

  def _fresh_env_carry(rng, scramble_rng):
    reset_keys = jax.random.split(rng, num_envs)
    env_state = _jit_env_reset(reset_keys)
    if scramble_time:
      env_state.info['steps'] = jp.round(
          jax.random.uniform(
              scramble_rng, (num_envs,), maxval=float(episode_length)
          )
      )
    env_state = _jit_env_step(
        env_state, jp.zeros((num_envs, env.action_size))
    )
    warp_data0 = None
    stack0 = None
    if vision:
      warp_data0, frame0 = _jit_render_depth(
          assets.warp_data, env_state.data.qpos, env_state.data.qvel
      )
      stack0 = jp.repeat(frame0[..., None], frame_stack, axis=-1)
    return (env_state, warp_data0, stack0)

  env_carry = _fresh_env_carry(env_key, scramble_key)

  # --- main loop --------------------------------------------------------------

  all_metrics = {}
  if num_evals > 1:
    all_metrics = evaluator.run_evaluation(
        _params(training_state), training_metrics={}
    )
    progress_fn(0, all_metrics)

  key, block_rng = jax.random.split(key)
  for it in range(num_evals_after_init):
    if reset_envs_per_block and it > 0:
      block_rng, fresh_rng, fresh_scramble_rng = jax.random.split(
          block_rng, 3
      )
      env_carry = _fresh_env_carry(fresh_rng, fresh_scramble_rng)
    training_state, env_carry, block_rng, tmetrics = training_block(
        training_state, env_carry, block_rng
    )
    tmetrics = jax.tree.map(lambda x: jax.device_get(x), tmetrics)
    # NaN policy (deviation from dva.py:341-343, which hard-raises on the
    # first bad gradient): transient sim/render NaNs are survivable — the
    # env self-heals on truncation and apply_if_finite skips the poisoned
    # update — so only warn on them. Raise only when the actor params
    # themselves went non-finite (protection exhausted): that run is dead.
    gnb = tmetrics['grad_norm_before_clip']
    skipped = int(tmetrics['actor_notfinite_total'][-1])
    if not np.isfinite(gnb).all() or (gnb > 1e8).any() or skipped > 0:
      bad = int(np.sum(~np.isfinite(gnb) | (gnb > 1e8)))
      print(
          f'WARNING block {it}: {bad} epochs with NaN/huge actor grads, '
          f'{skipped} updates skipped so far (apply_if_finite)'
      )
    params_finite = all(
        bool(np.isfinite(x).all())
        for x in jax.tree_util.tree_leaves(
            jax.device_get(training_state.actor_params)
        )
    )
    if not params_finite:
      raise ValueError(
          f'Actor params non-finite after block {it} '
          f'(consecutive NaN protection exhausted).'
      )
    ep_count = float(tmetrics['episode_count'].sum())
    training_metrics = {
        'training/actor_loss': float(tmetrics['actor_loss'][-1]),
        'training/value_loss': float(tmetrics['value_loss'][-1]),
        'training/grad_norm_before_clip': float(
            tmetrics['grad_norm_before_clip'].mean()
        ),
        'training/grad_norm_after_clip': float(
            tmetrics['grad_norm_after_clip'].mean()
        ),
        'training/episode_return': (
            float(tmetrics['episode_return_sum'].sum() / ep_count)
            if ep_count > 0
            else float('nan')
        ),
        'training/episode_length': (
            float(tmetrics['episode_len_sum'].sum() / ep_count)
            if ep_count > 0
            else float('nan')
        ),
    }
    epoch = (it + 1) * epochs_per_eval
    steps = epoch * num_envs * horizon_length * action_repeat
    all_metrics = evaluator.run_evaluation(
        _params(training_state), training_metrics
    )
    progress_fn(it + 1, all_metrics)
    if checkpoint_fn is not None:
      checkpoint_fn(steps, jax.device_get(_params(training_state)))

  params = jax.device_get(_params(training_state))
  if return_training_state:
    full_state = jax.device_get((
        training_state.normalizer,
        training_state.actor_params,
        training_state.critic_params,
        training_state.target_critic_params,
    ))
    return make_policy, params, all_metrics, full_state
  return make_policy, params, all_metrics


def vision_video_rollout(
    environment: mjx_env.MjxEnv,
    params: Any,
    episode_length: int,
    cam_res: Tuple[int, int] = (64, 64),
    frame_stack: int = 3,
    depth_scale: float = 3.0,
    vision_camera: str = 'egocentric',
    vision_proprio: str = 'full',
    encoder_dim: int = 128,
    actor_hidden: Sequence[int] = (128, 64, 32),
    normalize_observations: bool = True,
    seed: int = 0,
):
  """Deterministic single-env rollout of a vision policy for video rendering.

  Returns (states, total_reward, depth_frames) — states for env.render, and
  the (T, H, W) depth stream for a side-by-side visualization.
  """
  assets = VisionAssets(environment.mj_model, 1, vision_camera, cam_res)
  render = make_depth_render_fn(assets, depth_scale)
  actor = VisionActor(
      action_size=environment.action_size,
      encoder_dim=encoder_dim,
      hidden=tuple(actor_hidden),
  )
  normalizer, actor_params = params
  if vision_proprio == 'full':
    proprio_fn = lambda x: x
  elif vision_proprio == 'no_linvel':
    proprio_fn = lambda x: jp.concatenate([x[..., :42], x[..., 45:]], -1)
  else:
    raise ValueError(f'Unknown vision_proprio={vision_proprio!r}')

  @jax.jit
  def rollout_step(state, warp_data, stack):
    # Act on the incoming stack (whose newest frame renders the current
    # state), then step and render the NEW state so the returned frame is
    # aligned with the returned state (depth_frames[i] == render(states[i])).
    obs = state.obs
    if normalize_observations:
      obs = rms_normalize(normalizer, obs)
    proprio = proprio_fn(obs.astype(jp.float32))
    mu, _ = actor.apply(actor_params, stack, proprio)
    state = environment.step(state, jp.tanh(mu))
    warp_data, frame = render(
        warp_data, state.data.qpos[None], state.data.qvel[None]
    )
    stack = jp.concatenate([stack[..., 1:], frame[0][..., None]], axis=-1)
    return state, warp_data, stack, frame[0]

  state = jax.jit(environment.reset)(jax.random.PRNGKey(seed))
  warp_data, frame0 = jax.jit(render)(
      assets.warp_data, state.data.qpos[None], state.data.qvel[None]
  )
  stack = jp.repeat(frame0[0][..., None], frame_stack, axis=-1)

  states = [state]
  depth_frames = [np.asarray(frame0[0])]
  total_reward = 0.0
  for _ in range(episode_length):
    state, warp_data, stack, frame = rollout_step(state, warp_data, stack)
    states.append(state)
    depth_frames.append(np.asarray(frame))
    total_reward += float(state.reward)
    if bool(state.done):
      break
  return states, total_reward, np.stack(depth_frames)
