import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ['MUJOCO_GL'] = 'egl'

os.environ['JAX_LOG_COMPILES'] = '0'

import copy
import json
import shutil
import sys
import time

import functools
from pathlib import Path

import jax.numpy as jp
import numpy as np
import jax
from mujoco import mjx
print("JAX Device:", jax.devices())
from jax import config # Analytical gradients work much better with double precision.
# debug_nans makes ANY NaN fatal. Good for pure-APG debugging; wrong for the
# dva trainer, which tolerates transient sim/render NaNs (apply_if_finite
# skips poisoned updates and the env self-heals at truncation). Default: on
# for apg, off for dva; override with GO2_SAMPLING_DEBUG_NANS=0/1.
_debug_nans_default = "1" if os.environ.get("GO2_SAMPLING_ALGO", "apg") == "apg" else "0"
config.update(
    "jax_debug_nans",
    os.environ.get("GO2_SAMPLING_DEBUG_NANS", _debug_nans_default) == "1",
)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')

print("jax.devices():", jax.devices())
print("local_device_count:", jax.local_device_count())

from absl import logging
logging.set_verbosity(logging.DEBUG)

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground._src.locomotion.go2 import go2_online_ref
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

# Per-task defaults. The sampling reference for seq_jump is one-shot (5 jump
# stages chained back-to-back), while the trot reference is a steady-state gait
# cycle that should loop. Cap the per-episode training horizon so APG unrolls
# stay cheap; the cap is clipped to the available reference length below.
#
# `env_overrides` are dotted-path overrides applied to the apg_env_cfg
# ConfigDict after registry.get_default_config(). For trotting we mirror the
# kp=30, kd=0 in dial-mpc's UnitreeGo2EnvConfig + unitree_go2_trot.yaml so the
# APG sim PD matches the controller used to generate the trot reference. We
# keep Go2SampleAPG's sim_dt=0.002 substepping for APG gradient stability and
# leave the per-joint action_scale list as is (Go2SampleAPG uses additive
# action_loc + action*action_scale, not dial-mpc's linear-range act2joint).
TASK_DEFAULTS = {
    # Reference files are the *_pg.npz outputs of
    # learning/fix_reference_leg_order.py: the raw sampling rollouts are
    # recorded in dial-mpc joint order (leg blocks FR,FL,RR,RL) and must be
    # permuted to playground order (FL,FR,RL,RR) before training on them.
    "seq_jump": dict(
        reference_path="references/go2_seqjump_sampling_ref_pg.npz",
        reference_loop=False,
        max_episode_length=128,
        reference_video="go2_seqjump_apg_reference.mp4",
        env_overrides={},
    ),
    # dial-mpc unitree_go2_crate_climb port: one-shot climb onto the 0.6 m
    # crate. The reference is sampled ON THE TRAINING PLANT (position-PD,
    # 2 ms substeps, real calf torque limits) via
    #   JAX_ENABLE_X64=1 python learning/play_go2_sampling.py \
    #     --task crate_climb_pg --num_steps 130 --hsample 25 \
    #     --num_diffuse 4 --horizon_diffuse_factor 1.0 --seed 1
    # NO leg-order conversion (native playground order). The dial-generated
    # reference (go2_crate_climb_sampling_ref_pg.npz) is NOT open-loop
    # reproducible on this plant (coarse 20 ms torque-clipped integration
    # there) — do not train on it.
    "crate_climb": dict(
        # Recovery ENSEMBLE (learning/harvest_crate_recoveries.py +
        # merge_crate_recovery_refs.py): member 0 is the nominal
        # go2_crate_climb_pg_sampling_ref.npz, members 1..59 are closed-loop
        # DIAL-MPC recoveries from noised reference states grafted onto the
        # nominal prefix — the demonstrated-feedback funnel a single
        # trajectory lacks.
        reference_path="references/go2_crate_climb_recovery_refs_merged.npz",
        reference_loop=False,
        # Shorter than the 101-frame reference on purpose: RSI then spreads
        # episode starts over [0, 37), DeepMimic-style, so the policy
        # practices the leap/mantle phases directly instead of having to
        # nail the whole approach first. (At 100 every episode starts at
        # frame 0 and training converges to standing in front of the crate.)
        max_episode_length=64,
        reference_video="go2_crate_climb_apg_reference.mp4",
        env_overrides={
            "Kp": 30.0,
            "Kd": 0.65,
            "ctrl_dt": 0.02,
            "env.terrain": "crate",
            # No mid-leap velocity kicks for a one-shot acrobatic motion.
            "pert_config.enable": False,
            # The mantle needs joint targets far outside the locomotion
            # action envelope (the expert COMMANDS front thighs to -0.93 rad
            # vs the default floor of +0.10, and the policy converges to
            # parking in front of the crate). Sized to the recorded expert
            # command envelope around action_loc = (0, 0.9, -1.8).
            "env.action_scale": [0.4, 1.9, 0.9] * 4,
            # DeepMimic ingredient: off-reference states end the episode
            # (bootstrap 0), so parking in front of the crate collects
            # nothing. Without this, every optimizer tried converged to
            # parking/collapsing instead of climbing.
            "env.terminate_on_deviation": True,
            # Deviation measured against the NEAREST ensemble member (the
            # demonstrated corridor). Against only the tracked member,
            # injection + exploration noise cut training episodes to ~13
            # steps — credit never crossed the launch.
            "env.deviation_nearest_member": True,
            # Widen the RSI start tube to the noise level the recovery
            # ensemble was harvested at (harvest_crate_recoveries.py tier 0;
            # tier 1 is deliberately wider than the training tube), so noisy
            # starts have a demonstrated correction nearby.
            "env.rsi_noise_qpos": 0.02,
            "env.rsi_noise_qvel": 0.05,
        },
    ),
    "trot": dict(
        reference_path="references/go2_trot_sampling_ref_pg.npz",
        reference_loop=True,
        max_episode_length=256,
        reference_video="go2_trot_apg_reference.mp4",
        env_overrides={
            # dial-mpc UnitreeGo2EnvConfig: kp=30.0, kd=0.0.
            "Kp": 30.0,
            "Kd": 0.0,
            # dial-mpc unitree_go2_trot.yaml: dt=0.02 (matches Go2SampleAPG
            # default), kept here for documentation/explicitness.
            "ctrl_dt": 0.02,
        },
    ),
    # Stage-1 online reference: the DIAL-MPC sampler runs at trainer startup
    # on the *training* model (go2_online_ref.py) instead of loading an
    # offline npz — no leg-order conversion, no loop wrap, dynamically
    # feasible reference by construction. Set GO2_SAMPLING_REFERENCE_PATH to
    # reuse a previously generated npz; GO2_SAMPLING_GENERATE_ONLY=1 to stop
    # after generation+validation (for inspecting the reference).
    # trot_online on the +-5 cm hfield (env.terrain="rough"). Intended use:
    # reuse the FLAT stage-1 reference via GO2_SAMPLING_REFERENCE_PATH as a
    # gait prior — the tracking tolerance absorbs terrain-scale deviations.
    # (Without a reuse path this would regenerate the reference by sampling
    # ON the rough model — a different, untested experiment.)
    "trot_online_rough": dict(
        reference_path=None,
        reference_loop=False,
        max_episode_length=240,
        reference_video="go2_trot_online_rough_reference.mp4",
        online_reference=True,
        env_overrides={
            "Kp": 30.0,
            "Kd": 0.65,
            "ctrl_dt": 0.02,
            "env.terrain": "rough",
        },
    ),
    "trot_online": dict(
        reference_path=None,
        reference_loop=False,
        max_episode_length=240,
        reference_video="go2_trot_online_reference.mp4",
        online_reference=True,
        env_overrides={
            # Kd maps to dof_damping (base.py): 0.65 is the faithful dial-mpc
            # plant (its yaml kd=0 was the manual-PD knob riding on XML
            # damping 0.65). On the undamped Kd=0 plant the sampler finds a
            # crouched trot (median base z 0.19 vs 0.30 target); damping
            # restores the upright gait and better-conditions BPTT gradients.
            "Kp": 30.0,
            "Kd": 0.65,
            "ctrl_dt": 0.02,
        },
    ),
}


def _apply_env_overrides(cfg, overrides):
    """Apply dotted-path overrides to a ConfigDict.

    Example: `_apply_env_overrides(cfg, {"Kp": 30.0, "env.impratio": 50})`
    sets cfg.Kp=30.0 and cfg.env.impratio=50.
    """
    for dotted_key, value in overrides.items():
        target = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
TASK = os.environ.get("GO2_SAMPLING_TASK", "seq_jump")
if TASK not in TASK_DEFAULTS:
  raise ValueError(
      f"Unknown GO2_SAMPLING_TASK='{TASK}'. "
      f"Supported: {sorted(TASK_DEFAULTS)}"
  )
_task_cfg = TASK_DEFAULTS[TASK]

# Trainer selection: "apg" = brax analytic policy gradient (default),
# "dva" = D.VA decoupled first-order method (JAX port, dva_train.py) with an
# asymmetric TD-lambda critic and detached-obs actor. Both train the same env
# on the same reference; dva is the state-mode stage of the visual pipeline.
ALGO = os.environ.get("GO2_SAMPLING_ALGO", "apg")
if ALGO not in ("apg", "dva"):
    raise ValueError(
        f"Unknown GO2_SAMPLING_ALGO='{ALGO}'. Supported: apg, dva"
    )

_ref_override = os.environ.get("GO2_SAMPLING_REFERENCE_PATH")
_ref_default = _task_cfg["reference_path"]
REFERENCE_PATH = (
    Path(_ref_override or _ref_default)
    if (_ref_override or _ref_default)
    else None
)
ONLINE_REFERENCE = bool(_task_cfg.get("online_reference", False))
GENERATE_ONLY = os.environ.get("GO2_SAMPLING_GENERATE_ONLY", "0") == "1"
VALIDATE_REFERENCE = os.environ.get("GO2_VALIDATE_REFERENCE", "0") == "1"
REFERENCE_VIDEO_PATH = Path(
    os.environ.get("GO2_REFERENCE_VIDEO_PATH", _task_cfg["reference_video"])
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = (
        f"{env_name}-{TASK}-{timestamp}"
        if ALGO == "apg"
        else f"{env_name}-{TASK}-{ALGO}-{timestamp}"
    )
    if RUN_SUFFIX is not None:
        exp_name += f"-{RUN_SUFFIX}"
    logdir = LOGDIR.resolve() / exp_name
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment name: {exp_name}")
    print(f"Logs are being stored in: {logdir}")
    print(f"Checkpoint path: {ckpt_path}")

    apg_env_cfg = registry.get_default_config(env_name)
    apg_env_cfg['env']['reset2ref'] = False
    apg_env_cfg['env']['reference_state_init'] = True
    apg_env_cfg['env']['reference_loop'] = _task_cfg["reference_loop"]
    apg_env_cfg['pert_config']['enable'] = True
    apg_env_cfg['pert_config']['velocity_kick'] = [0.0, 3.0]
    apg_env_cfg['env']['impratio'] = 100
    # Task-specific env overrides (e.g. Kp/Kd for trot match dial-mpc).
    _apply_env_overrides(apg_env_cfg, _task_cfg["env_overrides"])

    if ONLINE_REFERENCE and REFERENCE_PATH is not None and not REFERENCE_PATH.exists():
        # An explicit reuse path that doesn't exist is a user error (likely a
        # typo) — don't silently pay for a fresh generation on a different
        # reference than the one asked for.
        raise FileNotFoundError(
            f"GO2_SAMPLING_REFERENCE_PATH points to a missing file: "
            f"{REFERENCE_PATH}"
        )
    if ONLINE_REFERENCE and REFERENCE_PATH is None:
        # Stage-1 online reference: run the DIAL-MPC sampler on the training
        # model at startup. gen_cfg has no reference_path set, so the env
        # falls back to the (unused) cosine reference during generation.
        gen_cfg = copy.deepcopy(apg_env_cfg)
        gen_env = registry.load(env_name, gen_cfg)
        generator = go2_online_ref.OnlineReferenceGenerator(gen_env)
        gen_seed = int(os.environ.get("GO2_SAMPLING_GEN_SEED", "0"))
        print(
            f"Generating online reference on the training model "
            f"(N={generator.num_samples}, Hsample={generator.hsample}, "
            f"steps={generator.cfg.num_steps}, seed={gen_seed})..."
        )
        gen_t0 = time.time()
        gen_qpos, gen_qvel, gen_rews = generator.generate(
            jax.random.PRNGKey(gen_seed)
        )
        # Save before validating so a rejected trajectory stays inspectable.
        REFERENCE_PATH = logdir / "generated_reference.npz"
        np.savez(
            REFERENCE_PATH,
            qpos=gen_qpos,
            qvel=gen_qvel,
            rewards=gen_rews,
            dt=np.array(gen_env.dt),
        )
        print(
            f"Generated reference in {time.time() - gen_t0:.0f} s "
            f"(mean task reward {gen_rews.mean():.3f}), saved to "
            f"{REFERENCE_PATH}"
        )
        gen_stats = go2_online_ref.validate_reference(
            gen_qpos, gen_qvel, float(gen_env.dt)
        )
        print(f"Reference validation passed: {gen_stats}")

    if REFERENCE_PATH is None:
        raise ValueError(
            f"Task '{TASK}' has no reference path and online_reference is "
            "not enabled."
        )
    if ONLINE_REFERENCE:
        # Keep the run self-contained: eval resolves
        # <run_dir>/generated_reference.npz, so a reused reference is copied
        # into this run's directory.
        _canonical_ref = logdir / "generated_reference.npz"
        if REFERENCE_PATH.resolve() != _canonical_ref.resolve():
            shutil.copy2(REFERENCE_PATH, _canonical_ref)
            REFERENCE_PATH = _canonical_ref
    apg_env_cfg['env']['reference_path'] = REFERENCE_PATH.as_posix()
    if REFERENCE_PATH.exists():
        with np.load(REFERENCE_PATH) as ref:
            _ref_qpos = ref["qpos"]
            # Reference ensembles (harvest_crate_recoveries.py) are stored
            # [M, T, nq]; single references are [T, nq].
            ref_len = int(
                _ref_qpos.shape[1] if _ref_qpos.ndim == 3 else _ref_qpos.shape[0]
            )
            apg_env_cfg['episode_length'] = min(
                _task_cfg["max_episode_length"], ref_len - 1
            )
    else:
        if TASK == "crate_climb":
            _hint = (
                "Generate the training-plant reference with "
                "`JAX_ENABLE_X64=1 python learning/play_go2_sampling.py "
                "--task crate_climb_pg --num_steps 130 --hsample 25 "
                "--num_diffuse 4 --horizon_diffuse_factor 1.0 --seed 1 "
                "--reference_output_path "
                "references/go2_crate_climb_pg_sampling_ref.npz` (references "
                "from the pg plant are already in playground joint order — "
                "do NOT run fix_reference_leg_order.py), or build the "
                "recovery ensemble with "
                "`python learning/harvest_crate_recoveries.py`."
            )
        else:
            _hint = (
                f"Generate the raw rollout with `python "
                f"learning/play_go2_sampling.py --task {TASK} "
                f"--reference_output_path <path>.npz` and convert it to "
                f"playground joint order with "
                f"`python learning/fix_reference_leg_order.py`."
            )
        raise FileNotFoundError(
            f"Sampling reference not found: {REFERENCE_PATH}. {_hint}"
        )
    print(
        f"Training Go2SampleAPG on task='{TASK}': reference={REFERENCE_PATH}, "
        f"loop={apg_env_cfg['env']['reference_loop']}, "
        f"episode_length={apg_env_cfg['episode_length']}, "
        f"Kp={apg_env_cfg['Kp']}, Kd={apg_env_cfg['Kd']}, "
        f"ctrl_dt={apg_env_cfg['ctrl_dt']}, sim_dt={apg_env_cfg['sim_dt']}."
    )

    env = registry.load(env_name, apg_env_cfg)
    # Eval must measure imitation quality, not kick robustness: keep the eval
    # env identical except for perturbations. (Domain randomization is still
    # applied to eval inside brax's apg.train; acceptable for now.)
    eval_env_cfg = copy.deepcopy(apg_env_cfg)
    eval_env_cfg['pert_config']['enable'] = False
    eval_env = registry.load(env_name, eval_env_cfg)
    if VALIDATE_REFERENCE or GENERATE_ONLY:
        ref_video_path = (
            logdir / _task_cfg["reference_video"]
            if GENERATE_ONLY
            else REFERENCE_VIDEO_PATH
        )
        validate_reference_motion(env, ref_video_path)
    if GENERATE_ONLY:
        print("GO2_SAMPLING_GENERATE_ONLY=1 — exiting before training.")
        sys.exit(0)

    if ALGO == "dva":
        from mujoco_playground.experimental.learning import dva_train

        dva_params = locomotion_params.dva_config(env_name)
        dva_params["episode_length"] = int(apg_env_cfg["episode_length"])
        # Hyperparameter overrides for sweeps, e.g.
        # GO2_SAMPLING_DVA_OVERRIDES='{"actor_logstd_init": -2.0}'. Values are
        # recorded in training_params.json like everything else.
        _dva_overrides = os.environ.get("GO2_SAMPLING_DVA_OVERRIDES")
        if _dva_overrides:
            for _k, _v in json.loads(_dva_overrides).items():
                if _k not in dva_params:
                    raise KeyError(f"Unknown DVA override: {_k}")
                dva_params[_k] = (
                    tuple(_v) if isinstance(dva_params[_k], (tuple, list)) else _v
                )

        with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
            json.dump(apg_env_cfg.to_dict(), fp, indent=4)
        with open(
            ckpt_path / "training_params.json", "w", encoding="utf-8"
        ) as fp:
            json.dump({"algo": "dva", **dva_params.to_dict()}, fp, indent=4)
        print(f"DVA params: {dva_params}")

        wandb.init(project="mujoco-playground-apg-go2", name=exp_name)

        num_evals_after_init = max(dva_params["num_evals"] - 1, 1)
        epochs_per_eval = dva_params["max_epochs"] // num_evals_after_init
        steps_per_eval = (
            epochs_per_eval
            * dva_params["num_envs"]
            * dva_params["horizon_length"]
            * dva_params["action_repeat"]
        )

        def progress_dva(it, metrics):
            wandb.log(metrics, step=it * steps_per_eval)
            print(
                f"[dva eval {it}] steps={it * steps_per_eval} "
                f"eval/episode_reward="
                f"{float(metrics['eval/episode_reward']):.1f} "
                f"+- {float(metrics['eval/episode_reward_std']):.1f}"
            )

        def checkpoint_dva(steps, params):
            model.save_params((ckpt_path / str(steps)).as_posix(), params)

        # Optional warm start (e.g. a behavior-cloned actor):
        # GO2_SAMPLING_DVA_RESTORE=<path to (normalizer, actor_params) file>.
        _restore_path = os.environ.get("GO2_SAMPLING_DVA_RESTORE")
        _restore_params = (
            model.load_params(_restore_path) if _restore_path else None
        )
        if _restore_path:
            print(f"Warm-starting actor from {_restore_path}")

        train_t0 = time.time()
        make_policy, dva_final_params, dva_metrics = dva_train.train(
            environment=env,
            eval_env=eval_env,
            randomization_fn=randomizer,
            progress_fn=progress_dva,
            checkpoint_fn=checkpoint_dva,
            restore_params=_restore_params,
            **dva_params.to_dict(),
        )
        wandb.finish()
        print(f"DVA training finished in {time.time() - train_t0:.0f} s.")

        total_steps = (
            dva_params["max_epochs"]
            * dva_params["num_envs"]
            * dva_params["horizon_length"]
            * dva_params["action_repeat"]
        )
        model.save_params(
            (ckpt_path / str(total_steps)).as_posix(), dva_final_params
        )
        print(f"Saved DVA policy params to {ckpt_path / str(total_steps)}.")

        # Render the deterministic policy on the kick-free eval env.
        if dva_params["vision"]:
            rollout, video_reward, depth_frames = (
                dva_train.vision_video_rollout(
                    eval_env,
                    dva_final_params,
                    int(apg_env_cfg["episode_length"]),
                    cam_res=tuple(dva_params["cam_res"]),
                    frame_stack=int(dva_params["frame_stack"]),
                    depth_scale=float(dva_params["depth_scale"]),
                    vision_camera=dva_params["vision_camera"],
                    vision_proprio=dva_params["vision_proprio"],
                    encoder_dim=int(dva_params["encoder_dim"]),
                    actor_hidden=tuple(dva_params["actor_hidden"]),
                    normalize_observations=bool(
                        dva_params["normalize_observations"]
                    ),
                )
            )
            np.save(logdir / "eval_depth_frames.npy", depth_frames)
            print(f"Saved policy-view depth stream to {logdir}/eval_depth_frames.npy")
        else:
            inference_fn = jax.jit(
                make_policy(dva_final_params, deterministic=True)
            )
            jit_reset = jax.jit(eval_env.reset)
            jit_step = jax.jit(eval_env.step)
            state = jit_reset(jax.random.PRNGKey(0))
            rollout = [state]
            video_rng = jax.random.PRNGKey(1)
            video_reward = 0.0
            for _ in tqdm(
                range(int(apg_env_cfg["episode_length"])), desc="Eval rollout"
            ):
                video_rng, act_rng = jax.random.split(video_rng)
                action, _ = inference_fn(state.obs, act_rng)
                state = jit_step(state, action)
                rollout.append(state)
                video_reward += float(state.reward)
                if bool(state.done):
                    print(
                        f"Eval rollout terminated early at step {len(rollout) - 1}."
                    )
                    break
        print(f"Eval rollout total reward: {video_reward:.1f}")
        render_every = 2
        frames = eval_env.render(
            rollout[::render_every], height=480, width=640, camera="track"
        )
        video_path = logdir / f"eval_{TASK}_dva.mp4"
        media.write_video(
            video_path.as_posix(),
            frames,
            fps=1.0 / (eval_env.dt * render_every),
        )
        print(f"Saved DVA eval video to {video_path}")
        sys.exit(0)

    apg_params = locomotion_params.brax_apg_config(env_name)
    # Plumb the per-task episode cap into the trainer's EpisodeWrapper (the
    # env-config episode_length is not read by Go2SampleAPG itself), and make
    # eval report the deterministic policy instead of sampled actions.
    apg_params["episode_length"] = int(apg_env_cfg["episode_length"])
    apg_params["deterministic_eval"] = True
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