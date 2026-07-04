"""Convert dial-mpc-ordered Go2 reference npz files to playground joint order.

The dial-mpc Go2 model orders leg joint blocks (FR, FL, RR, RL) while the
playground Go2 model orders them (FL, FR, RL, RR) — and the dial model's
name->side mapping is itself flipped (dial's "FR" foot sits at +y, the
physical left). References recorded by learning/play_go2_sampling.py store
qpos/qvel/actions in dial order, so loading them into Go2SampleAPG without
remapping swaps the left/right legs of the gait.

The fix is a pure relabeling: permute 3-joint leg blocks by [1, 0, 3, 2].
Validation: forward kinematics of the permuted qpos in the playground model
must reproduce the recorded world-frame feet positions (which are physical
ground truth, independent of labeling).

Usage:
  python learning/fix_reference_leg_order.py references/go2_trot_sampling_ref.npz
Writes <stem>_pg.npz next to the input.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PG_SCENE = (
    REPO_ROOT
    / "mujoco_playground/_src/locomotion/go2/xmls/scene_mjx_collision_free.xml"
)
LEG_PERM = [1, 0, 3, 2]
PG_FEET_GEOMS = ["FR", "FL", "RR", "RL"]
# The recorded feet_xpos row convention differs between reference files (the
# generator/model changed between generations), so it is auto-detected: after
# the qpos permutation, whichever row mapping matches playground FK is the
# one the file was recorded with.
FEET_ROW_CANDIDATES = {"identity": [0, 1, 2, 3], "swapped": [1, 0, 3, 2]}
# Recorded site_xpos can LAG qpos by one control step: MJX computes site
# positions at the start of a physics step, so data.site_xpos in a stepped
# state still reflects the pre-integration qpos (verified exact on the crate
# reference: feet_xpos[t] == FK(qpos[t-1]) to machine precision). Both
# alignments are tried; at high foot speeds only the lagged one matches.
FEET_SHIFT_CANDIDATES = (0, 1)


def permute_legs(arr: np.ndarray, joint_offset: int) -> np.ndarray:
  out = arr.copy()
  legs = out[:, joint_offset : joint_offset + 12].reshape(-1, 4, 3)
  out[:, joint_offset : joint_offset + 12] = legs[:, LEG_PERM].reshape(-1, 12)
  return out


def feet_fk(model, qpos: np.ndarray) -> np.ndarray:
  data = mujoco.MjData(model)
  feet_ids = [model.geom(n).id for n in PG_FEET_GEOMS]
  fk = np.empty((qpos.shape[0], 4, 3))
  for t in range(qpos.shape[0]):
    data.qpos[:] = qpos[t]
    mujoco.mj_forward(model, data)
    fk[t] = data.geom_xpos[feet_ids]
  return fk


def feet_error(
    fk: np.ndarray, feet_ref: np.ndarray, rows, shift: int
) -> tuple:
  n = fk.shape[0] - shift
  errs = np.linalg.norm(fk[:n] - feet_ref[shift:][:, rows], axis=-1).max(-1)
  return float(errs.mean()), float(errs.max())


def convert(path: Path) -> Path:
  with np.load(path) as ref:
    arrays = {k: ref[k] for k in ref.keys()}

  out = dict(arrays)
  out["qpos"] = permute_legs(arrays["qpos"], joint_offset=7)
  out["qvel"] = permute_legs(arrays["qvel"], joint_offset=6)
  if arrays.get("actions") is not None and arrays["actions"].size:
    out["actions"] = permute_legs(arrays["actions"], joint_offset=0)

  model = mujoco.MjModel.from_xml_path(PG_SCENE.as_posix())
  fk_after = feet_fk(model, out["qpos"])
  fk_before = feet_fk(model, arrays["qpos"])
  scored = {
      (rows_name, shift): feet_error(
          fk_after, arrays["feet_xpos"], rows, shift
      )
      for rows_name, rows in FEET_ROW_CANDIDATES.items()
      for shift in FEET_SHIFT_CANDIDATES
  }
  best_rows, best_shift = min(scored, key=lambda k: scored[k][0])
  before = feet_error(
      fk_before,
      arrays["feet_xpos"],
      FEET_ROW_CANDIDATES[best_rows],
      best_shift,
  )
  after = scored[(best_rows, best_shift)]
  # Write EXACT feet positions: playground FK of the permuted qpos, in
  # playground geom order (FR, FL, RR, RL) and aligned with qpos — strictly
  # better than the recorded (possibly one-step-stale) site positions.
  out["feet_xpos"] = fk_after
  print(
      f"{path.name}: recorded feet rows '{best_rows}', lag {best_shift} "
      f"step(s); feet FK error vs recorded (mean/max) "
      f"before={before[0]:.4f}/{before[1]:.4f} m -> "
      f"after={after[0]:.4f}/{after[1]:.4f} m"
  )
  if not (after[0] < before[0] * 0.5 and after[0] < 0.02):
    raise RuntimeError(
        "Permutation did not reduce feet FK error as expected; refusing to "
        "write output. Check whether this reference is already in playground "
        "order."
    )

  out_path = path.with_name(path.stem + "_pg.npz")
  np.savez(out_path, **out)
  print(f"Wrote {out_path}")
  return out_path


if __name__ == "__main__":
  paths = [Path(p) for p in sys.argv[1:]] or [
      REPO_ROOT / "references/go2_trot_sampling_ref.npz",
      REPO_ROOT / "references/go2_seqjump_sampling_ref.npz",
  ]
  for p in paths:
    convert(p)
