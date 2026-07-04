"""Merge crate-recovery reference ensembles produced by harvest_crate_recoveries.py.

Each input npz holds member 0 = the nominal reference plus harvested
recoveries. The merge keeps a single nominal (member 0, verified identical
across inputs) and concatenates all recovery members.

Run:
  python learning/merge_crate_recovery_refs.py \
      references/go2_crate_climb_recovery_refs.npz \
      references/go2_crate_climb_recovery_refs_topup.npz \
      --out references/go2_crate_climb_recovery_refs_merged.npz
"""

import argparse

import numpy as np

KEYS = ("qpos", "qvel", "actions", "start_frame", "noise_qpos", "noise_qvel",
        "final_feet_on", "final_head_err")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("inputs", nargs="+")
  p.add_argument("--out", required=True)
  args = p.parse_args()

  ensembles = [dict(np.load(path)) for path in args.inputs]
  first = ensembles[0]
  for path, ens in zip(args.inputs[1:], ensembles[1:]):
    if not np.allclose(ens["qpos"][0], first["qpos"][0]):
      raise ValueError(f"{path}: member 0 (nominal) differs from "
                       f"{args.inputs[0]}'s — refusing to merge.")

  merged = {}
  for k in KEYS:
    parts = [first[k]] + [ens[k][1:] for ens in ensembles[1:]]
    merged[k] = np.concatenate(parts, axis=0)
  merged["dt"] = first["dt"]

  np.savez(args.out, **merged)
  counts = [ens["qpos"].shape[0] - 1 for ens in ensembles]
  print(f"Merged {'+'.join(str(c) for c in counts)} recoveries + nominal "
        f"-> {merged['qpos'].shape[0]} members at {args.out}")


if __name__ == "__main__":
  main()
