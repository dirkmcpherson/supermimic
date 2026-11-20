"""Build the offline dataset: constants, demos (recorded and replayed),
exploration branches.

Runs on the CPU only.

  python -m supermimic.manip.scripts.prepare_data --task square --out data/square \
    --demos <square_ph.hdf5> <square_mh.hdf5>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from supermimic.manip.collect import (
  KIND_NAMES,
  BranchCfg,
  collect_branches,
  recorded_episodes,
  replay_demos,
  start_bank,
)
from supermimic.manip.demos import load_demos
from supermimic.manip.sim import SIMS, SimCfg, make_sim


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=sorted(SIMS), required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--demos", nargs="+", required=True)
  parser.add_argument("--demo-limit", type=int, default=None)
  parser.add_argument("--branch-steps", type=int, default=800_000)
  parser.add_argument("--num-workers", type=int, default=16)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  sim_cfg = SimCfg(task=args.task)
  meta: dict = {"args": vars(args)}

  sim = make_sim(sim_cfg, seed=args.seed)
  constants = sim.constants()
  np.savez(out / "constants.npz", **constants)
  grip_cmd_step = float(constants["grip_cmd_step"])
  flat_layout = sim.flat_layout()
  del sim

  demos = []
  for path in args.demos:
    loaded = load_demos(
      path,
      grip_cmd_step,
      flat_layout,
      SIMS[args.task].variant_from_model_xml,
      limit=args.demo_limit,
    )
    print(
      f"[INFO] {path}: {len(loaded)} demos, {sum(len(d.actions) for d in loaded)} steps"
    )
    demos += loaded

  start = time.time()
  replays, obj_error = replay_demos(demos, sim_cfg, args.num_workers, args.seed)
  np.savez_compressed(out / "demos.npz", **replays)
  meta["demo_replay"] = {
    "demos": len(demos),
    "transitions": len(replays["state"]),
    "success_rate": float(replays["episode_success"].mean()),
    "obj_error_m_median": float(np.median(obj_error)),
    "obj_error_m_p90": float(np.percentile(obj_error, 90)),
    "obj_error_m_max": float(obj_error.max()),
    "seconds": time.time() - start,
  }
  print(f"[INFO] demo replay: {json.dumps(meta['demo_replay'])}")

  np.savez_compressed(out / "demos_recorded.npz", **recorded_episodes(demos))
  states, variants = start_bank(demos)
  np.savez_compressed(
    out / "start_bank.npz", states=states.astype(np.float32), variants=variants
  )

  start = time.time()
  branches = collect_branches(
    demos, args.branch_steps, sim_cfg, BranchCfg(), args.num_workers, args.seed + 1
  )
  np.savez_compressed(out / "branches.npz", **branches)
  seconds = time.time() - start
  ends = np.flatnonzero(branches["truncated"])
  kinds = branches["kind"][ends]
  meta["branches"] = {
    "transitions": len(branches["state"]),
    "episodes": len(ends),
    "seconds": seconds,
    "steps_per_second": len(branches["state"]) / seconds,
    "episodes_by_kind": {
      KIND_NAMES[k]: int((kinds == k).sum()) for k in np.unique(kinds)
    },
    "success_rate_by_kind": {
      KIND_NAMES[k]: float(branches["episode_success"][kinds == k].mean())
      for k in np.unique(kinds)
    },
  }
  print(f"[INFO] branches: {json.dumps(meta['branches'])}")
  (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
