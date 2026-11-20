"""Re-execute the demonstration tapes in the simulator and store their full
states, with the constants of the built environment.

  python -m supermimic.restock.scripts.prepare_data --tapes <dir> --out <dir> \
      --repo <task repository> --sim-python <python of the simulator env>

A tape that ends in the environment's terminal (home, or a tipped can) is cut
there, as an online episode would be; rows the tape holds after that point are
not used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from supermimic.restock import layout as L
from supermimic.restock.calibration import rule_agreement
from supermimic.restock.representation import RestockRepresentation
from supermimic.restock.vec_env import SimCfg, WorkerPool


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tapes", required=True, help="directory of .npz tapes")
  parser.add_argument("--out", required=True)
  parser.add_argument("--repo", required=True)
  parser.add_argument("--sim-python", required=True)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--limit", type=int, default=None)
  args = parser.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  tapes = sorted(str(p) for p in Path(args.tapes).glob("*.npz"))[: args.limit]
  if not tapes:
    raise SystemExit(f"no tapes in {args.tapes}")

  start = time.time()
  workers = min(args.workers, len(tapes))
  pool = WorkerPool(
    workers, SimCfg(args.repo, args.sim_python, str(out / "workers")), seed=0
  )
  print(f"[DATA] {workers} simulators up in {time.time() - start:.0f} s", flush=True)
  try:
    constants = pool.constants()
    replays = []
    for first in range(0, len(tapes), workers):
      chunk = tapes[first : first + workers]
      replays += pool.call("replay", [(p,) for p in chunk], list(range(len(chunk))))
      print(f"[DATA] {len(replays)}/{len(tapes)} tapes", flush=True)
  finally:
    pool.close()
  np.savez(out / "constants.npz", **constants)

  columns: dict[str, list[np.ndarray]] = {
    k: []
    for k in ("state", "next_state", "action", "prev_action", "terminated", "truncated", "home", "tipped", "tape")
  }  # fmt: skip
  home_i, tipped_i = L.STAGES.index("home"), L.STAGES.index("tipped")
  rows = []
  for index, r in enumerate(replays):
    n = len(r["actions"])
    last = np.zeros(n, dtype=bool)
    last[-1] = True
    ended = bool(r["terminated"][-1])
    columns["state"].append(r["states"])
    columns["next_state"].append(r["next_states"])
    columns["action"].append(r["actions"])
    columns["prev_action"].append(
      np.concatenate([np.zeros((1, L.ACTION_DIM), np.float32), r["actions"][:-1]])
    )
    columns["terminated"].append(r["terminated"])
    columns["truncated"].append(last & ~r["terminated"])
    columns["home"].append(last & ended & bool(r["stages"][home_i]))
    columns["tipped"].append(last & ended & bool(r["stages"][tipped_i]))
    columns["tape"].append(np.full(n, index))
    rows.append(
      {
        "name": r["name"],
        "decisions": n,
        "tape_decisions": int(r["tape_decisions"]),
        "stages": {s: bool(v) for s, v in zip(L.STAGES, r["stages"], strict=True)},
        "start": r["start"],
        "start_gap_m": float(r["start_gap_m"]),
        "can_dev_m": float(r["can_dev_m"]),
        "joint_dev_rad": float(r["joint_dev_rad"]),
        "tape_end_reason": None
        if r["tape_end_reason"] is None
        else str(r["tape_end_reason"]),
        "tape_end_decision": None
        if r["tape_end_decision"] is None
        else int(r["tape_end_decision"]),
      }
    )
  data = {k: np.concatenate(v) for k, v in columns.items()}
  data["sim_ctx"] = np.zeros((len(data["state"]), L.CTX_DIM), np.float32)
  np.savez(out / "demos.npz", **data)

  agreement = rule_agreement(RestockRepresentation(constants), data)
  stage_counts = {s: sum(r["stages"][s] for r in rows) for s in L.STAGES}
  meta = {
    "tapes": len(rows),
    "tape_dir": str(Path(args.tapes).resolve()),
    "transitions": int(len(data["state"])),
    "tape_decisions": int(sum(r["tape_decisions"] for r in rows)),
    "simulator_frames": int(len(data["state"])) * 4,
    "stages": stage_counts,
    "can_dev_m_median": float(np.median([r["can_dev_m"] for r in rows])),
    "can_dev_m_max": float(np.max([r["can_dev_m"] for r in rows])),
    "end_matches_tape": sum(
      r["tape_end_decision"] == r["decisions"]
      for r in rows
      if r["tape_end_decision"] is not None
    ),
    "seconds": time.time() - start,
    "rule_agreement": agreement,
    "tapes_detail": rows,
  }
  (out / "meta.json").write_text(json.dumps(meta, indent=2))
  print(
    f"[DATA] {meta['tapes']} tapes, {meta['transitions']} transitions, stages {stage_counts}"
  )
  print(
    f"[DATA] can deviation from the tapes: median {meta['can_dev_m_median'] * 1000:.2f} mm, max {meta['can_dev_m_max'] * 1000:.2f} mm"
  )
  print(f"[DATA] rule agreement: {json.dumps(agreement)}")


if __name__ == "__main__":
  main()
