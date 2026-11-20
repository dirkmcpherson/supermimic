"""Score a trained actor the way the task's evaluator does: one episode per
evaluation start, in order, in one simulator process, deterministic actions.

  python -m supermimic.restock.scripts.score --run <train dir> --data <dir> \
      --repo <task repository> --sim-python <simulator python>

Writes ``score_<set>_<checkpoint>.json`` in the run directory. The statistic
of interest is ``stages.home``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from supermimic.restock import layout as L
from supermimic.restock.data import load_npz
from supermimic.restock.evaluate import eval_starts, run_episodes
from supermimic.restock.models import Actor
from supermimic.restock.representation import RestockRepresentation
from supermimic.restock.vec_env import SimCfg, WorkerPool


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True)
  parser.add_argument("--data", required=True)
  parser.add_argument("--repo", required=True)
  parser.add_argument("--sim-python", required=True)
  parser.add_argument("--checkpoint", default="final", choices=("final", "latest"))
  parser.add_argument("--set", default="rnd", dest="ic_set")
  parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="1 reproduces the task evaluator's single process",
  )
  parser.add_argument(
    "--actor",
    default="raw",
    choices=("raw", "ema"),
    help="the trained actor, or the averaged copy a run with --actor-ema keeps",
  )
  parser.add_argument(
    "--no-settle",
    action="store_true",
    help="diagnostic: skip the settle the task's evaluator runs after each episode",
  )
  args = parser.parse_args()

  run = Path(args.run)
  constants = load_npz(Path(args.data) / "constants.npz")
  cfg = json.loads((run / "runner_cfg.json").read_text())
  state = torch.load(
    run / f"checkpoint_{args.checkpoint}.pt", map_location="cpu", weights_only=False
  )
  representation = RestockRepresentation(constants)
  representation.load_state_dict(state["representation"])
  actor = Actor(
    representation.rep_state_dim, L.ACTION_DIM, tuple(cfg["actor_hidden_dims"])
  )
  actor.load_state_dict(state["actor" if args.actor == "raw" else "actor_ema"])
  actor.eval()

  starts = eval_starts(args.repo, args.ic_set)
  pool = WorkerPool(
    args.workers,
    SimCfg(args.repo, args.sim_python, str(run / "score_workers")),
    seed=900_000,
  )
  start = time.time()
  try:
    result = run_episodes(
      pool, actor, representation, starts, settle=not args.no_settle
    )
  finally:
    pool.close()
  stages = result["stages"]
  report = {
    "set": args.ic_set,
    "checkpoint": args.checkpoint,
    "round": state["round"],
    "env_steps": state["env_steps"],
    "episodes": len(starts),
    "actions": "deterministic",
    "settle": not args.no_settle,
    "actor": args.actor,
    "stages": {name: float(stages[:, i].mean()) for i, name in enumerate(L.STAGES)},
    "per_episode": [
      {"decisions": int(d), **{name: bool(row[i]) for i, name in enumerate(L.STAGES)}}
      for d, row in zip(result["decisions"], stages, strict=True)
    ],
    "workers": args.workers,
    "node": platform.node(),
    "seconds": time.time() - start,
  }
  suffix = ("_ema" if args.actor == "ema" else "") + (
    "_nosettle" if args.no_settle else ""
  )
  path = run / f"score_{args.ic_set}_{args.checkpoint}{suffix}.json"
  path.write_text(json.dumps(report, indent=2))
  print(
    f"[SCORE] {args.ic_set} {args.checkpoint}: {json.dumps(report['stages'])} -> {path}",
    flush=True,
  )


if __name__ == "__main__":
  main()
