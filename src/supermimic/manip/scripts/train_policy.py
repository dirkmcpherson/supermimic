"""Stage 2: train the policy through the world model, online.

  python -m supermimic.manip.scripts.train_policy --data data/square \
    --wm logs/square/run/world_model.pt --out logs/square/run/main --hours 6

The task is the one ``--data`` was built for. ``--wm scratch`` starts from an
untrained world model (with ``--wm-mix 1 0 0 --start-mix 1 0 0
--bank-fraction 0`` that is a run without any offline data).

Writes ``policy_metrics.jsonl``, ``policy_eval.jsonl``, checkpoints and
``policy_report.json`` into ``--out``. ``--resume`` continues from
``checkpoint_latest.pt`` there (weights and counters; optimizers and the
online replay buffer start fresh).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.manip.collect import split_episodes
from supermimic.manip.data import (
  buffer_from_episodes,
  load_constants,
  load_episodes,
  task_of,
)
from supermimic.manip.runner import Runner, RunnerCfg
from supermimic.manip.runtime import configure_torch, limit_gpu_memory
from supermimic.manip.scripts.train_wm import build_world_model, load_world_model
from supermimic.manip.sim import SimCfg
from supermimic.manip.vec_env import StartBank, VecEnv

SCRATCH = "scratch"


def final_evaluations(runner: Runner, out: Path, device: str) -> dict:
  """Score the last policy (training simulator and stock robosuite) and the
  best checkpoint seen during training."""
  cfg = runner.cfg
  report = {"final": runner.evaluate(cfg.final_eval_episodes)}
  stock_env = VecEnv(
    cfg.eval_envs,
    SimCfg(
      task=cfg.task,
      horizon=cfg.eval_episode_steps,
      proportional_gripper=False,
      nominal_nullspace=False,
    ),
    cfg.seed + 900_000,
  )
  report["final_stock_env"] = runner.evaluate(cfg.final_eval_episodes, stock_env)
  stock_env.close()
  best_path = out / "checkpoint_best.pt"
  if best_path.exists():
    best = torch.load(best_path, map_location=device)
    report["best_checkpoint_round"] = best["round"]
    if best["round"] != runner.round:
      # Last use of the runner: the final weights are in checkpoint_latest.pt.
      runner.load_state_dict(best)
      report["best"] = runner.evaluate(cfg.final_eval_episodes)
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True)
  parser.add_argument(
    "--wm", required=True, help=f"world_model.pt from train_wm, or '{SCRATCH}'"
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--hours", type=float, default=6.0)
  parser.add_argument("--max-rounds", type=int, default=1_000_000)
  parser.add_argument("--num-envs", type=int, default=24)
  parser.add_argument("--eval-envs", type=int, default=8)
  parser.add_argument("--episode-steps", type=int, default=None)
  parser.add_argument("--policy-batch-size", type=int, default=2048)
  parser.add_argument("--policy-horizon", type=int, default=None)
  parser.add_argument("--gamma", type=float, default=None)
  parser.add_argument("--wm-batch-size", type=int, default=2048)
  parser.add_argument("--steps-per-round", type=int, default=107)
  parser.add_argument("--eval-interval", type=int, default=50)
  parser.add_argument("--eval-episodes", type=int, default=48)
  parser.add_argument("--final-eval-episodes", type=int, default=200)
  parser.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048])
  parser.add_argument("--actor-hidden", type=int, nargs="+", default=None)
  parser.add_argument(
    "--bank-fraction",
    type=float,
    default=None,
    help="share of real training episodes started from a demo state",
  )
  parser.add_argument(
    "--start-mix",
    type=float,
    nargs=3,
    default=None,
    metavar=("ONLINE", "DEMO", "BRANCH"),
    help="shares of imagination start states from the three buffers",
  )
  parser.add_argument(
    "--wm-mix",
    type=float,
    nargs=3,
    default=None,
    metavar=("ONLINE", "DEMO", "BRANCH"),
    help="shares of each world-model batch from the three buffers",
  )
  parser.add_argument("--val-fraction", type=float, default=0.1)
  parser.add_argument("--no-compile", action="store_true")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--gpu-memory-gb", type=float, default=8.0)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  configure_torch(args.seed)
  # The simulator workers need the cores; CPU torch only runs the small actor.
  torch.set_num_threads(2)
  limit_gpu_memory(args.device, args.gpu_memory_gb)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  device = args.device
  data_dir = Path(args.data)

  constants = load_constants(data_dir)
  scratch = args.wm == SCRATCH
  if scratch:
    hidden = tuple(args.hidden)
  else:
    hidden = tuple(torch.load(args.wm, map_location="cpu")["hidden"])
  representation, dynamics = build_world_model(constants, hidden, device)
  if not scratch:
    load_world_model(Path(args.wm), representation, dynamics, device)

  cfg = RunnerCfg(
    task=task_of(constants),
    num_envs=args.num_envs,
    eval_envs=args.eval_envs,
    episode_steps=args.episode_steps,
    steps_per_round=args.steps_per_round,
    dynamics_hidden_dims=hidden,
    eval_interval=args.eval_interval,
    eval_episodes=args.eval_episodes,
    final_eval_episodes=args.final_eval_episodes,
    max_rounds=args.max_rounds,
    max_hours=args.hours,
    compile=not args.no_compile,
    seed=args.seed,
  )
  cfg.policy.batch_size = args.policy_batch_size
  cfg.wm.batch_size = args.wm_batch_size
  if args.policy_horizon is not None:
    cfg.policy.horizon = args.policy_horizon
  if args.gamma is not None:
    cfg.policy.gamma = args.gamma
  if args.actor_hidden is not None:
    cfg.actor_hidden_dims = tuple(args.actor_hidden)
  if args.bank_fraction is not None:
    cfg.bank_fraction = args.bank_fraction
  if args.start_mix is not None:
    cfg.start_mix = tuple(args.start_mix)
  if args.wm_mix is not None:
    cfg.wm_mix = tuple(args.wm_mix)

  # The same training episodes stage 1 fit the world model on.
  demos, _ = split_episodes(load_episodes(data_dir / "demos.npz"), args.val_fraction)
  branches, _ = split_episodes(
    load_episodes(data_dir / "branches.npz"), args.val_fraction
  )
  with np.load(data_dir / "start_bank.npz") as f:
    bank = StartBank(states=f["states"], variants=f["variants"])

  runner = Runner(
    cfg,
    representation,
    dynamics,
    buffer_from_episodes(demos, 16, device),
    buffer_from_episodes(branches, 64, device),
    bank,
    out,
    device,
  )
  try:
    if args.resume and (out / "checkpoint_latest.pt").exists():
      runner.load_state_dict(
        torch.load(out / "checkpoint_latest.pt", map_location=device)
      )
      print(f"[INFO] resumed at round {runner.round} ({runner.elapsed / 3600:.2f} h)")
    elif scratch:
      scales = runner.bootstrap_statistics(10 * cfg.steps_per_round)
      print(f"[INFO] untrained world model; prediction scales from rollouts: {scales}")
    else:
      offline = torch.cat(
        [torch.as_tensor(demos["state"]), torch.as_tensor(branches["state"])]
      ).to(device)
      runner.init_actor_normalizer(offline)

    report = runner.run()
    print(f"[POLICY] stopped: {json.dumps(report)}", flush=True)
    report.update(final_evaluations(runner, out, device))
    (out / "policy_report.json").write_text(json.dumps(report, indent=2))
    print(f"[POLICY] report: {json.dumps(report)}", flush=True)
  finally:
    runner.close()


if __name__ == "__main__":
  main()
