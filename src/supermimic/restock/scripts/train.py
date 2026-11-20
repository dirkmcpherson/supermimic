"""Train SuperMimic on the restock task: fit the world model to the
re-executed demonstrations, then run the online loop.

  python -m supermimic.restock.scripts.train --data <prepare_data dir> \
      --out <dir> --repo <task repository> --sim-python <simulator python>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from supermimic.manip.runtime import configure_torch, gpu_memory_gb
from supermimic.restock import layout as L
from supermimic.restock.data import (
  MixtureSampler,
  buffer_from_episodes,
  load_npz,
  with_final_states,
)
from supermimic.restock.evaluate import eval_starts
from supermimic.restock.models import ManipDynamics, prediction_floors
from supermimic.restock.outcomes import demo_outcomes
from supermimic.restock.policy import CRITIC_TARGETS
from supermimic.restock.representation import RestockRepresentation
from supermimic.restock.runner import Runner, RunnerCfg
from supermimic.restock.vec_env import SimCfg, WorkerPool


def pretrain_world_model(
  runner: Runner, demo_sampler: MixtureSampler, updates: int, out: Path
) -> None:
  """World-model updates on demonstration windows only, before any online data."""
  cfg = runner.cfg.wm
  start = time.time()
  with (out / "wm_pretrain.jsonl").open("w") as f:
    for update in range(1, updates + 1):
      data = (
        demo_sampler.sample(cfg.horizon, cfg.batch_size) for _ in range(cfg.num_batches)
      )
      logs = runner.world_model_learner.update(data)
      if update % 50 == 0 or update == updates:
        row = {
          "update": update,
          "seconds": time.time() - start,
          **{k: float(v) for k, v in logs.items()},
        }
        f.write(json.dumps(row) + "\n")
        f.flush()
      if update % 500 == 0 or update == updates:
        print(
          f"[WM] update {update}/{updates}  loss {float(logs['Loss/state']):.3f}  {time.time() - start:.0f} s",
          flush=True,
        )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--repo", required=True)
  parser.add_argument("--sim-python", required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--decisions", type=int, default=250_000)
  parser.add_argument(
    "--wm-updates",
    type=int,
    default=1000,
    help="pretraining rounds on the demos (0: none)",
  )
  parser.add_argument("--steps-per-round", type=int, default=8)
  parser.add_argument("--wm-batches", type=int, default=16)
  parser.add_argument("--policy-batches", type=int, default=16)
  parser.add_argument("--policy-horizon", type=int, default=16)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--explore-scale", type=float, default=0.3)
  parser.add_argument(
    "--demo-mix",
    type=float,
    default=0.5,
    help="demo share of world-model batches and of imagination starts",
  )
  parser.add_argument(
    "--demo-starts",
    type=float,
    default=0.0,
    help="share of training episodes started part-way along a tape (prefix replayed, counted)",
  )
  parser.add_argument(
    "--critic",
    default="imagined",
    choices=CRITIC_TARGETS,
    help="what the critic is fitted to: imagined returns, real episodes' returns, or both",
  )
  parser.add_argument(
    "--actor-ema",
    type=float,
    default=0.0,
    help="per-round decay of an averaged copy of the actor (0: none)",
  )
  parser.add_argument(
    "--actor-lam",
    type=float,
    default=1.0,
    help="below 1 the actor climbs the lambda-return, which weighs the critic at every imagined step",
  )
  parser.add_argument("--policy-lr", type=float, default=None)
  parser.add_argument(
    "--wm-guard",
    type=float,
    default=0.0,
    help="skip a round's policy update when the world-model loss exceeds this multiple of its recent level (0: never)",
  )
  parser.add_argument(
    "--trace", action="store_true", help="log every round to round_trace.jsonl"
  )
  parser.add_argument("--eval-interval", type=int, default=100)
  parser.add_argument("--eval-set", default="rnd")
  parser.add_argument("--hours", type=float, default=12.0)
  parser.add_argument("--no-compile", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  configure_torch(args.seed)
  device = args.device

  constants = load_npz(Path(args.data) / "constants.npz")
  demos = load_npz(Path(args.data) / "demos.npz")
  cfg = RunnerCfg(
    steps_per_round=args.steps_per_round,
    explore_scale=args.explore_scale,
    demo_start_fraction=args.demo_starts,
    wm_mix=(1 - args.demo_mix, args.demo_mix),
    start_mix=(1 - args.demo_mix, args.demo_mix),
    eval_interval=args.eval_interval,
    max_decisions=args.decisions,
    max_hours=args.hours,
    compile=not args.no_compile,
    seed=args.seed,
  )
  cfg.wm.num_batches = args.wm_batches
  cfg.policy.num_batches = args.policy_batches
  cfg.policy.horizon = args.policy_horizon
  cfg.policy.gamma = args.gamma
  cfg.policy.critic_targets = args.critic
  cfg.policy.actor_lam = args.actor_lam
  cfg.actor_ema = args.actor_ema
  cfg.wm_guard = args.wm_guard
  cfg.trace = args.trace
  if args.policy_lr is not None:
    cfg.policy.lr = args.policy_lr

  representation = RestockRepresentation(constants).to(device)
  state = torch.as_tensor(demos["state"], dtype=torch.float32, device=device)
  next_state = torch.as_tensor(demos["next_state"], dtype=torch.float32, device=device)
  action = torch.as_tensor(demos["action"], dtype=torch.float32, device=device)
  scales = representation.fit_prediction_scale(
    state, next_state, action, prediction_floors(cfg.wm)
  )
  dynamics = ManipDynamics(
    representation, L.ACTION_DIM, L.CTX_DIM, cfg.dynamics_hidden_dims
  ).to(device)
  demo_buffer = buffer_from_episodes(with_final_states(demos), 16, device)

  start = time.time()
  pool = WorkerPool(
    args.num_envs,
    SimCfg(args.repo, args.sim_python, str(out / "workers")),
    args.seed * 1000,
  )
  print(
    f"[SIM] {args.num_envs} simulators up in {time.time() - start:.0f} s", flush=True
  )
  try:
    live = pool.constants()
    for key in ("task_code", "sim_variant", "ladder", "tip_guard"):
      if str(live[key]) != str(constants[key]):
        raise SystemExit(
          f"{key}: the data was built with {constants[key]}, the simulator has {live[key]}"
        )
    runner = Runner(
      cfg,
      representation,
      dynamics,
      demo_buffer,
      pool,
      eval_starts(args.repo, args.eval_set),
      out,
      device,
      demo_outcomes=demo_outcomes(
        demos, cfg.policy.gamma, float(representation.rules.home_reward)
      ),
    )
    if args.demo_starts > 0:
      meta = json.loads((Path(args.data) / "meta.json").read_text())
      tapes = meta["tapes_detail"]
      runner.env.set_demo_starts(
        [str(Path(meta["tape_dir"]) / t["name"]) for t in tapes],
        [t["decisions"] for t in tapes],
        args.demo_starts,
      )
    runner.init_actor_normalizer(state)
    runner.world_model_learner.update_normalization(state, action)
    (out / "setup.json").write_text(
      json.dumps(
        {
          "args": vars(args),
          "prediction_scales": scales,
          "demo_transitions": len(demos["state"]),
        },
        indent=2,
      )
    )
    if args.wm_updates > 0:
      pretrain_world_model(
        runner, MixtureSampler([(demo_buffer, 1.0)]), args.wm_updates, out
      )
    summary = runner.run()
    summary["gpu_gb"] = gpu_memory_gb(device)
    summary["final_training_stages"] = runner._episode_stats()
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] {json.dumps(summary)}", flush=True)
  finally:
    pool.close()


if __name__ == "__main__":
  main()
