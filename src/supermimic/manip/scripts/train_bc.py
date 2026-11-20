"""Baseline: clone the demonstrations, with the policy's own actor and inputs.

  python -m supermimic.manip.scripts.train_bc --data data/square --out logs/square/run/bc

Regresses the recorded actions on the recorded states (mean squared error),
then scores the result like a trained policy: in the training simulator and in
stock robosuite. Uses no environment interaction and no world model. Writes
``bc_metrics.jsonl`` and ``bc_report.json`` into ``--out``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from supermimic.manip import layout as L
from supermimic.manip.collect import split_episodes
from supermimic.manip.data import load_constants, load_episodes, task_of
from supermimic.manip.evaluate import evaluate_policy
from supermimic.manip.models import Actor
from supermimic.manip.representation import ArmObjectRepresentation
from supermimic.manip.runtime import configure_torch, limit_gpu_memory
from supermimic.manip.sim import SimCfg
from supermimic.manip.tasks import TASKS
from supermimic.manip.vec_env import VecEnv


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--steps", type=int, default=40_000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--hidden", type=int, nargs="+", default=[2048, 1024, 512, 256])
  parser.add_argument("--val-fraction", type=float, default=0.1)
  parser.add_argument(
    "--evals", type=int, default=4, help="evaluations during training"
  )
  parser.add_argument("--eval-envs", type=int, default=8)
  parser.add_argument("--eval-episodes", type=int, default=48)
  parser.add_argument("--final-eval-episodes", type=int, default=200)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--gpu-memory-gb", type=float, default=8.0)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  configure_torch(args.seed)
  limit_gpu_memory(args.device, args.gpu_memory_gb)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  device = args.device

  constants = load_constants(args.data)
  task = task_of(constants)
  horizon = TASKS[task].eval_episode_steps
  representation = ArmObjectRepresentation(constants).to(device)
  train, val = split_episodes(
    load_episodes(Path(args.data) / "demos_recorded.npz"), args.val_fraction
  )

  def tensors(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(data["state"], device=device)
    with torch.no_grad():
      return representation.represent(state), torch.as_tensor(
        data["action"], device=device
      )

  obs, target = tensors(train)
  val_obs, val_target = tensors(val)
  print(f"[INFO] {task}: {len(obs)} training and {len(val_obs)} held-out demo steps")

  actor = Actor(representation.rep_state_dim, L.ACTION_DIM, tuple(args.hidden)).to(
    device
  )
  actor.train()
  with torch.no_grad():
    for chunk in obs.split(65536):
      actor.update_normalization(chunk)
  # Input statistics are fixed from here on (they only update in train mode;
  # the MLP has no mode-dependent layers).
  actor.eval()
  optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

  env = VecEnv(args.eval_envs, SimCfg(task=task, horizon=horizon), args.seed + 500_000)
  cpu_representation = copy.deepcopy(representation).cpu()

  def score(episodes: int, eval_env: VecEnv) -> dict[str, float]:
    cpu_actor = copy.deepcopy(actor).cpu()
    return evaluate_policy(
      cpu_actor, cpu_representation, eval_env, episodes, horizon, "cpu"
    )

  def val_loss() -> float:
    with torch.no_grad():
      return float(((actor({"actor": val_obs}) - val_target) ** 2).mean())

  metrics_path = out / "bc_metrics.jsonl"
  metrics_path.write_text("")
  eval_every = max(1, args.steps // max(1, args.evals))
  generator = torch.Generator(device=device).manual_seed(args.seed)
  start = time.time()
  evals = []
  try:
    for step in range(1, args.steps + 1):
      index = torch.randint(
        len(obs), (args.batch_size,), device=device, generator=generator
      )
      loss = ((actor({"actor": obs[index]}) - target[index]) ** 2).mean()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      scheduler.step()
      if step % eval_every == 0 or step == args.steps:
        row = {
          "step": step,
          "seconds": time.time() - start,
          "val_loss": val_loss(),
          **score(args.eval_episodes, env),
        }
        evals.append(row)
        with metrics_path.open("a") as f:
          f.write(json.dumps(row) + "\n")
        print(
          f"[BC] step {step}/{args.steps}  val {row['val_loss']:.4f}  "
          f"success {row['success_rate']:.2f}",
          flush=True,
        )

    report = {
      "task": task,
      "demo_steps": len(obs),
      "steps": args.steps,
      "train_seconds": time.time() - start,
      "val_loss": val_loss(),
      "best_eval_success": max(e["success_rate"] for e in evals),
      "final": score(args.final_eval_episodes, env),
    }
    stock_env = VecEnv(
      args.eval_envs,
      SimCfg(
        task=task, horizon=horizon, proportional_gripper=False, nominal_nullspace=False
      ),
      args.seed + 900_000,
    )
    report["final_stock_env"] = score(args.final_eval_episodes, stock_env)
    stock_env.close()
    torch.save(
      {"actor": actor.state_dict(), "hidden": args.hidden}, out / "bc_actor.pt"
    )
    (out / "bc_report.json").write_text(json.dumps(report, indent=2))
    print(f"[BC] report: {json.dumps(report)}", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
