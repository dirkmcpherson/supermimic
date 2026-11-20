"""Stage 1: fit the world model alone on the offline dataset and score it.

  python -m supermimic.manip.scripts.train_wm --data data/lift --out logs/lift/run

Writes ``world_model.pt`` (the update with the lowest validation loss),
``wm_metrics.jsonl`` and ``wm_eval.json`` into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.manip import layout as L
from supermimic.manip.collect import KIND_NAMES, split_episodes
from supermimic.manip.data import (
  MixtureSampler,
  buffer_from_episodes,
  load_constants,
  load_episodes,
)
from supermimic.manip.models import ManipDynamics, ManipWorldModelLearnerCfg
from supermimic.manip.representation import ArmObjectRepresentation
from supermimic.manip.runtime import configure_torch, gpu_memory_gb, limit_gpu_memory
from supermimic.manip.wm_eval import open_loop_errors


def build_world_model(
  constants: dict, hidden_dims: tuple[int, ...], device: str
) -> tuple[ArmObjectRepresentation, ManipDynamics]:
  representation = ArmObjectRepresentation(constants).to(device)
  dynamics = ManipDynamics(
    representation, L.ACTION_DIM, L.CTX_DIM, hidden_dims=hidden_dims
  ).to(device)
  return representation, dynamics


def save_world_model(path: Path, representation, dynamics, extra: dict) -> None:
  torch.save(
    {
      "dynamics": dynamics.state_dict(),
      "representation": representation.state_dict(),
      **extra,
    },
    path,
  )


def load_world_model(path: Path, representation, dynamics, device: str) -> dict:
  bundle = torch.load(path, map_location=device)
  # In place: captured updates and compiled functions hold these tensors.
  dynamics.load_state_dict(bundle["dynamics"])
  representation.load_state_dict(bundle["representation"])
  return bundle


def fit_statistics(
  learner: WorldModelLearner, representation, data: dict, device: str
) -> dict[str, float]:
  """Prediction scale and input normalizers from the training transitions."""
  state = torch.as_tensor(data["state"], device=device)
  action = torch.as_tensor(data["action"], device=device)
  # Consecutive rows are one transition unless the first ends an episode.
  same_episode = ~torch.as_tensor(data["truncated"][:-1], device=device)
  scales = representation.fit_prediction_scale(
    state[:-1][same_episode], state[1:][same_episode]
  )
  for chunk in torch.arange(len(state), device=device).split(65536):
    learner.update_normalization(state[chunk], action[chunk])
  return scales


@torch.no_grad()
def validation_loss(learner: WorldModelLearner, windows) -> float:
  """The training loss on held-out windows."""
  rep = learner.representation
  full = windows["state"]
  all_rep, all_aux = rep.represent(full), rep.extract_aux_state(full)
  pred_rep, pred_aux = learner._rollout(
    all_rep[:, 0], windows["action"], all_aux[:, 0], windows["sim_ctx"][:, 0]
  )
  loss, _ = learner._compute_losses(pred_rep, pred_aux, all_rep[:, 1:], all_aux[:, 1:])
  return float(loss)


def evaluate(dynamics, representation, val: dict, device: str, steps, batch) -> dict:
  """Open-loop errors on held-out demos and on each kind of held-out branch."""
  report = {}
  horizon = max(steps) + 1
  for name, episodes in val.items():
    buffer = buffer_from_episodes(episodes, num_rows=4, device=device)
    try:
      windows = buffer.sample(horizon, batch)
    except RuntimeError:
      report[name] = {"windows": 0}
      continue
    report[name] = open_loop_errors(dynamics, representation, windows, steps)
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--updates", type=int, default=2000)
  parser.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048])
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--num-batches", type=int, default=10)
  parser.add_argument("--val-interval", type=int, default=50)
  parser.add_argument("--val-fraction", type=float, default=0.1)
  parser.add_argument("--demo-share", type=float, default=0.25)
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
  demos_train, demos_val = split_episodes(
    load_episodes(Path(args.data) / "demos.npz"), args.val_fraction
  )
  branch_train, branch_val = split_episodes(
    load_episodes(Path(args.data) / "branches.npz"), args.val_fraction
  )
  print(
    f"[INFO] train: {len(demos_train['state'])} demo + {len(branch_train['state'])} "
    f"branch transitions; val: {len(demos_val['state'])} + {len(branch_val['state'])}"
  )

  representation, dynamics = build_world_model(constants, tuple(args.hidden), device)
  cfg = ManipWorldModelLearnerCfg(
    batch_size=args.batch_size, num_batches=args.num_batches
  )
  learner = WorldModelLearner(dynamics, representation, cfg, device=device)

  both = {
    k: torch.cat(
      [torch.as_tensor(demos_train[k]), torch.as_tensor(branch_train[k])]
    ).numpy()
    for k in ("state", "action", "truncated")
  }
  scales = fit_statistics(learner, representation, both, device)
  print(f"[INFO] prediction scales: { {k: round(v, 5) for k, v in scales.items()} }")

  sampler = MixtureSampler(
    [
      (buffer_from_episodes(demos_train, 16, device), args.demo_share),
      (buffer_from_episodes(branch_train, 64, device), 1.0 - args.demo_share),
    ]
  )
  val_sampler = MixtureSampler(
    [
      (buffer_from_episodes(demos_val, 4, device), args.demo_share),
      (buffer_from_episodes(branch_val, 16, device), 1.0 - args.demo_share),
    ]
  )
  torch.manual_seed(args.seed + 1)
  val_windows = val_sampler.sample(cfg.horizon, 8192)

  best, best_update = float("inf"), -1
  metrics_path = out / "wm_metrics.jsonl"
  metrics_path.write_text("")
  start = time.time()
  for update in range(1, args.updates + 1):
    data = (sampler.sample(cfg.horizon, cfg.batch_size) for _ in range(cfg.num_batches))
    logs = learner.update(data)
    if update % args.val_interval == 0 or update == args.updates:
      dynamics.eval()
      val = validation_loss(learner, val_windows)
      row = {
        "update": update,
        "seconds": time.time() - start,
        "val_loss": val,
        "gpu_gb": gpu_memory_gb(device),
        **{k: float(v) for k, v in logs.items()},
      }
      with metrics_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
      if val < best:
        best, best_update = val, update
        save_world_model(
          out / "world_model.pt",
          representation,
          dynamics,
          {"update": update, "val_loss": val, "hidden": list(args.hidden)},
        )
      print(
        f"[WM] update {update}/{args.updates}  train {row['Loss/state']:.4f}  "
        f"val {val:.4f}  best {best:.4f}@{best_update}  {row['seconds']:.0f}s"
      )
  train_seconds = time.time() - start

  load_world_model(out / "world_model.pt", representation, dynamics, device)
  dynamics.eval()
  kinds = branch_val["kind"]
  val_sets = {"demos": demos_val, "branches": branch_val}
  # Per-kind branch sets, keeping each episode whole.
  ends = torch.as_tensor(branch_val["truncated"]).nonzero().squeeze(-1).tolist()
  for kind, name in enumerate(KIND_NAMES):
    keep, lo = [], 0
    for end in ends:
      if kinds[end] == kind:
        keep += range(lo, end + 1)
      lo = end + 1
    if keep:
      val_sets[f"branches/{name}"] = {
        k: v[keep] for k, v in branch_val.items() if k != "episode_success"
      }
  report = {
    "best_update": best_update,
    "best_val_loss": best,
    "train_seconds": train_seconds,
    "gpu_gb": gpu_memory_gb(device),
    "prediction_scales": scales,
    "open_loop": evaluate(
      dynamics, representation, val_sets, device, (1, 4, 8, 16), 4096
    ),
    "open_loop_long": evaluate(
      dynamics, representation, {"demos": demos_val}, device, (16, 32), 4096
    ),
  }
  (out / "wm_eval.json").write_text(json.dumps(report, indent=2))
  k16 = report["open_loop"]["branches"].get("k16", {})
  print(
    f"[WM] done in {train_seconds:.0f}s; held-out branches at 16 steps: {json.dumps(k16)}"
  )


if __name__ == "__main__":
  main()
