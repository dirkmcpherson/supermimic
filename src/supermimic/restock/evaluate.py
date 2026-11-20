"""Roll the deterministic actor out in the real environment from given starts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.restock import layout as L
from supermimic.restock.representation import RestockRepresentation
from supermimic.restock.vec_env import WorkerPool


@torch.no_grad()
def act(
  actor: torch.nn.Module,
  representation: RestockRepresentation,
  states: np.ndarray,
  device: str,
) -> torch.Tensor:
  """Noise-free actions for simulator states."""
  state = torch.as_tensor(states, dtype=torch.float32, device=device)
  rep = representation.represent(state)
  return actor(TensorDict({"actor": rep}, batch_size=[rep.shape[0]], device=device))


def eval_starts(repo: str | Path, ic_set: str) -> list[dict]:
  """The task's evaluation starts (``baselines/eval_ics.json``): ``rnd``
  is the 30 random can positions, ``hold`` 15 held-out training starts. Each
  entry is the keyword arguments of the worker's ``reset``."""
  table = json.loads((Path(repo) / "baselines" / "eval_ics.json").read_text())
  if ic_set == "rnd":
    return [
      {
        "ic": {
          "can_pos": tuple(e["can_pos"]),
          "goal_pos": tuple(e["goal_pos"]),
          "uid": None,
        }
      }
      for e in table["rnd"]
    ]
  return [{"uid": int(u)} for u in table[ic_set]]


def run_episodes(
  pool: WorkerPool,
  actor: torch.nn.Module,
  representation: RestockRepresentation,
  starts: list[dict],
  device: str = "cpu",
  settle: bool = False,
) -> dict[str, np.ndarray]:
  """One episode per start, in order, spread over the pool's workers.

  Returns ``stages`` ``(episodes, len(L.STAGES))`` and ``decisions``
  ``(episodes,)``. With ``settle`` each episode is followed by the settle the
  task's evaluator runs (it matters only to later episodes of that worker).
  """
  was_training = actor.training
  actor.eval()
  stages = np.zeros((len(starts), len(L.STAGES)), dtype=bool)
  decisions = np.zeros(len(starts), dtype=np.int64)
  pending = list(range(len(starts)))
  episode: dict[int, int] = {}
  states: dict[int, np.ndarray] = {}

  def begin(workers: list[int]) -> None:
    chosen = [(w, pending.pop(0)) for w in workers if pending]
    for w, e in chosen:
      pool._conns[w].send(("reset", (), starts[e]))
    for w, e in chosen:
      states[w] = pool._receive(pool._conns[w])
      episode[w] = e

  begin(list(range(pool.num)))
  while episode:
    workers = sorted(episode)
    actions = act(actor, representation, np.stack([states[w] for w in workers]), device)
    outs = pool.call("step", [(a,) for a in actions.cpu().numpy()], workers)
    finished = []
    for w, out in zip(workers, outs, strict=True):
      e = episode[w]
      decisions[e] += 1
      states[w] = out["next_state"]
      if out["terminated"] or out["truncated"]:
        stages[e] = out["stages"]
        finished.append(w)
    if finished:
      if settle:
        pool.call("finish_episode", [()] * len(finished), finished)
      for w in finished:
        del episode[w], states[w]
      begin(finished)
  actor.train(was_training)
  return {"stages": stages, "decisions": decisions}


def summarize(result: dict[str, np.ndarray]) -> dict[str, float]:
  rates = result["stages"].mean(axis=0)
  return {
    "episodes": len(result["stages"]),
    **{name: float(rate) for name, rate in zip(L.STAGES, rates, strict=True)},
    "decisions": float(result["decisions"].mean()),
  }
