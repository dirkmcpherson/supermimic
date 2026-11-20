"""Compare a trained run's value estimate and world model with real episodes.

  python -m supermimic.restock.scripts.diagnose --run <train dir> --data <dir> \
      --repo <task repository> --sim-python <simulator python>

Rolls the deterministic actor out in the simulator from the given starts, then
asks, on those real trajectories:

- what the critic says of the states against the return that followed;
- what a closed-loop imagined rollout from each real start predicts (does it
  end in ``home``, in a tipped can, or in neither) against what happened;
- whether the world model, fed the real actions from ``k`` decisions before a
  real ending, predicts that ending.

Writes ``diagnose_<set>_<checkpoint>.json`` in the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.manip.wm_env import actor_critic_obs
from supermimic.models import build_mlp
from supermimic.restock import layout as L
from supermimic.restock.data import load_npz
from supermimic.restock.evaluate import act, eval_starts
from supermimic.restock.models import Actor, ManipDynamics
from supermimic.restock.representation import REP_FLAGS, RestockRepresentation
from supermimic.restock.vec_env import SimCfg, WorkerPool

LOOKBACKS = (1, 2, 4, 8, 16)


def record_episodes(pool: WorkerPool, actor, representation, starts: list[dict]):
  """One noise-free episode per start. Each result holds the states visited
  (the final one included), the actions and the stages reached."""
  pending = list(range(len(starts)))
  episodes: list[dict | None] = [None] * len(starts)
  live: dict[int, dict] = {}

  def begin(workers: list[int]) -> None:
    chosen = [(w, pending.pop(0)) for w in workers if pending]
    for w, e in chosen:
      pool._conns[w].send(("reset", (), starts[e]))
    for w, e in chosen:
      live[w] = {"index": e, "states": [pool._receive(pool._conns[w])], "actions": []}

  begin(list(range(pool.num)))
  while live:
    workers = sorted(live)
    states = np.stack([live[w]["states"][-1] for w in workers])
    actions = act(actor, representation, states, "cpu").numpy()
    outs = pool.call("step", [(a,) for a in actions], workers)
    finished = []
    for w, a, out in zip(workers, actions, outs, strict=True):
      live[w]["actions"].append(a)
      live[w]["states"].append(out["next_state"])
      if out["terminated"] or out["truncated"]:
        episode = live[w]
        episodes[episode["index"]] = {
          "states": np.stack(episode["states"]).astype(np.float32),
          "actions": np.stack(episode["actions"]).astype(np.float32),
          "stages": dict(zip(L.STAGES, map(bool, out["stages"]), strict=True)),
        }
        finished.append(w)
    for w in finished:
      del live[w]
    begin(finished)
  return episodes


def outcome(episode: dict) -> str:
  stages = episode["stages"]
  return "home" if stages["home"] else "tipped" if stages["tipped"] else "neither"


class Model:
  def __init__(self, run: Path, data: Path, checkpoint: str):
    constants = load_npz(data / "constants.npz")
    cfg = json.loads((run / "runner_cfg.json").read_text())
    state = torch.load(
      run / f"checkpoint_{checkpoint}.pt", map_location="cpu", weights_only=False
    )
    self.gamma = cfg["policy"]["gamma"]
    self.horizon = cfg["policy"]["horizon"]
    self.env_steps = state["env_steps"]
    self.rep = RestockRepresentation(constants)
    self.rep.load_state_dict(state["representation"])
    rep_dim = self.rep.rep_state_dim
    self.dynamics = ManipDynamics(
      self.rep, L.ACTION_DIM, L.CTX_DIM, tuple(cfg["dynamics_hidden_dims"])
    )
    self.dynamics.load_state_dict(state["dynamics"])
    self.actor = Actor(rep_dim, L.ACTION_DIM, tuple(cfg["actor_hidden_dims"]))
    self.actor.load_state_dict(state["actor"])
    self.critic = build_mlp(
      rep_dim + L.CTX_DIM, tuple(cfg["critic_hidden_dims"]), output_dim=1
    )
    self.critic.load_state_dict(state["critic"])
    for module in (self.rep, self.dynamics, self.actor, self.critic):
      module.eval()
    self.home_reward = float(self.rep.rules.home_reward)

  def ends(self, rep_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The smooth rules' ``home`` and ``tipped`` at a rep state."""
    scene = self.rep.scene(rep_state)
    return self.rep.rules.home(scene, rep_state[..., REP_FLAGS]), self.rep.rules.tipped(
      scene
    )

  def obs(self, rep_state: torch.Tensor):
    ctx = torch.zeros(rep_state.shape[0], L.CTX_DIM)
    return actor_critic_obs(self.dynamics, rep_state, ctx)

  def value(self, states: torch.Tensor) -> torch.Tensor:
    return self.critic(self.obs(self.rep.represent(states))["critic"]).squeeze(-1)

  def step(self, rep_state, aux, action):
    ctx = torch.zeros(rep_state.shape[0], L.CTX_DIM)
    aux, rep_state = self.dynamics(rep_state, action, aux, ctx)
    return rep_state, aux

  def imagine(self, states: torch.Tensor, steps: int) -> dict[str, torch.Tensor]:
    """Closed-loop rollout with the actor: the probability of having ended in
    ``home`` and in a tipped can within ``steps`` decisions."""
    rep_state, aux = self.rep.represent(states), self.rep.extract_aux_state(states)
    survival = torch.ones(len(states))
    p_home, p_tip = torch.zeros(len(states)), torch.zeros(len(states))
    for _ in range(steps):
      rep_state, aux = self.step(rep_state, aux, self.actor(self.obs(rep_state)))
      home, tipped = self.ends(rep_state)
      p_home = p_home + survival * home
      p_tip = p_tip + survival * (1 - home) * tipped
      survival = survival * (1 - home) * (1 - tipped)
    return {"home": p_home, "tipped": p_tip, "alive": survival}

  def replay(self, states: torch.Tensor, actions: torch.Tensor):
    """Open-loop rollout of one real action sequence from its first state:
    the smooth rules at the last predicted state, and the largest ``tipped``
    along the way."""
    rep_state = self.rep.represent(states[:1])
    aux = self.rep.extract_aux_state(states[:1])
    worst_tip = 0.0
    for action in actions:
      rep_state, aux = self.step(rep_state, aux, action[None])
      home, tipped = self.ends(rep_state)
      worst_tip = max(worst_tip, float(tipped))
    return float(home), float(tipped), worst_tip


def _mean(values) -> float | None:
  values = list(values)
  return float(np.mean(values)) if values else None


@torch.no_grad()
def analyse(model: Model, episodes: list[dict], imagine_steps: int) -> dict:
  kinds = ("home", "tipped", "neither")
  by_kind = {k: [e for e in episodes if outcome(e) == k] for k in kinds}
  report: dict = {
    "episodes": len(episodes),
    "outcomes": {k: len(v) for k, v in by_kind.items()},
    "decisions": {k: _mean(len(e["actions"]) for e in v) for k, v in by_kind.items()},
  }

  # 1. The critic against the return that followed.
  value_rows: dict[str, dict[str, list[float]]] = {
    k: {"first": [], "all": [], "return_first": [], "return_all": []} for k in kinds
  }
  before_end: dict[str, dict[int, list[float]]] = {
    k: {n: [] for n in LOOKBACKS} for k in kinds
  }
  for e in episodes:
    kind, n = outcome(e), len(e["actions"])
    values = model.value(torch.as_tensor(e["states"][:-1])).numpy()
    paid = model.home_reward if kind == "home" else 0.0
    returns = paid * model.gamma ** np.arange(n - 1, -1, -1)
    rows = value_rows[kind]
    rows["first"].append(float(values[0]))
    rows["return_first"].append(float(returns[0]))
    rows["all"].extend(values.tolist())
    rows["return_all"].extend(returns.tolist())
    for back in LOOKBACKS:
      if n >= back:
        before_end[kind][back].append(float(values[n - back]))
  report["critic"] = {
    k: {
      "value_at_start": _mean(value_rows[k]["first"]),
      "return_from_start": _mean(value_rows[k]["return_first"]),
      "value_all_states": _mean(value_rows[k]["all"]),
      "return_all_states": _mean(value_rows[k]["return_all"]),
      "value_decisions_before_end": {
        str(back): _mean(before_end[k][back]) for back in LOOKBACKS
      },
    }
    for k in kinds
  }
  every_value = np.concatenate([value_rows[k]["all"] for k in kinds])
  every_return = np.concatenate([value_rows[k]["return_all"] for k in kinds])
  report["critic"]["overall"] = {
    "value": float(every_value.mean()),
    "return": float(every_return.mean()),
    "correlation": float(np.corrcoef(every_value, every_return)[0, 1]),
    "share_above_9": float((every_value > 9.0).mean()),
  }

  # 2. Closed-loop imagination from the real starts.
  starts = torch.as_tensor(np.stack([e["states"][0] for e in episodes]))
  report["imagined_from_start"] = {}
  for steps in (model.horizon, imagine_steps):
    imagined = {k: v.numpy() for k, v in model.imagine(starts, steps).items()}
    report["imagined_from_start"][str(steps)] = {
      kind: {
        name: _mean(
          imagined[name][i] for i, e in enumerate(episodes) if outcome(e) == kind
        )
        for name in ("home", "tipped", "alive")
      }
      for kind in kinds
    } | {"all": {name: float(imagined[name].mean()) for name in imagined}}
  report["real_within"] = {
    str(steps): {
      kind: float(
        np.mean([outcome(e) == kind and len(e["actions"]) <= steps for e in episodes])
      )
      for kind in ("home", "tipped")
    }
    for steps in (model.horizon, imagine_steps)
  }

  # 3. Real actions from k decisions before a real ending.
  report["replayed_endings"] = {}
  for kind in ("home", "tipped"):
    rows = {}
    for back in LOOKBACKS:
      found = []
      for e in by_kind[kind]:
        n = len(e["actions"])
        if n < back:
          continue
        states = torch.as_tensor(e["states"][n - back :])
        home, tipped, worst_tip = model.replay(
          states, torch.as_tensor(e["actions"][n - back :])
        )
        found.append((home, tipped, worst_tip))
      if found:
        home, tipped, worst_tip = map(np.array, zip(*found, strict=True))
        mine = home if kind == "home" else tipped
        rows[str(back)] = {
          "episodes": len(found),
          "predicted_probability": float(mine.mean()),
          "share_above_half": float((mine > 0.5).mean()),
          "tipped_any_step": float((worst_tip > 0.5).mean()),
        }
    report["replayed_endings"][kind] = rows

  # The rule on the true final state (what a perfect model would give).
  report["rule_on_true_final_state"] = {}
  for kind in ("home", "tipped"):
    if by_kind[kind]:
      finals = torch.as_tensor(np.stack([e["states"][-1] for e in by_kind[kind]]))
      home, tipped = model.ends(model.rep.represent(finals))
      report["rule_on_true_final_state"][kind] = float(
        (home if kind == "home" else tipped).mean()
      )
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True)
  parser.add_argument("--data", required=True)
  parser.add_argument("--repo", required=True)
  parser.add_argument("--sim-python", required=True)
  parser.add_argument("--checkpoint", default="final", choices=("final", "latest"))
  parser.add_argument("--set", default="success_uid_universe", dest="ic_set")
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--imagine-steps", type=int, default=64)
  args = parser.parse_args()

  run = Path(args.run)
  torch.set_num_threads(2)
  model = Model(run, Path(args.data), args.checkpoint)
  starts = eval_starts(args.repo, args.ic_set)
  pool = WorkerPool(
    args.workers, SimCfg(args.repo, args.sim_python, None), seed=910_000
  )
  try:
    episodes = record_episodes(pool, model.actor, model.rep, starts)
  finally:
    pool.close()
  report = {
    "set": args.ic_set,
    "checkpoint": args.checkpoint,
    "env_steps": model.env_steps,
    **analyse(model, episodes, args.imagine_steps),
  }
  path = run / f"diagnose_{args.ic_set}_{args.checkpoint}.json"
  path.write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2), flush=True)
  print(f"[DIAGNOSE] wrote {path}", flush=True)


if __name__ == "__main__":
  main()
