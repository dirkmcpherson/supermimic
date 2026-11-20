"""Offline data for the world model: demo replays and exploration branches.

NumPy only. Every transition is produced by this simulator: demonstrations are
replayed (their recorded actions, from their recorded first state), and
branches restore a random demo frame and act with noise from there.

Episodes are stored back to back as flat arrays; ``truncated`` marks the last
transition of each episode. A transition holds the state an action was applied
in, like the replay buffer's rows; the state the last action led to is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from supermimic.manip import layout as L
from supermimic.manip.demos import Demo
from supermimic.manip.sim import ArmObjectSim, SimCfg, make_sim
from supermimic.manip.vec_env import single_threaded_children, spawn_context

KIND_DEMO = 0
KIND_DEMO_NOISE = 1
KIND_RANDOM = 2
KIND_TASK_RANDOM = 3
KIND_NAMES = ("demo", "demo_noise", "random", "task_random")


@dataclass(frozen=True)
class BranchCfg:
  length: tuple[int, int] = (16, 41)
  """Branch length range in control steps, ``[lo, hi)``."""
  demo_noise_scale: tuple[float, float] = (0.05, 0.5)
  """Log-uniform range of the noise added to the demo's own actions."""
  random_scale: tuple[float, float] = (0.2, 1.0)
  """Log-uniform range of the noise that drives a random branch."""
  correlation: float = 0.8
  """Per-step correlation of the action noise."""
  gripper_switch_prob: float = 0.1
  """Per-step chance that a random branch redraws its gripper command."""
  gripper_override_prob: float = 0.3
  """Chance that a demo-noise branch replaces the demo's gripper command for a
  stretch (a dropped or premature grasp)."""
  kind_probs: tuple[float, float, float] = (0.5, 0.4, 0.1)
  """Shares of demo-noise, random and task-reset-random branches."""
  task_random_length: int = 40


class _Recorder:
  def __init__(self) -> None:
    self.rows: dict[str, list] = {
      k: [] for k in ("state", "action", "prev_action", "sim_ctx", "truncated", "kind")
    }
    self.success: list[bool] = []

  def add(self, state, action, prev_action, ctx, last: bool, kind: int) -> None:
    for key, value in zip(
      self.rows, (state, action, prev_action, ctx, last, kind), strict=True
    ):
      self.rows[key].append(value)

  def __len__(self) -> int:
    return len(self.rows["state"])

  def arrays(self) -> dict[str, np.ndarray]:
    return {
      "state": np.array(self.rows["state"], dtype=np.float32),
      "action": np.array(self.rows["action"], dtype=np.float32),
      "prev_action": np.array(self.rows["prev_action"], dtype=np.float32),
      "sim_ctx": np.array(self.rows["sim_ctx"], dtype=np.float32),
      "truncated": np.array(self.rows["truncated"], dtype=bool),
      "kind": np.array(self.rows["kind"], dtype=np.int8),
      "episode_success": np.array(self.success, dtype=bool),
    }


def _run_episode(
  sim: ArmObjectSim,
  rec: _Recorder,
  state: np.ndarray,
  actions,
  prev_action: np.ndarray,
  kind: int,
) -> None:
  """Play ``actions`` (an iterable) from ``state`` and record the transitions."""
  ctx = sim.context()
  actions = list(actions)
  succeeded = False
  for k, action in enumerate(actions):
    action = np.clip(action, -1.0, 1.0)
    rec.add(state, action, prev_action, ctx, k == len(actions) - 1, kind)
    state, success, _ = sim.step(action)
    succeeded |= success
    prev_action = action
  rec.success.append(succeeded)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
  return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _noise_sequence(
  rng: np.random.Generator, steps: int, scale: float, correlation: float
) -> np.ndarray:
  """Stationary correlated Gaussian noise ``(steps, 7)`` with std ``scale``."""
  out = np.empty((steps, L.ACTION_DIM))
  value = rng.standard_normal(L.ACTION_DIM)
  innovation = np.sqrt(1 - correlation**2)
  for t in range(steps):
    out[t] = value
    value = correlation * value + innovation * rng.standard_normal(L.ACTION_DIM)
  return scale * out


def _gripper_holds(rng: np.random.Generator, steps: int, switch_prob: float):
  """Piecewise-constant gripper commands: each hold is +-1 or a value in between."""
  out = np.empty(steps)
  value = 0.0
  for t in range(steps):
    if t == 0 or rng.random() < switch_prob:
      value = rng.choice([-1.0, 1.0]) if rng.random() < 0.5 else rng.uniform(-1, 1)
    out[t] = value
  return out


def _branch_actions(
  rng: np.random.Generator, cfg: BranchCfg, kind: int, demo: Demo | None, frame: int
) -> np.ndarray:
  if kind == KIND_TASK_RANDOM:
    steps = cfg.task_random_length
  else:
    steps = int(rng.integers(*cfg.length))
  if kind == KIND_DEMO_NOISE:
    assert demo is not None
    base = np.zeros((steps, L.ACTION_DIM))
    rest = demo.actions[frame : frame + steps]
    base[: len(rest)] = rest
    # Past the demo's end, hold position and keep its last gripper command.
    base[len(rest) :, -1] = demo.actions[-1, -1]
    scale = _log_uniform(rng, *cfg.demo_noise_scale)
    actions = base + _noise_sequence(rng, steps, scale, cfg.correlation)
    if rng.random() < cfg.gripper_override_prob:
      start = int(rng.integers(steps))
      stop = start + int(rng.integers(1, steps + 1))
      actions[start:stop, -1] = _gripper_holds(rng, steps, cfg.gripper_switch_prob)[
        start:stop
      ]
    return actions
  scale = _log_uniform(rng, *cfg.random_scale)
  actions = _noise_sequence(rng, steps, scale, cfg.correlation)
  actions[:, -1] = _gripper_holds(rng, steps, cfg.gripper_switch_prob)
  return actions


def _branch_worker(args) -> dict[str, np.ndarray]:
  seed, num_steps, demos, sim_cfg, cfg = args
  sim = make_sim(sim_cfg, seed)
  rng = np.random.default_rng(seed + 104729)
  frames = np.array([len(d.actions) for d in demos], dtype=np.float64)
  demo_probs = frames / frames.sum()
  rec = _Recorder()
  while len(rec) < num_steps:
    kind = int(
      rng.choice([KIND_DEMO_NOISE, KIND_RANDOM, KIND_TASK_RANDOM], p=cfg.kind_probs)
    )
    if kind == KIND_TASK_RANDOM:
      state = sim.reset()
      demo, frame = None, 0
      prev_action = np.zeros(L.ACTION_DIM)
    else:
      demo = demos[int(rng.choice(len(demos), p=demo_probs))]
      frame = int(rng.integers(len(demo.actions)))
      state = sim.restore(demo.states[frame], demo.variant)
      prev_action = demo.actions[frame - 1] if frame > 0 else np.zeros(L.ACTION_DIM)
    actions = _branch_actions(rng, cfg, kind, demo, frame)
    _run_episode(sim, rec, state, actions, prev_action, kind)
  return rec.arrays()


def _replay_worker(args) -> tuple[dict[str, np.ndarray], np.ndarray]:
  seed, demos, sim_cfg = args
  sim = make_sim(sim_cfg, seed)
  rec = _Recorder()
  obj_error = []
  for demo in demos:
    state = sim.restore(demo.states[0], demo.variant)
    start = len(rec)
    _run_episode(sim, rec, state, demo.actions, np.zeros(L.ACTION_DIM), KIND_DEMO)
    replayed = np.array(rec.rows["state"][start:])
    obj_error.append(np.abs(replayed[:, L.OBJ_POS] - demo.states[:, L.OBJ_POS]).max())
  return rec.arrays(), np.array(obj_error)


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
  return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def replay_demos(
  demos: list[Demo], sim_cfg: SimCfg, num_workers: int, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
  """Replay every demo in this simulator, in order. Returns the transitions and
  each demo's largest object-position deviation from its recording (metres)."""
  chunks = [c for c in np.array_split(np.arange(len(demos)), num_workers) if len(c)]
  jobs = [(seed + i, [demos[j] for j in c], sim_cfg) for i, c in enumerate(chunks)]
  with single_threaded_children(), spawn_context().Pool(len(jobs)) as pool:
    out = pool.map(_replay_worker, jobs)
  return _concat([o[0] for o in out]), np.concatenate([o[1] for o in out])


def collect_branches(
  demos: list[Demo],
  num_steps: int,
  sim_cfg: SimCfg,
  cfg: BranchCfg,
  num_workers: int,
  seed: int,
) -> dict[str, np.ndarray]:
  per_worker = -(-num_steps // num_workers)
  jobs = [(seed + 31 * i, per_worker, demos, sim_cfg, cfg) for i in range(num_workers)]
  with single_threaded_children(), spawn_context().Pool(num_workers) as pool:
    return _concat(pool.map(_branch_worker, jobs))


def start_bank(demos: list[Demo]) -> tuple[np.ndarray, np.ndarray]:
  """Every recorded demo frame as a start state, with its model variant."""
  states = np.concatenate([d.states for d in demos])
  variants = np.concatenate([np.tile(d.variant, (len(d.states), 1)) for d in demos])
  return states, variants


def recorded_episodes(demos: list[Demo]) -> dict[str, np.ndarray]:
  """The demonstrations as recorded (not replayed): the states the human saw
  and the actions taken in them, in the offline-episode layout."""
  truncated = np.zeros(sum(len(d.actions) for d in demos), dtype=bool)
  truncated[np.cumsum([len(d.actions) for d in demos]) - 1] = True
  return {
    "state": np.concatenate([d.states for d in demos]).astype(np.float32),
    "action": np.concatenate([d.actions for d in demos]).astype(np.float32),
    "truncated": truncated,
  }


def split_episodes(
  data: dict[str, np.ndarray], val_fraction: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Hold out the last ``val_fraction`` of the episodes."""
  ends = np.flatnonzero(data["truncated"]) + 1
  num_val = max(1, int(round(len(ends) * val_fraction)))
  cut_episode = len(ends) - num_val
  cut = int(ends[cut_episode - 1]) if cut_episode > 0 else 0

  def take(lo: int, hi: int, ep_lo: int, ep_hi: int) -> dict[str, np.ndarray]:
    out = {k: v[lo:hi] for k, v in data.items() if k != "episode_success"}
    if "episode_success" in data:
      out["episode_success"] = data["episode_success"][ep_lo:ep_hi]
    return out

  total = len(data["state"])
  return take(0, cut, 0, cut_episode), take(cut, total, cut_episode, len(ends))
