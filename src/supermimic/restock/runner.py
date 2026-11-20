"""Online SuperMimic loop on the restock task: collect, fit the world model,
improve the policy. The counterpart of :class:`supermimic.manip.runner.Runner`.

Training episodes start at the environment's own training starts and end at
its own terminals; the re-executed demonstrations stay
in a fixed buffer that is mixed into every world-model batch and every batch
of imagination start states.
"""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.manip.runtime import gpu_memory_gb
from supermimic.models import build_mlp
from supermimic.replay_buffer import ReplayBuffer
from supermimic.restock import layout as L
from supermimic.restock.data import SCHEMA, MixtureSampler
from supermimic.restock.evaluate import act, run_episodes, summarize
from supermimic.restock.models import Actor, ManipDynamics, RestockWorldModelLearnerCfg
from supermimic.restock.outcomes import OutcomeBuffer, discounted_outcomes
from supermimic.restock.policy import RestockPolicyLearner, RestockPolicyLearnerCfg
from supermimic.restock.representation import REP_FLAGS, RestockRepresentation
from supermimic.restock.vec_env import VecEnv, WorkerPool
from supermimic.restock.wm_env import RestockWorldModelEnv

SPEED_COLUMNS = {
  "can_speed": L.CAN_LIN_VEL,
  "can_spin": L.CAN_ANG_VEL,
  "goal_speed": L.GOAL_LIN_VEL,
  "goal_spin": L.GOAL_ANG_VEL,
  "joint_speed": L.DQ,
}
"""Largest norm of each of these velocities among a logging interval's collected
states: m/s, rad/s (logged as ``Collect/max_<name>``)."""


def _wm_cfg() -> RestockWorldModelLearnerCfg:
  return RestockWorldModelLearnerCfg(num_batches=8)


def _policy_cfg() -> RestockPolicyLearnerCfg:
  # 16 decisions are 1.9 s.
  return RestockPolicyLearnerCfg(batch_size=2048, horizon=16, num_batches=4, gamma=0.99)


@dataclass
class RunnerCfg:
  buffer_size: int = 20_000
  """Online replay length per env."""
  steps_per_round: int = 8
  """Vectorized env steps (decisions) between learning rounds."""
  explore_scale: float = 0.3
  """Action-noise std of the noisiest env. The first quarter of the envs act
  noise-free; the rest are spread evenly up to this value."""
  explore_correlation: float = 0.8
  demo_start_fraction: float = 0.0
  """Share of training episodes that start part-way along a demonstration. The
  simulator cannot be set to such a state; it is reached by replaying the tape
  from its start, and those decisions count towards ``max_decisions``."""
  wm_mix: tuple[float, float] = (0.5, 0.5)
  """Shares of each world-model batch from the online and demo buffers."""
  start_mix: tuple[float, float] = (0.5, 0.5)
  """Shares of imagination start states from the same two buffers."""
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (512, 256)
  wm: RestockWorldModelLearnerCfg = field(default_factory=_wm_cfg)
  policy: RestockPolicyLearnerCfg = field(default_factory=_policy_cfg)
  outcome_capacity: int = 60_000
  """Real (state, following return) pairs kept for the critic: the most recent."""
  actor_ema: float = 0.0
  """Per-round decay of an averaged copy of the actor's weights (0: none). With
  it, half of the noise-free envs act with the averaged actor, evaluations
  score it as well, and checkpoints carry it."""
  wm_guard: float = 0.0
  """Skip a round's policy update when the world model's loss on that round's
  batches exceeds this multiple of its recent level (0: never skip). The level
  is a running geometric mean (about the last 100 rounds), so a lasting change
  is followed and only a sudden rise holds the policy back."""
  trace: bool = False
  """Write every round's losses and the largest collected velocities to
  ``round_trace.jsonl`` (the metrics file has every ``log_interval``-th)."""
  log_interval: int = 10
  eval_interval: int = 100
  """Rounds between evaluations on the evaluation starts."""
  max_decisions: int = 250_000
  """Online decisions to collect (1M env frames)."""
  max_hours: float = 12.0
  compile: bool = True
  seed: int = 0


class Runner:
  def __init__(
    self,
    cfg: RunnerCfg,
    representation: RestockRepresentation,
    dynamics: ManipDynamics,
    demo_buffer: ReplayBuffer,
    pool: WorkerPool,
    eval_starts: list[dict],
    out_dir: Path,
    device: str,
    demo_outcomes: tuple[np.ndarray, np.ndarray] | None = None,
  ):
    self.cfg = cfg
    self.device = device
    self.out = out_dir
    self.representation = representation
    self.dynamics = dynamics
    self.pool = pool
    self.env = VecEnv(pool)
    self.num_envs = pool.num
    self.eval_starts = eval_starts
    self.prefix_decisions = 0

    self.replay_buffer = ReplayBuffer(
      SCHEMA, buffer_size=cfg.buffer_size, num_envs=self.num_envs, device=device
    )
    sources = (self.replay_buffer, demo_buffer)
    self.wm_sampler = MixtureSampler(list(zip(sources, cfg.wm_mix, strict=True)))
    start_sampler = MixtureSampler(list(zip(sources, cfg.start_mix, strict=True)))

    self.world_model_learner = WorldModelLearner(
      dynamics, representation, cfg.wm, device=device
    )
    rep_dim = representation.rep_state_dim
    self.actor = Actor(rep_dim, L.ACTION_DIM, cfg.actor_hidden_dims).to(device)
    self.critic = build_mlp(
      rep_dim + L.CTX_DIM, cfg.critic_hidden_dims, output_dim=1
    ).to(device)
    self.wm_env = RestockWorldModelEnv(
      dynamics,
      representation,
      start_sampler,
      batch_size=cfg.policy.batch_size,
      gamma=cfg.policy.gamma,
      device=device,
    )
    if cfg.compile and torch.device(device).type == "cuda":
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)
    self.outcomes = OutcomeBuffer(cfg.outcome_capacity, L.STATE_DIM, device)
    if demo_outcomes is not None:
      self.outcomes.add(*demo_outcomes)
    self.policy_learner = RestockPolicyLearner(
      self.actor, self.critic, self.wm_env, cfg.policy, self.outcomes, device=device
    )
    self.ema_actor = copy.deepcopy(self.actor) if cfg.actor_ema > 0 else None

    # CPU copies for acting in the real envs, refreshed from the trained actor.
    self._cpu_actor = copy.deepcopy(self.actor).cpu()
    self._cpu_ema = copy.deepcopy(self.actor).cpu()
    self._cpu_representation = copy.deepcopy(representation).cpu()

    self._rng = np.random.default_rng(cfg.seed)
    self._noise = np.zeros((self.num_envs, L.ACTION_DIM))
    clean = max(1, self.num_envs // 4)
    scale = np.zeros(self.num_envs)
    scale[clean:] = np.linspace(0.0, cfg.explore_scale, self.num_envs - clean + 1)[1:]
    self._noise_scale = scale[:, None]
    self._clean = scale == 0.0
    # The averaged actor drives the second half of the noise-free envs.
    self._ema_env = np.zeros(self.num_envs, dtype=bool)
    if self.ema_actor is not None:
      self._ema_env[clean // 2 : clean] = True
    self._episodes: dict[str, deque] = {
      key: deque(maxlen=200) for key in ("clean", "ema", "noisy", "demo_start")
    }
    self._home = L.STAGES.index("home")
    self._needs_reset = True
    self.round = 0
    self.policy_decisions = 0
    self.env_steps = 0
    """Decisions simulated for training: the policy's own plus tape prefixes."""
    self.elapsed = 0.0
    self._wm_level: float | None = None
    self._extremes = dict.fromkeys(SPEED_COLUMNS, 0.0)
    self._round_extremes = dict(self._extremes)
    self.best_home = -1.0
    self.evals: list[dict] = []

  # ------------------------------------------------------------------
  # Setup
  # ------------------------------------------------------------------

  @torch.no_grad()
  def init_actor_normalizer(self, states: torch.Tensor) -> None:
    """Seed the actor's input statistics from offline states."""
    self.actor.train()
    for chunk in states.split(65536):
      self.actor.update_normalization(self.representation.represent(chunk))

  def state_dict(self) -> dict:
    return {
      "actor": self.actor.state_dict(),
      **(
        {"actor_ema": self.ema_actor.state_dict()} if self.ema_actor is not None else {}
      ),
      "critic": self.critic.state_dict(),
      "dynamics": self.dynamics.state_dict(),
      "representation": self.representation.state_dict(),
      "round": self.round,
      "env_steps": self.env_steps,
      "elapsed": self.elapsed,
      "best_home": self.best_home,
      "evals": self.evals,
    }

  # ------------------------------------------------------------------
  # Collection
  # ------------------------------------------------------------------

  def _reset_envs(self) -> None:
    self._state, prefix = self.env.reset()
    self.prefix_decisions += int(prefix.sum())
    self._prev_action = np.zeros((self.num_envs, L.ACTION_DIM), dtype=np.float32)
    # Envs whose episode has ended and whose final state is still to be stored.
    self._end_pending = np.zeros(self.num_envs, dtype=bool)
    self._end_terminated = np.zeros(self.num_envs, dtype=bool)
    self._end_state = np.zeros((self.num_envs, L.STATE_DIM), dtype=np.float32)
    # States of each env's running episode, for the outcome buffer.
    self._episode_states: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
    self._needs_reset = False

  def _tensor(self, value: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=self.device)

  def _sync_cpu_actor(self) -> None:
    self._cpu_actor.load_state_dict(self.actor.state_dict())
    self._cpu_actor.eval()
    if self.ema_actor is not None:
      self._cpu_ema.load_state_dict(self.ema_actor.state_dict())
      self._cpu_ema.eval()

  @torch.no_grad()
  def _update_ema(self, decay: float) -> None:
    """Move the averaged actor towards the trained one. Its input statistics
    are not averaged: they are the trained actor's."""
    if self.ema_actor is None:
      return
    for averaged, trained in zip(
      self.ema_actor.parameters(), self.actor.parameters(), strict=True
    ):
      averaged.lerp_(trained, 1.0 - decay)
    for averaged, trained in zip(
      self.ema_actor.buffers(), self.actor.buffers(), strict=True
    ):
      averaged.copy_(trained)

  def collect(self, steps: int, cut: bool = False) -> None:
    """Step the real envs with the current actor and store the transitions.

    The replay buffer keeps one row per env and step: a state and the action
    taken in it. The state an episode ends in gets a row of its own (with a
    null action and the end flag), stored on the following step while that
    env waits; without it the world model would never train on the transition
    into a tipped can or into ``home``. The waiting step is not a decision.

    The stepping loop runs the actor on the CPU and touches no GPU state; the
    round's transitions are moved to the device in one go afterwards. With
    ``cut`` the envs are abandoned after the last step (their episodes are
    marked as cut short), for an evaluation that needs the workers.
    """
    cfg = self.cfg
    if self._needs_reset:
      self._reset_envs()
    innovation = np.sqrt(1 - cfg.explore_correlation**2)
    self._sync_cpu_actor()
    names = ("state", "action", "prev_action", "terminated", "truncated")
    rows: dict[str, list[np.ndarray]] = {k: [] for k in names}
    outcomes: list[tuple[np.ndarray, np.ndarray]] = []
    for step in range(steps):
      waiting = self._end_pending.copy()
      active = np.flatnonzero(~waiting)
      policy_action = act(self._cpu_actor, self._cpu_representation, self._state, "cpu")
      if self._ema_env.any():
        policy_action[self._ema_env] = act(
          self._cpu_ema, self._cpu_representation, self._state[self._ema_env], "cpu"
        )
      self._noise = cfg.explore_correlation * self._noise + (
        innovation * self._rng.standard_normal(self._noise.shape)
      )
      action = np.clip(
        policy_action.numpy() + self._noise_scale * self._noise, -1.0, 1.0
      ).astype(np.float32)
      action[waiting] = 0.0
      terminated = waiting & self._end_terminated
      truncated = waiting & ~self._end_terminated
      if cut and step == steps - 1:
        truncated |= ~terminated
      for key, value in zip(
        names,
        (
          np.where(waiting[:, None], self._end_state, self._state),
          action,
          self._prev_action.copy(),
          terminated,
          truncated,
        ),
        strict=True,
      ):
        rows[key].append(value)
      # The envs that waited start their new episode on the next step.
      self._prev_action[waiting] = 0.0
      self._end_pending[waiting] = False

      if not len(active):
        # Every env's episode ended on the previous step; nothing to simulate.
        continue
      for i in active:
        self._episode_states[i].append(self._state[i].copy())
      out = self.env.step(action[active], active.tolist())
      ended = out["terminated"] | out["truncated"]
      for j in np.flatnonzero(ended):
        i = active[j]
        key = (
          "demo_start"
          if out["from_demo"][j]
          else "ema"
          if self._ema_env[i]
          else "clean"
          if self._clean[i]
          else "noisy"
        )
        self._episodes[key].append(out["stages"][j])
        visited = np.stack(self._episode_states[i])
        paid = float(self.wm_env.home_reward) if out["stages"][j][self._home] else 0.0
        outcomes.append(
          (visited, discounted_outcomes(len(visited), paid, cfg.policy.gamma))
        )
        self._episode_states[i] = []
        self._end_pending[i] = True
        self._end_terminated[i] = out["terminated"][j]
        self._end_state[i] = out["next_state"][j]
      self.prefix_decisions += int(out["prefix_decisions"].sum())
      self._prev_action[active] = action[active]
      self._state[active] = out["state"]
      self.policy_decisions += len(active)
    self.env_steps = self.policy_decisions + self.prefix_decisions
    self._needs_reset = cut
    visited = np.stack(rows["state"])
    self._round_extremes = {
      name: float(np.linalg.norm(visited[..., columns], axis=-1).max())
      for name, columns in SPEED_COLUMNS.items()
    }
    for name, top in self._round_extremes.items():
      self._extremes[name] = max(self._extremes[name], top)
    if outcomes:
      self.outcomes.add(
        np.concatenate([s for s, _ in outcomes]),
        np.concatenate([g for _, g in outcomes]),
      )

    # (steps, num_envs, ...) on the device.
    state = self._tensor(np.stack(rows["state"]))
    action = self._tensor(np.stack(rows["action"]))
    prev_action = self._tensor(np.stack(rows["prev_action"]))
    terminated = self._tensor(np.stack(rows["terminated"]), torch.bool)
    truncated = self._tensor(np.stack(rows["truncated"]), torch.bool)
    sim_ctx = torch.zeros(steps, self.num_envs, L.CTX_DIM, device=self.device)
    for t in range(steps):
      self.replay_buffer.insert(
        TensorDict(
          {
            "state": state[t],
            "action": action[t],
            "prev_action": prev_action[t],
            "sim_ctx": sim_ctx[t],
            "terminated": terminated[t],
            "truncated": truncated[t],
          },
          batch_size=[self.num_envs],
          device=self.device,
        )
      )

    # Normalizer updates between learner updates (as SuperMimicRunner does).
    flat_state = state.flatten(0, 1)
    self.world_model_learner.update_normalization(flat_state, action.flatten(0, 1))
    self.actor.train()
    with torch.no_grad():
      self.actor.update_normalization(self.representation.represent(flat_state))

  @torch.no_grad()
  def _model_check(self) -> dict[str, float]:
    """Does the world model see real episode ends coming? On one-step windows
    of online data: the end probability (home or tipped) it assigns to the
    transitions that really ended an episode in the environment's terminal,
    and to those that did not; and the same rule read off the true next state."""
    if self.replay_buffer.num_transitions < 4:
      return {}
    batch = self.replay_buffer.sample(2, 4096)
    state, rep = batch["state"], self.representation

    def end_prob(rep_state: torch.Tensor) -> torch.Tensor:
      scene = rep.scene(rep_state)
      home = rep.rules.home(scene, rep_state[..., REP_FLAGS])
      return 1.0 - (1.0 - home) * (1.0 - rep.rules.tipped(scene))

    _, predicted = self.dynamics(
      rep.represent(state[:, 0]),
      batch["action"][:, 0],
      rep.extract_aux_state(state[:, 0]),
      batch["sim_ctx"][:, 0],
    )
    predicted, actual = end_prob(predicted), end_prob(rep.represent(state[:, 1]))
    ended = batch["terminated"][:, 1]
    out = {"Model/end_share": float(ended.float().mean())}
    if ended.any():
      out["Model/end_prob_at_ends"] = float(predicted[ended].mean())
      out["Model/rule_at_ends"] = float(actual[ended].mean())
    if (~ended).any():
      out["Model/end_prob_elsewhere"] = float(predicted[~ended].mean())
    if self.outcomes.size:
      # The critic on real states against the return that followed them.
      real_state, followed = self.outcomes.sample(4096)
      value = self.critic(self.policy_learner.critic_obs(real_state)).squeeze(-1)
      out["Value/real_states"] = float(value.mean())
      out["Value/real_return"] = float(followed.mean())
      out["Value/abs_error"] = float((value - followed).abs().mean())
    return out

  def _episode_stats(self) -> dict[str, float]:
    stats = {}
    for key, episodes in self._episodes.items():
      if episodes:
        rates = np.mean(np.stack(episodes), axis=0)
        for name, rate in zip(L.STAGES, rates, strict=True):
          stats[f"Train/{key}/{name}"] = float(rate)
    return stats

  # ------------------------------------------------------------------
  # Main loop
  # ------------------------------------------------------------------

  def evaluate(self) -> dict[str, float]:
    """Score the current actor, noise-free, on the evaluation starts, on the
    training workers (whose episodes must have been cut)."""
    assert self._needs_reset, "the training episodes were not cut"
    self._sync_cpu_actor()
    rep = self._cpu_representation
    result = summarize(run_episodes(self.pool, self._cpu_actor, rep, self.eval_starts))
    if self.ema_actor is not None:
      averaged = summarize(
        run_episodes(self.pool, self._cpu_ema, rep, self.eval_starts)
      )
      result |= {f"ema_{k}": v for k, v in averaged.items() if k != "episodes"}
    return result

  def _log(self, path: Path, row: dict) -> None:
    with path.open("a") as f:
      f.write(json.dumps(row) + "\n")

  def run(self) -> dict:
    cfg = self.cfg
    (self.out / "runner_cfg.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics_path = self.out / "policy_metrics.jsonl"
    eval_path = self.out / "policy_eval.jsonl"
    start = time.time() - self.elapsed
    timers = {"collect": 0.0, "wm": 0.0, "policy": 0.0, "eval": 0.0}
    logs: dict[str, torch.Tensor] = {}
    skipped = 0
    stop_reason = "max_decisions"

    self._update_ema(0.0)
    # Fill the online buffer enough for the first world-model windows.
    self.collect(4 * cfg.wm.horizon)

    while self.env_steps < cfg.max_decisions:
      self.round += 1
      evaluating = self.round % cfg.eval_interval == 0
      t0 = time.time()
      self.collect(cfg.steps_per_round, cut=evaluating)
      t1 = time.time()
      data = (
        self.wm_sampler.sample(cfg.wm.horizon, cfg.wm.batch_size)
        for _ in range(cfg.wm.num_batches)
      )
      logs.update(self.world_model_learner.update(data))
      t2 = time.time()
      trusted = self._model_trusted(float(logs["Loss/state"]))
      if trusted:
        logs.update(self.policy_learner.update())
        self._update_ema(cfg.actor_ema)
      else:
        skipped += 1
      t3 = time.time()
      if cfg.trace:
        row = {
          "round": self.round,
          "policy_updated": trusted,
          **{k: float(v) for k, v in logs.items()},
          **{f"Collect/max_{k}": v for k, v in self._round_extremes.items()},
        }
        self._log(self.out / "round_trace.jsonl", row)
      timers["collect"] += t1 - t0
      timers["wm"] += t2 - t1
      timers["policy"] += t3 - t2
      self.elapsed = time.time() - start

      if self.round % cfg.log_interval == 0:
        bad = [k for k, v in logs.items() if not bool(torch.isfinite(v).all())]
        if bad:
          raise FloatingPointError(f"non-finite training values: {bad}")
        row = {
          "round": self.round,
          "env_steps": self.env_steps,
          "prefix_decisions": self.prefix_decisions,
          "hours": self.elapsed / 3600,
          "gpu_gb": gpu_memory_gb(self.device),
          **{f"Time/{k}": v / cfg.log_interval for k, v in timers.items()},
          **{k: float(v) for k, v in logs.items()},
          "Policy/skipped_rounds": skipped,
          "Model/loss_level": self._wm_level,
          **{f"Collect/max_{k}": v for k, v in self._extremes.items()},
          **self._model_check(),
          **self._episode_stats(),
        }
        self._log(metrics_path, row)
        timers = dict.fromkeys(timers, 0.0)
        skipped = 0
        self._extremes = dict.fromkeys(SPEED_COLUMNS, 0.0)

      if evaluating:
        t4 = time.time()
        result = {
          "round": self.round,
          "env_steps": self.env_steps,
          "hours": self.elapsed / 3600,
          **self.evaluate(),
        }
        timers["eval"] += time.time() - t4
        self.evals.append(result)
        self._log(eval_path, result)
        train = self._episode_stats()
        print(
          f"[POLICY] round {self.round}  {result['hours']:.2f} h  "
          f"{self.env_steps} decisions  eval home {result['home']:.2f}  "
          f"picked {result['picked']:.2f}  placed {result['placed_v2']:.2f}  "
          f"tipped {result['tipped']:.2f}  averaged-actor home "
          f"{result.get('ema_home', float('nan')):.2f}  | train (clean) picked "
          f"{train.get('Train/clean/picked', float('nan')):.2f}  home "
          f"{train.get('Train/clean/home', float('nan')):.2f}  | imagined return "
          f"{self._imagined_return(logs):.3f}",
          flush=True,
        )
        self.best_home = max(self.best_home, result["home"])
        torch.save(self.state_dict(), self.out / "checkpoint_latest.pt")
      if self.elapsed > cfg.max_hours * 3600:
        stop_reason = "max_hours"
        break

    torch.save(self.state_dict(), self.out / "checkpoint_final.pt")
    return {
      "stop_reason": stop_reason,
      "rounds": self.round,
      "env_steps": self.env_steps,
      "prefix_decisions": self.prefix_decisions,
      "hours": self.elapsed / 3600,
      "best_eval_home": self.best_home,
    }

  def _model_trusted(self, loss: float) -> bool:
    """Whether this round's world-model loss is near its recent level; also
    moves that level towards it."""
    first = self._wm_level is None
    level = loss if first else self._wm_level
    guard = self.cfg.wm_guard
    trusted = guard <= 0 or first or loss <= guard * level
    self._wm_level = float(np.exp(0.99 * np.log(level) + 0.01 * np.log(loss)))
    return trusted

  def _imagined_return(self, logs: dict[str, torch.Tensor]) -> float:
    """Mean discounted return of the imagined rollouts (the learner logs it
    per step and per second)."""
    if "Policy/reward" not in logs:
      return float("nan")
    return (
      float(logs["Policy/reward"]) * self.cfg.policy.horizon * self.representation.dt
    )
