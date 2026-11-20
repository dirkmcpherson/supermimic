"""Online SuperMimic loop: collect, fit the world model, improve the policy.

Mirrors :class:`supermimic.runners.supermimic_runner.SuperMimicRunner` with a
CPU simulator: each round steps the real environments with the current actor,
then runs the repo's world-model and actor-critic learners unchanged. Demos and
exploration branches stay in fixed buffers that are mixed into every
world-model batch and every batch of imagination start states.
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

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.manip import layout as L
from supermimic.manip.data import SCHEMA, MixtureSampler
from supermimic.manip.evaluate import act, evaluate_policy
from supermimic.manip.models import Actor, ManipDynamics, ManipWorldModelLearnerCfg
from supermimic.manip.representation import ArmObjectRepresentation
from supermimic.manip.reward import REWARDS
from supermimic.manip.runtime import gpu_memory_gb
from supermimic.manip.sim import SimCfg
from supermimic.manip.tasks import TASKS
from supermimic.manip.vec_env import StartBank, VecEnv
from supermimic.manip.wm_env import ManipWorldModelEnv
from supermimic.models import build_mlp
from supermimic.replay_buffer import ReplayBuffer


def _wm_cfg() -> ManipWorldModelLearnerCfg:
  # 5 batches of 2048 x 8, half of each from the online buffer: every new
  # transition of a round (107 steps x 24 envs) is drawn about 16 times.
  return ManipWorldModelLearnerCfg(num_batches=5)


def _policy_cfg() -> ActorCriticPolicyLearnerCfg:
  # 16 steps at 20 Hz look as far ahead as the humanoid's 32 steps at 50 Hz.
  return ActorCriticPolicyLearnerCfg(batch_size=2048, horizon=16)


@dataclass
class RunnerCfg:
  task: str = "lift"
  num_envs: int = 24
  eval_envs: int = 8
  episode_steps: int | None = None
  """Control steps of a training episode (default: the task's)."""
  eval_episode_steps: int | None = None
  bank_fraction: float = 0.5
  """Share of training episodes that start from a demo state, not a task reset."""
  buffer_size: int = 40_000
  """Online replay length per env."""
  steps_per_round: int = 107
  """Vectorized env steps between learning rounds."""
  explore_scale: float = 0.3
  """Action-noise std of the noisiest env. The first quarter of the envs act
  noise-free; the rest are spread evenly up to this value."""
  explore_correlation: float = 0.8
  wm_mix: tuple[float, float, float] = (0.5, 0.15, 0.35)
  """Shares of each world-model batch from the online, demo and branch buffers."""
  start_mix: tuple[float, float, float] = (0.5, 0.25, 0.25)
  """Shares of imagination start states from the same three buffers."""
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  wm: ManipWorldModelLearnerCfg = field(default_factory=_wm_cfg)
  policy: ActorCriticPolicyLearnerCfg = field(default_factory=_policy_cfg)
  reward: object = None
  """The task's reward config (default: its defaults)."""
  log_interval: int = 5
  eval_interval: int = 50
  eval_episodes: int = 48
  final_eval_episodes: int = 200
  max_rounds: int = 1_000_000
  max_hours: float = 6.0
  stop_success: float = 0.95
  """Stop early once this success rate is reached on ``stop_patience``
  consecutive evaluations."""
  stop_patience: int = 3
  compile: bool = True
  seed: int = 0

  def __post_init__(self) -> None:
    spec = TASKS[self.task]
    if self.episode_steps is None:
      self.episode_steps = spec.episode_steps
    if self.eval_episode_steps is None:
      self.eval_episode_steps = spec.eval_episode_steps
    if self.reward is None:
      self.reward = REWARDS[self.task][0]()


class Runner:
  def __init__(
    self,
    cfg: RunnerCfg,
    representation: ArmObjectRepresentation,
    dynamics: ManipDynamics,
    demo_buffer: ReplayBuffer,
    branch_buffer: ReplayBuffer,
    bank: StartBank,
    out_dir: Path,
    device: str,
  ):
    self.cfg = cfg
    self.device = device
    self.out = out_dir
    self.representation = representation
    self.dynamics = dynamics

    self.replay_buffer = ReplayBuffer(
      SCHEMA, buffer_size=cfg.buffer_size, num_envs=cfg.num_envs, device=device
    )
    sources = (self.replay_buffer, demo_buffer, branch_buffer)
    self.wm_sampler = MixtureSampler(list(zip(sources, cfg.wm_mix, strict=True)))
    start_sampler = MixtureSampler(list(zip(sources, cfg.start_mix, strict=True)))

    self.world_model_learner = WorldModelLearner(
      dynamics, representation, cfg.wm, device=device
    )
    rep_dim = representation.rep_state_dim
    self.reward_fn = REWARDS[cfg.task][1]
    self.actor = Actor(rep_dim, L.ACTION_DIM, cfg.actor_hidden_dims).to(device)
    self.critic = build_mlp(
      rep_dim + L.CTX_DIM, cfg.critic_hidden_dims, output_dim=1
    ).to(device)
    self.wm_env = ManipWorldModelEnv(
      dynamics,
      representation,
      start_sampler,
      batch_size=cfg.policy.batch_size,
      task=cfg.task,
      reward_cfg=cfg.reward,
      device=device,
    )
    if cfg.compile and torch.device(device).type == "cuda":
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)
    self.policy_learner = ActorCriticPolicyLearner(
      self.actor, self.critic, self.wm_env, cfg.policy, device=device
    )

    sim_cfg = SimCfg(task=cfg.task, horizon=cfg.episode_steps)
    self.env = VecEnv(
      cfg.num_envs, sim_cfg, cfg.seed, bank=bank, bank_fraction=cfg.bank_fraction
    )
    self.eval_env = VecEnv(
      cfg.eval_envs,
      SimCfg(task=cfg.task, horizon=cfg.eval_episode_steps),
      cfg.seed + 500_000,
    )

    # CPU copies for acting in the real envs, refreshed from the trained actor.
    self._cpu_actor = copy.deepcopy(self.actor).cpu()
    self._cpu_representation = copy.deepcopy(representation).cpu()

    self._rng = np.random.default_rng(cfg.seed)
    self._noise = np.zeros((cfg.num_envs, L.ACTION_DIM))
    clean = max(1, cfg.num_envs // 4)
    scale = np.zeros(cfg.num_envs)
    scale[clean:] = np.linspace(0.0, cfg.explore_scale, cfg.num_envs - clean + 1)[1:]
    self._noise_scale = scale[:, None]
    self._clean = self._noise_scale[:, 0] == 0.0
    self._episodes: dict[str, deque] = {
      key: deque(maxlen=200)
      for key in ("task/clean", "task/noisy", "bank/clean", "bank/noisy")
    }
    self.round = 0
    self.env_steps = 0
    self.elapsed = 0.0
    self.best_success = -1.0
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
      "critic": self.critic.state_dict(),
      "dynamics": self.dynamics.state_dict(),
      "representation": self.representation.state_dict(),
      "round": self.round,
      "env_steps": self.env_steps,
      "elapsed": self.elapsed,
      "best_success": self.best_success,
      "evals": self.evals,
    }

  def load_state_dict(self, state: dict) -> None:
    """Restore weights and counters (optimizers and the online buffer restart)."""
    self.actor.load_state_dict(state["actor"])
    self.critic.load_state_dict(state["critic"])
    self.dynamics.load_state_dict(state["dynamics"])
    self.representation.load_state_dict(state["representation"])
    self.round = state["round"]
    self.env_steps = state["env_steps"]
    self.elapsed = state["elapsed"]
    self.best_success = state["best_success"]
    self.evals = state["evals"]

  # ------------------------------------------------------------------
  # Collection
  # ------------------------------------------------------------------

  def _reset_envs(self) -> None:
    out = self.env.reset()
    self._state, self._ctx = out["state"], out["ctx"]
    self._from_bank = out["from_bank"]
    self._prev_action = np.zeros((self.cfg.num_envs, L.ACTION_DIM), dtype=np.float32)
    self._ep_return = np.zeros(self.cfg.num_envs)
    self._ep_success = np.zeros(self.cfg.num_envs, dtype=bool)

  def _tensor(self, value: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=self.device)

  def _sync_cpu_actor(self) -> None:
    self._cpu_actor.load_state_dict(self.actor.state_dict())
    self._cpu_actor.eval()

  def bootstrap_statistics(self, steps: int) -> dict[str, float]:
    """For a world model that starts untrained: collect ``steps`` of the
    untrained actor's noisy rollouts and fit the prediction scale from them
    (the input normalizers update on every collection anyway). The object
    hardly moves in such rollouts, so each scale is kept at or above the
    loss's nominal error for that quantity."""
    self._reset_envs()
    state, next_state = self.collect(steps)
    wm = self.cfg.wm
    floors = {
      "d_arm_vel": wm.nominal_error_arm_vel,
      "d_finger_vel": wm.nominal_error_finger_vel,
      "d_obj_lin_vel": wm.nominal_error_obj_vel,
      "d_obj_ang_vel": wm.nominal_error_obj_avel,
      "r_arm_pos": wm.nominal_error_arm_pos,
      "r_finger_pos": wm.nominal_error_finger_pos,
      "r_obj_pos": wm.nominal_error_root_pose_pos,
      "r_obj_rot": wm.nominal_error_root_pose_ori,
    }
    return self.representation.fit_prediction_scale(state, next_state, floors)

  def collect(self, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Step the real envs with the current actor and store the transitions.

    The stepping loop runs the actor on the CPU and touches no GPU state; the
    round's transitions are moved to the device in one go afterwards. Returns
    the round's states and next states, flattened.
    """
    cfg = self.cfg
    innovation = np.sqrt(1 - cfg.explore_correlation**2)
    self._sync_cpu_actor()
    rows: dict[str, list[np.ndarray]] = {
      k: []
      for k in (
        "state",
        "action",
        "prev_action",
        "sim_ctx",
        "truncated",
        "next_state",
        "success",
        "from_bank",
      )
    }
    for _ in range(steps):
      policy_action, _ = act(
        self._cpu_actor, self._cpu_representation, self._state, "cpu"
      )
      self._noise = cfg.explore_correlation * self._noise + (
        innovation * self._rng.standard_normal(self._noise.shape)
      )
      action = np.clip(
        policy_action.numpy() + self._noise_scale * self._noise, -1.0, 1.0
      ).astype(np.float32)
      out = self.env.step(action)
      for key, value in zip(
        rows,
        (
          self._state,
          action,
          self._prev_action,
          self._ctx,
          out["truncated"],
          out["next_state"],
          out["success"],
          self._from_bank,
        ),
        strict=True,
      ):
        rows[key].append(value)
      self._prev_action = np.where(out["truncated"][:, None], 0.0, action).astype(
        np.float32
      )
      self._from_bank = np.where(out["truncated"], out["from_bank"], self._from_bank)
      self._state, self._ctx = out["state"], out["ctx"]
    self.env_steps += steps * cfg.num_envs

    # (steps, num_envs, ...) on the device.
    state = self._tensor(np.stack(rows["state"]))
    action = self._tensor(np.stack(rows["action"]))
    prev_action = self._tensor(np.stack(rows["prev_action"]))
    sim_ctx = self._tensor(np.stack(rows["sim_ctx"]))
    truncated = self._tensor(np.stack(rows["truncated"]), torch.bool)
    terminated = torch.zeros_like(truncated)
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
          batch_size=[cfg.num_envs],
          device=self.device,
        )
      )

    next_state = self._tensor(np.stack(rows["next_state"]))
    with torch.no_grad():
      next_rep = self.representation.represent(next_state)
      reward = self.representation.dt * self.reward_fn(
        next_rep, action, prev_action, cfg.reward
      )
    reward = reward.cpu().numpy()
    for t in range(steps):
      self._ep_return += reward[t]
      self._ep_success |= rows["success"][t]
      for i in np.flatnonzero(rows["truncated"][t]):
        key = ("bank" if rows["from_bank"][t][i] else "task") + (
          "/clean" if self._clean[i] else "/noisy"
        )
        self._episodes[key].append((self._ep_success[i], self._ep_return[i]))
        self._ep_return[i] = 0.0
        self._ep_success[i] = False

    # Normalizer updates between learner updates (as SuperMimicRunner does).
    flat_state = state.flatten(0, 1)
    self.world_model_learner.update_normalization(flat_state, action.flatten(0, 1))
    self.actor.train()
    with torch.no_grad():
      self.actor.update_normalization(self.representation.represent(flat_state))
    return flat_state, next_state.flatten(0, 1)

  def _episode_stats(self) -> dict[str, float]:
    stats = {}
    for key, episodes in self._episodes.items():
      if episodes:
        success, ret = zip(*episodes, strict=True)
        stats[f"Train/{key}/success"] = float(np.mean(success))
        stats[f"Train/{key}/reward_rate"] = float(np.mean(ret)) / (
          self.cfg.episode_steps * self.representation.dt
        )
    return stats

  # ------------------------------------------------------------------
  # Main loop
  # ------------------------------------------------------------------

  def evaluate(self, episodes: int, env: VecEnv | None = None) -> dict[str, float]:
    """Score the current actor, noise-free, on task-reset episodes."""
    self._sync_cpu_actor()
    return evaluate_policy(
      self._cpu_actor,
      self._cpu_representation,
      env or self.eval_env,
      episodes,
      self.cfg.eval_episode_steps,
      "cpu",
    )

  def _log(self, path: Path, row: dict) -> None:
    with path.open("a") as f:
      f.write(json.dumps(row) + "\n")

  def run(self) -> dict:
    cfg = self.cfg
    (self.out / "runner_cfg.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics_path = self.out / "policy_metrics.jsonl"
    eval_path = self.out / "policy_eval.jsonl"
    start = time.time() - self.elapsed
    timers = {"collect": 0.0, "wm": 0.0, "policy": 0.0}
    logs: dict[str, torch.Tensor] = {}
    streak = 0
    stop_reason = "max_rounds"

    if not hasattr(self, "_state"):
      self._reset_envs()
    # Fill the online buffer enough for the first world-model windows.
    self.collect(4 * cfg.wm.horizon)

    while self.round < cfg.max_rounds:
      self.round += 1
      t0 = time.time()
      self.collect(cfg.steps_per_round)
      t1 = time.time()
      data = (
        self.wm_sampler.sample(cfg.wm.horizon, cfg.wm.batch_size)
        for _ in range(cfg.wm.num_batches)
      )
      logs.update(self.world_model_learner.update(data))
      t2 = time.time()
      logs.update(self.policy_learner.update())
      t3 = time.time()
      timers["collect"] += t1 - t0
      timers["wm"] += t2 - t1
      timers["policy"] += t3 - t2
      self.elapsed = time.time() - start

      if self.round % cfg.log_interval == 0:
        bad = [k for k, v in logs.items() if not bool(torch.isfinite(v).all())]
        if bad:
          # checkpoint_latest.pt predates this; a rerun with --resume restarts there.
          raise FloatingPointError(f"non-finite training values: {bad}")
        row = {
          "round": self.round,
          "env_steps": self.env_steps,
          "hours": self.elapsed / 3600,
          "gpu_gb": gpu_memory_gb(self.device),
          **{f"Time/{k}": v / cfg.log_interval for k, v in timers.items()},
          **{k: float(v) for k, v in logs.items()},
          **self._episode_stats(),
        }
        self._log(metrics_path, row)
        timers = dict.fromkeys(timers, 0.0)

      if self.round % cfg.eval_interval == 0:
        result = {
          "round": self.round,
          "env_steps": self.env_steps,
          "hours": self.elapsed / 3600,
          **self.evaluate(cfg.eval_episodes),
        }
        self.evals.append(result)
        self._log(eval_path, result)
        train = self._episode_stats()
        print(
          f"[POLICY] round {self.round}  {result['hours']:.2f} h  "
          f"{self.env_steps} env steps  eval success {result['success_rate']:.2f}  "
          f"held {result['held_rate']:.2f}  imagined reward "
          f"{float(logs.get('Policy/reward', float('nan'))):.3f}  real "
          f"{train.get('Train/task/clean/reward_rate', float('nan')):.3f}",
          flush=True,
        )
        # Ties go to the later checkpoint.
        if result["success_rate"] >= self.best_success:
          self.best_success = result["success_rate"]
          torch.save(self.state_dict(), self.out / "checkpoint_best.pt")
        torch.save(self.state_dict(), self.out / "checkpoint_latest.pt")
        streak = streak + 1 if result["success_rate"] >= cfg.stop_success else 0
        if streak >= cfg.stop_patience:
          stop_reason = "success"
          break
      if self.elapsed > cfg.max_hours * 3600:
        stop_reason = "max_hours"
        break

    torch.save(self.state_dict(), self.out / "checkpoint_latest.pt")
    return {
      "stop_reason": stop_reason,
      "rounds": self.round,
      "env_steps": self.env_steps,
      "hours": self.elapsed / 3600,
      "best_eval_success": self.best_success,
    }

  def close(self) -> None:
    self.env.close()
    self.eval_env.close()
