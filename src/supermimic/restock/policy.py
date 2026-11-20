"""The actor-critic learner with a critic that can be fitted to real outcomes.

The stock learner fits the critic to returns of the imagined rollouts,
bootstrapped from the critic itself; nothing ties it to what real episodes
return. ``critic_targets`` chooses what it is fitted to instead:

- ``imagined``: the stock behaviour.
- ``real``: only the returns real episodes went on to collect
  (:class:`supermimic.restock.outcomes.OutcomeBuffer`).
- ``mixed``: the imagined returns, then the real ones.

``actor_lam`` chooses the return the actor climbs. At 1 it is the stock one:
the imagined rewards of the whole rollout plus the critic's value of its last
state, so the critic is consulted once, ``horizon`` model steps away. Below 1
it is the lambda-return, which also weighs the critic's value of every state
on the way (``1 - actor_lam`` of what remains at each step). With a critic
fitted to real outcomes this lets real experience veto a state the world
model wrongly expects to pass through safely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.learner_utils import _optimize_step
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.manip.wm_env import actor_critic_obs
from supermimic.restock import layout as L
from supermimic.restock.outcomes import OutcomeBuffer

CRITIC_TARGETS = ("imagined", "real", "mixed")


@dataclass
class RestockPolicyLearnerCfg(ActorCriticPolicyLearnerCfg):
  critic_targets: str = "imagined"
  real_critic_steps: int = 4
  """Critic steps on real outcomes per actor step (one batch each)."""
  actor_lam: float = 1.0


class RestockPolicyLearner(ActorCriticPolicyLearner):
  def __init__(
    self,
    policy: nn.Module,
    critic: nn.Module,
    wm_env,
    cfg: RestockPolicyLearnerCfg,
    outcomes: OutcomeBuffer,
    device: str = "cpu",
  ):
    if cfg.critic_targets not in CRITIC_TARGETS:
      raise ValueError(f"critic_targets must be one of {CRITIC_TARGETS}")
    super().__init__(policy, critic, wm_env, cfg, device)
    self.outcomes = outcomes
    self._real: tuple[torch.Tensor, torch.Tensor] | None = None

  @property
  def uses_outcomes(self) -> bool:
    return self.cfg.critic_targets != "imagined"

  def update(self) -> dict[str, torch.Tensor]:
    self.actor.train()
    for module in (self.wm_env.dynamics, self.wm_env.representation):
      module.eval()
      module.requires_grad_(False)
    cfg = self.cfg
    batch = torch.stack(
      [self.wm_env.sample_initial_states() for _ in range(cfg.num_batches)]
    )
    if self.uses_outcomes:
      count = cfg.batch_size * cfg.real_critic_steps
      state, returns = self.outcomes.sample(cfg.num_batches * count)
      # Same leading shape as the starts; the critic steps split the rest.
      batch["real_state"] = state.view(
        cfg.num_batches, cfg.batch_size, cfg.real_critic_steps, -1
      )
      batch["real_return"] = returns.view(
        cfg.num_batches, cfg.batch_size, cfg.real_critic_steps
      )
    return self._captured_update(batch)

  def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
    cfg = self.cfg
    if self.uses_outcomes:
      self._real = (start["real_state"], start["real_return"])
    if cfg.actor_lam >= 1.0:
      return super()._train_batch(start)

    obs = self.wm_env.reset(start)
    self.critic.requires_grad_(False)
    rewards, alive, values = [], [], []
    rollout: list[TensorDict] = []
    for _ in range(cfg.horizon):
      actions = self._rollout_actor(obs)
      next_obs, reward, term_prob, _ = self.wm_env.step(actions)
      step = {"critic": obs["critic"], "reward": reward, "done": term_prob}
      rollout.append(TensorDict(step, batch_size=reward.shape).detach())
      rewards.append(reward)
      alive.append(1.0 - term_prob)
      values.append(self.critic(next_obs["critic"]).clamp(max=self._v_max).squeeze(-1))
      obs = next_obs
    # G_t = r_t + gamma * alive_t * ((1 - lam) * V(s_{t+1}) + lam * G_{t+1}),
    # with G past the last step replaced by the critic's value.
    ret = values[-1]
    for t in reversed(range(cfg.horizon)):
      ahead = cfg.actor_lam * ret + (1.0 - cfg.actor_lam) * values[t]
      ret = rewards[t] + cfg.gamma * alive[t] * ahead
    grad_norm = _optimize_step(
      self.actor_optimizers, (-ret).mean(), self.actor_params, 1.0, cfg.grad_clip_value
    )
    return {
      "return": ret.mean().detach(),
      "grad_norm": grad_norm,
      **self._fit_critic(rollout, obs),
    }

  @torch.no_grad()
  def critic_obs(self, state: torch.Tensor) -> torch.Tensor:
    """The critic's input for real simulator states."""
    wm = self.wm_env
    ctx = state.new_zeros(*state.shape[:-1], L.CTX_DIM)
    rep_state = wm.representation.represent(state)
    return actor_critic_obs(wm.dynamics, rep_state, ctx)["critic"]

  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    if not self.uses_outcomes:
      return super()._fit_critic(rollout, last_obs)
    stats = {}
    if self.cfg.critic_targets == "mixed":
      stats = super()._fit_critic(rollout, last_obs)
    assert self._real is not None
    state, returns = self._real
    self.critic.requires_grad_(True)
    losses = []
    for k in range(self.cfg.real_critic_steps):
      obs = self.critic_obs(state[:, k])
      loss = (returns[:, k, None] - self.critic(obs)).pow(2).mean()
      _optimize_step(
        self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0
      )
      losses.append(loss.detach())
    return stats | {"value_function_real": torch.stack(losses).mean()}
