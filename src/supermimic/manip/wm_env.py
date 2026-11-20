"""Differentiable task environment: learned dynamics plus the analytic reward.

Implements the surface the policy learners use from
:class:`supermimic.envs.world_model_env.WorldModelEnv` (``reset``, ``step``,
``sample_initial_states``, ``dynamics``, ``representation``,
``max_step_reward``) without a reference motion.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.manip import layout as L
from supermimic.manip.data import MixtureSampler
from supermimic.manip.models import ManipDynamics
from supermimic.manip.representation import ArmObjectRepresentation
from supermimic.manip.reward import REWARDS


def actor_critic_obs(
  dynamics: ManipDynamics, rep_state: torch.Tensor, sim_ctx: torch.Tensor
) -> TensorDict:
  """``actor``: the rep state (the actor normalizes it itself). ``critic``: the
  rep state normalized by the dynamics' running statistics, plus the
  privileged context."""
  critic = torch.cat([dynamics.state_normalizer(rep_state), sim_ctx], dim=-1)
  return TensorDict(
    {"actor": rep_state, "critic": critic},
    batch_size=[rep_state.shape[0]],
    device=rep_state.device,
  )


class ManipWorldModelEnv:
  def __init__(
    self,
    dynamics: ManipDynamics,
    representation: ArmObjectRepresentation,
    start_states: MixtureSampler,
    batch_size: int,
    task: str,
    reward_cfg: object,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.start_states = start_states
    self.num_envs = batch_size
    self.reward_cfg = reward_cfg
    self.reward_fn = REWARDS[task][1]
    # Analytic per-step reward ceiling: positive reward weights, dt-scaled.
    self.max_step_reward = representation.dt * reward_cfg.max_reward

  @property
  def num_actions(self) -> int:
    return L.ACTION_DIM

  def sample_initial_states(self) -> TensorDict:
    batch = self.start_states.sample(1, self.num_envs).squeeze(1)
    return batch.select("state", "prev_action", "sim_ctx")

  def reset(self, batch: TensorDict | None = None) -> TensorDict:
    if batch is None:
      batch = self.sample_initial_states()
    full_state = batch["state"]
    self.aux_state = self.representation.extract_aux_state(full_state).detach()
    self.rep_state = self.representation.represent(full_state).detach()
    self.last_actions = batch["prev_action"].detach()
    self.sim_ctx = batch["sim_ctx"]
    return self.get_observations()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    self.aux_state, self.rep_state = self.dynamics(
      self.rep_state, actions, self.aux_state, self.sim_ctx
    )
    reward = self.representation.dt * self.reward_fn(
      self.rep_state, actions, self.last_actions, self.reward_cfg
    )
    self.last_actions = actions
    # The tasks have no failure termination; episodes only time out.
    return self.get_observations(), reward, torch.zeros_like(reward), {}

  def get_observations(self) -> TensorDict:
    return actor_critic_obs(self.dynamics, self.rep_state, self.sim_ctx)
