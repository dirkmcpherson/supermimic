"""Differentiable restock environment: learned dynamics, and the environment's
own end-of-episode rules made smooth.

The only reward is the task's: ``home_reward`` (10) when the episode ends in
``home``. Nothing is paid for reaching, grasping, lifting or placing. An
episode also ends, unpaid, when a free can tips over.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.manip.models import ManipDynamics
from supermimic.manip.wm_env import actor_critic_obs
from supermimic.restock import layout as L
from supermimic.restock.data import MixtureSampler
from supermimic.restock.representation import REP_FLAGS, RestockRepresentation


class RestockWorldModelEnv:
  def __init__(
    self,
    dynamics: ManipDynamics,
    representation: RestockRepresentation,
    start_states: MixtureSampler,
    batch_size: int,
    gamma: float,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.start_states = start_states
    self.num_envs = batch_size
    self.home_reward = representation.rules.home_reward
    # The learner caps values at max_step_reward / (1 - gamma); the most an
    # episode can return is the one terminal reward.
    self.max_step_reward = self.home_reward * (1.0 - gamma)

  @property
  def num_actions(self) -> int:
    return L.ACTION_DIM

  def sample_initial_states(self) -> TensorDict:
    # The first state of a two-step window is never one an episode ended in.
    batch = self.start_states.sample(2, self.num_envs)[:, 0]
    return batch.select("state", "prev_action", "sim_ctx")

  def reset(self, batch: TensorDict | None = None) -> TensorDict:
    if batch is None:
      batch = self.sample_initial_states()
    full_state = batch["state"]
    self.aux_state = self.representation.extract_aux_state(full_state).detach()
    self.rep_state = self.representation.represent(full_state).detach()
    self.sim_ctx = batch["sim_ctx"]
    return self.get_observations()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    self.aux_state, self.rep_state = self.dynamics(
      self.rep_state, actions, self.aux_state, self.sim_ctx
    )
    rules = self.representation.rules
    scene = self.representation.scene(self.rep_state)
    home = rules.home(scene, self.rep_state[..., REP_FLAGS])
    tipped = rules.tipped(scene)
    reward = self.home_reward * home
    term_prob = 1.0 - (1.0 - home) * (1.0 - tipped)
    return self.get_observations(), reward, term_prob, {}

  def get_observations(self) -> TensorDict:
    return actor_critic_obs(self.dynamics, self.rep_state, self.sim_ctx)
