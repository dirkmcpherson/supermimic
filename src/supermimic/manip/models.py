"""Dynamics, actor and learner config for the manipulation task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.manip.representation import ArmObjectRepresentation
from supermimic.models import Dynamics, _CudaGraphNormalization, build_mlp


@dataclass
class ManipWorldModelLearnerCfg(WorldModelLearnerCfg):
  """Bryson's-rule nominal errors for the arm-and-cube loss groups."""

  # Cube pose (the ``root_pos`` / ``root_ori`` groups): metres, radians.
  nominal_error_root_pose_pos: float = 0.002
  nominal_error_root_pose_ori: float = 0.02
  nominal_error_arm_pos: float = 0.005
  nominal_error_arm_vel: float = 0.1
  # Finger joints are prismatic: metres, metres per second.
  nominal_error_finger_pos: float = 0.0005
  nominal_error_finger_vel: float = 0.01
  nominal_error_obj_vel: float = 0.02
  nominal_error_obj_avel: float = 0.2


class ManipDynamics(Dynamics):
  """``Dynamics`` whose integration step also sees the executed command.

  The environment clips actions to ``[-1, 1]`` before its controller scales
  them, so ``proc`` clips; the gripper command (controller memory, part of the
  state) is advanced analytically from the clipped action.
  """

  def __init__(
    self,
    representation: ArmObjectRepresentation,
    num_actions: int,
    sim_ctx_dim: int,
    hidden_dims: tuple[int, ...],
  ):
    super().__init__(
      representation,
      num_actions,
      sim_ctx_dim,
      action_scale=torch.ones(num_actions),
      action_offset=torch.zeros(num_actions),
      hidden_dims=hidden_dims,
    )

  def proc(self, action: torch.Tensor) -> torch.Tensor:
    return action.clamp(-1.0, 1.0)

  def forward(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    command = self.proc(action)
    inputs = torch.cat(
      [self.state_normalizer(rep_state), self.action_normalizer(command), sim_ctx],
      dim=-1,
    )
    return self._apply_prediction(
      aux_state, rep_state, self.mlp(inputs), self.dt, command
    )


class Actor(nn.Module):
  """State-feedback actor: normalized rep state -> action in ``[-1, 1]``."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
    super().__init__()
    self.obs_normalizer = _CudaGraphNormalization(obs_dim)
    self.mlp = build_mlp(obs_dim, hidden_dims, output_dim=action_dim)

  def forward(self, obs_td: TensorDict) -> torch.Tensor:
    return torch.tanh(self.mlp(self.obs_normalizer(obs_td["actor"])))

  @torch.jit.unused
  def update_normalization(self, obs: torch.Tensor) -> None:
    self.obs_normalizer.update(obs)
