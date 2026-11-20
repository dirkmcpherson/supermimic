"""Analytic task rewards on the rep state, differentiable for BPTT.

Each follows robosuite's shaped reward for its task, changed only where that
reward is not a smooth function of the state. The same function scores real
and imagined states; task success is always measured with robosuite's own
test.

Lift (reach + grasp + lift): the contact-based grasp test is replaced by a
geometric one (fingers closed to a cube's width around a cube at the grip
point), and the lift step at 4 cm by a ramp over the cube's height.

Square (reach, grasp, lift, hover over the peg, success): robosuite pays the
highest of its stages; here the stages add up, with the same increments
(times four). The grasp test is geometric as in Lift. Two changes remove the
drop in reward robosuite has between hovering over the peg and success: the
lift height is no longer required once the nut is over the peg, and the
success bonus ramps up as the nut goes down the peg instead of switching on
at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.manip import layout as L
from supermimic.manip.representation import (
  REP_EEF_POS,
  REP_GRASP_POS,
  REP_OBJ_FROM_GOAL,
  REP_OBJ_POS,
  REP_Q,
)


@dataclass(frozen=True)
class LiftRewardCfg:
  reach_weight: float = 1.0
  reach_scale: float = 10.0
  """``1 - tanh(scale * distance)`` between the grip point and the cube centre."""
  grasp_weight: float = 0.25
  grasp_radius: float = 0.02
  """Grip point within this distance (m) of the cube centre counts as around it."""
  grasp_radius_width: float = 0.005
  grasp_gap: tuple[float, float] = (0.036, 0.062)
  """Finger opening (m) of fingers stalled on the cube: its width across a face
  (0.040-0.044) up to its diagonal. Fully open is 0.08, closed on nothing 0."""
  grasp_gap_width: float = 0.003
  lift_weight: float = 2.0
  lift_height: tuple[float, float] = (0.022, 0.08)
  """Cube-centre height above the table (m) where the lift term starts (a cube
  at rest sits at 0.020-0.022) and where it saturates. robosuite's success
  threshold is 0.04."""
  action_rate_weight: float = 0.01

  @property
  def max_reward(self) -> float:
    return self.reach_weight + self.grasp_weight + self.lift_weight


def lift_reward_terms(
  rep_state: torch.Tensor, cfg: LiftRewardCfg
) -> dict[str, torch.Tensor]:
  """Unweighted ``reach``, ``grasp`` and ``lift`` terms in ``[0, 1]``."""
  cube = rep_state[..., REP_OBJ_POS]
  offset = rep_state[..., REP_EEF_POS] - cube
  dist = torch.sqrt((offset * offset).sum(-1) + 1e-10)
  q = rep_state[..., REP_Q]
  gap = q[..., L.NUM_ARM] - q[..., L.NUM_ARM + 1]

  reach = 1 - torch.tanh(cfg.reach_scale * dist)
  near = torch.sigmoid((cfg.grasp_radius - dist) / cfg.grasp_radius_width)
  lo, hi = cfg.grasp_gap
  closed = torch.sigmoid((gap - lo) / cfg.grasp_gap_width) * torch.sigmoid(
    (hi - gap) / cfg.grasp_gap_width
  )
  z_lo, z_hi = cfg.lift_height
  lift = ((cube[..., 2] - z_lo) / (z_hi - z_lo)).clamp(0.0, 1.0)
  return {"reach": reach, "grasp": near * closed, "lift": lift}


def lift_reward(
  rep_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  cfg: LiftRewardCfg,
) -> torch.Tensor:
  """Per-env reward rate (multiply by the control step for a per-step reward)."""
  terms = lift_reward_terms(rep_state, cfg)
  action_rate = ((action - last_action) ** 2).sum(-1)
  return (
    cfg.reach_weight * terms["reach"]
    + cfg.grasp_weight * terms["grasp"]
    + cfg.lift_weight * terms["lift"]
    - cfg.action_rate_weight * action_rate
  )


@dataclass(frozen=True)
class SquareRewardCfg:
  reach_weight: float = 0.4
  reach_scale: float = 10.0
  """``1 - tanh(scale * distance)`` between the grip point and the nut's handle."""
  grasp_weight: float = 1.0
  grasp_radius: float = 0.025
  """Grip point within this distance (m) of the handle counts as around it
  (demonstrations carry the nut at 0.01, 95% within 0.027)."""
  grasp_radius_width: float = 0.005
  grasp_gap: tuple[float, float] = (0.016, 0.036)
  """Finger opening (m) of fingers stalled on the handle: 0.030 across its
  width, 0.020 across its thickness. Fully open is 0.08, closed on nothing 0."""
  grasp_gap_width: float = 0.003
  lift_weight: float = 0.6
  lift_target: float = 0.2
  """Nut height above the table (m) the lift term aims for, as in robosuite
  (the peg is 0.13 tall)."""
  lift_scale: float = 15.0
  hover_weight: float = 0.8
  hover_scale: float = 10.0
  """``1 - tanh(scale * distance)`` between the nut's centre and the peg axis."""
  over_radius: float = 0.014
  """Horizontal distance (m) from the peg axis below which the nut counts as
  over (or on) the peg; the hole has 0.007 of clearance around the peg."""
  over_radius_width: float = 0.0025
  seat_weight: float = 1.2
  seat_height: tuple[float, float] = (0.012, 0.15)
  """Nut height above the table (m) when seated (it rests at 0.010) and when
  it just clears the top of the peg."""
  release_distance: float = 0.0424
  """robosuite counts the nut as placed only if the grip point is further than
  this (m) from the nut's centre; the handle is 0.054 from it."""
  release_distance_width: float = 0.004
  action_rate_weight: float = 0.01

  @property
  def max_reward(self) -> float:
    return (
      self.reach_weight
      + self.grasp_weight
      + self.lift_weight
      + self.hover_weight
      + self.seat_weight
    )


def square_reward_terms(
  rep_state: torch.Tensor, cfg: SquareRewardCfg
) -> dict[str, torch.Tensor]:
  """Unweighted ``reach``, ``grasp``, ``lift``, ``hover`` and ``seat`` in ``[0, 1]``."""
  eef = rep_state[..., REP_EEF_POS]
  nut = rep_state[..., REP_OBJ_POS]
  to_handle = eef - rep_state[..., REP_GRASP_POS]
  handle_dist = torch.sqrt((to_handle * to_handle).sum(-1) + 1e-10)
  to_centre = eef - nut
  centre_dist = torch.sqrt((to_centre * to_centre).sum(-1) + 1e-10)
  from_peg = rep_state[..., REP_OBJ_FROM_GOAL]
  peg_dist = torch.sqrt((from_peg[..., :2] ** 2).sum(-1) + 1e-10)
  height = nut[..., 2]
  q = rep_state[..., REP_Q]
  gap = q[..., L.NUM_ARM] - q[..., L.NUM_ARM + 1]

  reach = 1 - torch.tanh(cfg.reach_scale * handle_dist)
  near = torch.sigmoid((cfg.grasp_radius - handle_dist) / cfg.grasp_radius_width)
  lo, hi = cfg.grasp_gap
  closed = torch.sigmoid((gap - lo) / cfg.grasp_gap_width) * torch.sigmoid(
    (hi - gap) / cfg.grasp_gap_width
  )
  grasp = near * closed

  over = torch.sigmoid((cfg.over_radius - peg_dist) / cfg.over_radius_width)
  z_seat, z_clear = cfg.seat_height
  lowered = ((z_clear - height) / (z_clear - z_seat)).clamp(0.0, 1.0)
  released = torch.sigmoid(
    (centre_dist - cfg.release_distance) / cfg.release_distance_width
  )
  seat = over * lowered * released

  raised = 1 - torch.tanh(cfg.lift_scale * (cfg.lift_target - height).clamp(min=0.0))
  # Height counts while the nut is held, and no longer once it is over the peg.
  held = grasp + seat - grasp * seat
  lift = held * (raised * (1 - over) + over)
  hover = lift * (1 - torch.tanh(cfg.hover_scale * peg_dist))
  return {"reach": reach, "grasp": grasp, "lift": lift, "hover": hover, "seat": seat}


def square_reward(
  rep_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  cfg: SquareRewardCfg,
) -> torch.Tensor:
  """Per-env reward rate (multiply by the control step for a per-step reward)."""
  terms = square_reward_terms(rep_state, cfg)
  action_rate = ((action - last_action) ** 2).sum(-1)
  return (
    cfg.reach_weight * terms["reach"]
    + cfg.grasp_weight * terms["grasp"]
    + cfg.lift_weight * terms["lift"]
    + cfg.hover_weight * terms["hover"]
    + cfg.seat_weight * terms["seat"]
    - cfg.action_rate_weight * action_rate
  )


REWARDS = {
  "lift": (LiftRewardCfg, lift_reward, lift_reward_terms),
  "square": (SquareRewardCfg, square_reward, square_reward_terms),
}
"""Per task: the reward's config class, the reward and its unweighted terms."""
