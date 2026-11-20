"""Differentiable versions of the environment's stage rules.

The environment decides success with hard thresholds and latches that live
outside the physics state (``baselines/stage_predicates.py`` and
``baselines/rl/full_env.py`` in the task repository). The world model carries
the latches in its state and advances them with the same rules, each threshold
replaced by a sigmoid of the given width, so the reward for ``home`` has a
gradient. Thresholds come from the built environment's constants; only the
widths and the two approximations below are chosen here.

Approximations (the rules are per 0.03 s frame, the model steps 0.12 s):
  * conditions the environment wants sustained for 10 frames (``picked``, the
    release) or 4 (the tip) are read once per decision, unsustained;
  * ``at_rest`` (the can's xy moved at most 2 mm over 12 frames) is read from
    the can's horizontal speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

PICKED, RELEASED, SETTLED, FARSIDE, SLIDE_EVENT = range(5)


@dataclass
class SoftCfg:
  """Sigmoid widths, in the unit of the quantity each one softens."""

  height: float = 0.004
  footprint: float = 0.01
  grip: float = 0.03
  lever: float = 0.003
  reach: float = 0.008
  wrist: float = 0.01
  cone: float = 0.08
  upright: float = 0.02
  tip: float = 0.05
  touch: float = 0.004
  slide_gain: float = 0.003
  rest_speed: float = 0.006
  """Horizontal can speed (m/s) that still counts as at rest."""
  rest_speed_width: float = 0.0015


def above(value: torch.Tensor, threshold: float, width: float) -> torch.Tensor:
  return torch.sigmoid((value - threshold) / width)


def below(value: torch.Tensor, threshold: float, width: float) -> torch.Tensor:
  return torch.sigmoid((threshold - value) / width)


@dataclass
class Scene:
  """What the rules read of one state; every entry is ``(...,)`` unless noted."""

  can_pos: torch.Tensor
  """``(..., 3)``"""
  can_speed_xy: torch.Tensor
  can_up: torch.Tensor
  """Cosine of the can's tilt from vertical."""
  goal_up: torch.Tensor
  lever: torch.Tensor
  """Horizontal distance from the tool point to the can's centre."""
  dist: torch.Tensor
  """Horizontal distance between the two cans' centres."""
  cos_far: torch.Tensor
  """Cosine of the angle, seen from the can, between the tool and the ray
  pointing away from the goal (1: the tool is right behind the can)."""
  wrist_dist: torch.Tensor
  """Distance from the wrist link to the can's centre."""
  grip_cmd: torch.Tensor


class StageRules(nn.Module):
  def __init__(self, constants: dict[str, np.ndarray], cfg: SoftCfg | None = None):
    super().__init__()
    self.cfg = cfg or SoftCfg()
    c = {
      k: float(v)
      for k, v in constants.items()
      if np.ndim(v) == 0 and v.dtype.kind == "f"
    }
    shelf = c["shelf_top_z"]
    self.band = (shelf + c["band_lo"], shelf + c["band_hi"])
    self.shelf_lo = [float(v) for v in constants["shelf_lo"]]
    self.shelf_hi = [float(v) for v in constants["shelf_hi"]]
    self.pick_z = c["pick_z"]
    self.pick_grip = c["pick_grip"]
    self.pick_wrist_dist = c["pick_wrist_dist"]
    self.place_release = c["place_release"]
    self.held_lever = c["held_lever"]
    self.farside_reach = c["farside_reach"]
    self.cos_cone = math.cos(math.radians(c["farside_cone_deg"]))
    self.cos_upright = math.cos(math.radians(c["tilt_max_deg"]))
    self.cos_tip = math.cos(math.radians(c["tip_deg"]))
    self.slide_gain_min = c["slide_gain_min"]
    self.touch = c["nested_touch_dist"]
    self.home_reward = c["home_reward"]

  # ------------------------------------------------------------------
  # Per-state conditions
  # ------------------------------------------------------------------

  def in_hand(self, s: Scene) -> torch.Tensor:
    return below(s.lever, self.held_lever, self.cfg.lever)

  def at_rest(self, s: Scene) -> torch.Tensor:
    return below(s.can_speed_xy, self.cfg.rest_speed, self.cfg.rest_speed_width)

  def in_band(self, s: Scene) -> torch.Tensor:
    z, w = s.can_pos[..., 2], self.cfg.height
    return above(z, self.band[0], w) * below(z, self.band[1], w)

  def on_shelf(self, s: Scene) -> torch.Tensor:
    w = self.cfg.footprint
    out = self.in_band(s)
    for axis in (0, 1):
      value = s.can_pos[..., axis]
      out = (
        out
        * above(value, self.shelf_lo[axis], w)
        * below(value, self.shelf_hi[axis], w)
      )
    return out

  def upright(self, up: torch.Tensor) -> torch.Tensor:
    return above(up, self.cos_upright, self.cfg.upright)

  def held_up(self, s: Scene) -> torch.Tensor:
    """The pick condition: the can above the pick height, in a closed gripper."""
    return (
      above(s.can_pos[..., 2], self.pick_z, self.cfg.height)
      * above(s.grip_cmd, self.pick_grip, self.cfg.grip)
      * below(s.wrist_dist, self.pick_wrist_dist, self.cfg.wrist)
    )

  def placed(self, s: Scene) -> torch.Tensor:
    """The release condition: standing on the shelf with the gripper open."""
    return (
      below(s.grip_cmd, self.place_release, self.cfg.grip)
      * self.on_shelf(s)
      * self.upright(s.can_up)
    )

  def behind(self, s: Scene) -> torch.Tensor:
    """The tool is off the can, within reach, on the side away from the goal."""
    return (
      (1 - self.in_hand(s))
      * below(s.lever, self.farside_reach, self.cfg.reach)
      * above(s.cos_far, self.cos_cone, self.cfg.cone)
    )

  def nested(self, s: Scene, flags: torch.Tensor) -> torch.Tensor:
    """The can rests upright on the shelf against the goal can, let go."""
    return (
      flags[..., PICKED]
      * flags[..., RELEASED]
      * below(s.dist, self.touch, self.cfg.touch)
      * self.upright(s.can_up)
      * self.upright(s.goal_up)
      * self.in_band(s)
      * (1 - self.in_hand(s))
      * self.at_rest(s)
    )

  # ------------------------------------------------------------------
  # Episode ends
  # ------------------------------------------------------------------

  def home(self, s: Scene, flags: torch.Tensor) -> torch.Tensor:
    """Degree to which this state ends the episode in success."""
    return self.nested(s, flags) * flags[..., SLIDE_EVENT]

  def tipped(self, s: Scene) -> torch.Tensor:
    """Degree to which this state ends the episode with a tipped, free can."""
    return below(s.can_up, self.cos_tip, self.cfg.tip) * (1 - self.in_hand(s))

  # ------------------------------------------------------------------
  # Latches
  # ------------------------------------------------------------------

  def advance(
    self,
    flags: torch.Tensor,
    slide_gain: torch.Tensor,
    slide_min: torch.Tensor,
    s: Scene,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latches after a step that led to ``s``. ``flags`` ``(..., 5)``;
    ``slide_gain`` and ``slide_min`` ``(...,)``."""
    picked, released, settled, farside, slide_event = flags.unbind(-1)
    picked = torch.maximum(picked, self.held_up(s))
    was_released = released
    released = torch.maximum(released, picked * self.placed(s))
    # Settling and sliding count from the frame after the release.
    settled = torch.maximum(settled, was_released * self.at_rest(s))
    behind = released * self.behind(s)
    farside = torch.maximum(farside, behind)
    progress = torch.relu(slide_min - s.dist)
    slide_gain = slide_gain + was_released * behind * settled * progress
    slide_min = torch.where(
      was_released > 0.5, torch.minimum(slide_min, s.dist), s.dist
    )
    slid = above(slide_gain, self.slide_gain_min, self.cfg.slide_gain)
    slide_event = torch.maximum(slide_event, released * farside * slid)
    flags = torch.stack([picked, released, settled, farside, slide_event], dim=-1)
    return flags, slide_gain, slide_min
