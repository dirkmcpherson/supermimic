"""The smooth stage rules follow the environment's ladder."""

from __future__ import annotations

import torch

from supermimic.restock.predicates import (
  FARSIDE,
  PICKED,
  RELEASED,
  SETTLED,
  SLIDE_EVENT,
  Scene,
  StageRules,
)

GOAL = (0.672, -0.221)
SHELF_Z = 0.17 + 0.0505  # a can standing on the shelf


def scene(**kw) -> Scene:
  """A can at rest on the shelf, 20 cm from the goal, arm far away, gripper open."""
  base = {
    "can_pos": (0.672, -0.021, SHELF_Z),
    "can_speed_xy": 0.0,
    "can_up": 1.0,
    "goal_up": 1.0,
    "lever": 0.3,
    "dist": 0.2,
    "cos_far": -1.0,
    "wrist_dist": 0.4,
    "grip_cmd": 0.0,
  }
  base.update(kw)
  return Scene(
    **{
      k: torch.tensor([v], dtype=torch.float32).squeeze(0)
      if k != "can_pos"
      else torch.tensor(v)
      for k, v in base.items()
    }
  )


def advance(rules, flags, gain, slide_min, s):
  f, g, m = rules.advance(
    torch.tensor(flags, dtype=torch.float32),
    torch.tensor(gain),
    torch.tensor(slide_min),
    s,
  )
  return f, float(g), float(m)


def test_the_ladder_advances_in_order(constants):
  rules = StageRules(constants)
  held = scene(
    can_pos=(0.5, 0.0, 0.25),
    lever=0.015,
    wrist_dist=0.146,
    grip_cmd=0.8,
    can_speed_xy=0.1,
  )
  flags, gain, slide_min = advance(rules, [0.0] * 5, 0.0, 0.2, held)
  assert flags[PICKED] > 0.95 and flags[RELEASED] < 0.05

  # Open over the shelf: released; the slide minimum starts at this distance.
  set_down = scene(lever=0.02, wrist_dist=0.15, dist=0.2)
  flags, gain, slide_min = advance(rules, flags.tolist(), gain, slide_min, set_down)
  assert flags[RELEASED] > 0.9 and flags[SETTLED] < 0.05 and abs(slide_min - 0.2) < 1e-6

  # At rest with the hand withdrawn: settled, but the tool is not behind the can.
  flags, gain, slide_min = advance(rules, flags.tolist(), gain, slide_min, scene())
  assert flags[SETTLED] > 0.85 and flags[FARSIDE] < 0.05 and gain == 0.0

  # Tool behind the can, can pushed 3 cm goalward.
  push = scene(lever=0.06, cos_far=0.95, dist=0.17, can_speed_xy=0.05)
  flags, gain, slide_min = advance(rules, flags.tolist(), gain, slide_min, push)
  assert flags[FARSIDE] > 0.9 and 0.02 < gain <= 0.03 and abs(slide_min - 0.17) < 1e-6
  assert flags[SLIDE_EVENT] > 0.8

  arrived = scene(lever=0.06, cos_far=0.95, dist=0.07)
  assert rules.home(arrived, flags) > 0.6
  assert (
    rules.home(scene(lever=0.06, cos_far=0.95, dist=0.07, can_speed_xy=0.05), flags)
    < 0.05
  )
  assert rules.home(scene(lever=0.06, cos_far=0.95, dist=0.12), flags) < 0.05


def test_shortcuts_do_not_pay(constants):
  rules = StageRules(constants)
  resting_at_goal = scene(dist=0.07)
  # Never picked: a can that starts on the shelf is not a release.
  flags, _, _ = advance(rules, [0.0] * 5, 0.0, 0.07, resting_at_goal)
  assert flags[RELEASED] < 0.05 and rules.home(resting_at_goal, flags) < 0.01
  # Carried to the goal and dropped there: released and nested, but no slide.
  flags = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
  assert (
    rules.nested(resting_at_goal, flags) > 0.8
    and rules.home(resting_at_goal, flags) < 0.01
  )
  # Progress made while the can is carried lowers the bar and earns nothing.
  carried = scene(lever=0.015, dist=0.1, can_speed_xy=0.1, grip_cmd=0.8)
  _, gain, slide_min = advance(rules, [1.0, 1.0, 1.0, 1.0, 0.0], 0.0, 0.2, carried)
  assert gain < 1e-3 and abs(slide_min - 0.1) < 1e-6


def test_tip_ends_the_episode_only_for_a_free_can(constants):
  rules = StageRules(constants)
  assert rules.tipped(scene(can_up=0.2)) > 0.95
  assert rules.tipped(scene(can_up=0.2, lever=0.015)) < 0.05
  assert rules.tipped(scene(can_up=0.9)) < 0.01


def test_home_has_a_gradient_towards_the_goal(constants):
  rules = StageRules(constants)
  dist = torch.tensor(0.09, requires_grad=True)
  s = scene(lever=0.06, cos_far=0.95)
  s.dist = dist
  rules.home(s, torch.ones(5)).backward()
  assert dist.grad < 0
