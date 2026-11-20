from __future__ import annotations

import torch

from supermimic.manip import layout as L
from supermimic.manip.representation import (
  REP_DIM,
  REP_DIM_WITH_GOAL,
  REP_EEF_POS,
  REP_GRASP_POS,
  REP_OBJ_FROM_GOAL,
  REP_OBJ_POS,
  REP_Q,
)
from supermimic.manip.reward import (
  LiftRewardCfg,
  SquareRewardCfg,
  lift_reward,
  lift_reward_terms,
  square_reward,
  square_reward_terms,
)


def _rep(cube, eef, gap) -> torch.Tensor:
  rep = torch.zeros(1, REP_DIM)
  rep[0, REP_OBJ_POS] = torch.tensor(cube)
  rep[0, REP_EEF_POS] = torch.tensor(eef)
  rep[0, REP_Q.start + L.NUM_ARM] = gap / 2
  rep[0, REP_Q.start + L.NUM_ARM + 1] = -gap / 2
  return rep


def test_terms_at_the_task_stages():
  cfg = LiftRewardCfg()
  far = lift_reward_terms(_rep([0.0, 0.0, 0.021], [0.0, 0.0, 0.3], 0.08), cfg)
  open_at_cube = lift_reward_terms(
    _rep([0.0, 0.0, 0.021], [0.0, 0.0, 0.021], 0.08), cfg
  )
  grasped = lift_reward_terms(_rep([0.0, 0.0, 0.021], [0.0, 0.0, 0.021], 0.042), cfg)
  closed_empty = lift_reward_terms(_rep([0.0, 0.0, 0.021], [0.0, 0.0, 0.021], 0.0), cfg)
  lifted = lift_reward_terms(_rep([0.0, 0.0, 0.09], [0.0, 0.0, 0.09], 0.042), cfg)

  assert far["reach"] < 0.02 and far["grasp"] < 1e-6 and far["lift"] == 0
  assert open_at_cube["reach"] > 0.99 and open_at_cube["grasp"] < 0.01
  assert grasped["grasp"] > 0.7
  assert closed_empty["grasp"] < 0.01
  assert lifted["lift"] == 1.0
  # robosuite's success height (0.04) sits inside the ramp.
  at_success = lift_reward_terms(_rep([0.0, 0.0, 0.04], [0.0, 0.0, 0.04], 0.042), cfg)
  assert 0.2 < at_success["lift"] < 0.5


def test_reward_is_bounded_by_the_positive_weights_and_charges_action_rate():
  cfg = LiftRewardCfg()
  rep = _rep([0.0, 0.0, 0.09], [0.0, 0.0, 0.09], 0.042)
  still = torch.zeros(1, L.ACTION_DIM)
  best = lift_reward(rep, still, still, cfg)
  assert best <= cfg.max_reward and best > 0.95 * cfg.max_reward
  jerk = lift_reward(rep, torch.ones(1, L.ACTION_DIM), still, cfg)
  assert torch.allclose(
    best - jerk, torch.tensor(cfg.action_rate_weight * L.ACTION_DIM)
  )


def test_reward_has_a_gradient_toward_the_cube_at_zero_distance():
  cfg = LiftRewardCfg()
  rep = _rep([0.0, 0.0, 0.021], [0.0, 0.0, 0.021], 0.08).requires_grad_()
  lift_reward(rep, torch.zeros(1, 7), torch.zeros(1, 7), cfg).sum().backward()
  assert torch.isfinite(rep.grad).all()


PEG = (0.23, 0.1)


def _square_rep(nut, eef, gap) -> torch.Tensor:
  """Rep state of a level nut at ``nut`` (table frame) with its handle along x."""
  rep = torch.zeros(1, REP_DIM_WITH_GOAL)
  nut = torch.tensor(nut)
  rep[0, REP_OBJ_POS] = nut
  rep[0, REP_GRASP_POS] = nut + torch.tensor([0.054, 0.0, 0.0])
  rep[0, REP_OBJ_FROM_GOAL] = nut - torch.tensor([*PEG, 0.0])
  rep[0, REP_EEF_POS] = torch.tensor(eef)
  rep[0, REP_Q.start + L.NUM_ARM] = gap / 2
  rep[0, REP_Q.start + L.NUM_ARM + 1] = -gap / 2
  return rep


def _square_stages() -> dict[str, torch.Tensor]:
  on_table, at_handle = (-0.1, 0.2, 0.01), (-0.046, 0.2, 0.01)
  return {
    "start": _square_rep(on_table, (-0.1, 0.0, 0.2), 0.08),
    "reached": _square_rep(on_table, at_handle, 0.08),
    "grasped": _square_rep(on_table, at_handle, 0.03),
    "lifted": _square_rep((-0.1, 0.2, 0.2), (-0.046, 0.2, 0.2), 0.03),
    "over_peg": _square_rep((*PEG, 0.2), (PEG[0] + 0.054, PEG[1], 0.2), 0.03),
    "half_down": _square_rep((*PEG, 0.08), (PEG[0] + 0.054, PEG[1], 0.08), 0.03),
    "seated": _square_rep((*PEG, 0.01), (PEG[0] + 0.054, PEG[1], 0.01), 0.03),
  }


def test_square_reward_rises_through_the_task_stages():
  cfg = SquareRewardCfg()
  zero = torch.zeros(1, L.ACTION_DIM)
  rewards = [
    float(square_reward(rep, zero, zero, cfg)) for rep in _square_stages().values()
  ]
  assert all(b > a + 0.1 for a, b in zip(rewards, rewards[1:], strict=False)), rewards
  assert rewards[-1] > 0.9 * cfg.max_reward


def test_square_terms_at_the_task_stages():
  cfg = SquareRewardCfg()
  stages = {k: square_reward_terms(v, cfg) for k, v in _square_stages().items()}
  assert stages["reached"]["reach"] > 0.99 and stages["reached"]["grasp"] < 0.01
  assert stages["grasped"]["grasp"] > 0.85 and stages["grasped"]["lift"] < 0.05
  assert stages["lifted"]["lift"] > 0.85 and stages["lifted"]["hover"] < 0.05
  assert stages["over_peg"]["hover"] > 0.85 and stages["over_peg"]["seat"] < 0.01
  # Going down the peg costs no lift reward and earns the seat term.
  assert stages["half_down"]["lift"] > 0.85
  assert 0.4 < stages["half_down"]["seat"] < 0.6
  assert stages["seated"]["seat"] > 0.9


def test_square_seat_needs_the_peg_axis_and_a_released_centre():
  cfg = SquareRewardCfg()
  beside = _square_rep(
    (PEG[0] + 0.04, PEG[1], 0.01), (PEG[0] + 0.094, PEG[1], 0.01), 0.03
  )
  assert square_reward_terms(beside, cfg)["seat"] < 0.01
  # Gripper over the nut's centre: robosuite would not count this as placed.
  covered = _square_rep((*PEG, 0.01), (*PEG, 0.03), 0.08)
  assert square_reward_terms(covered, cfg)["seat"] < 0.02
  # Let go and backed away: still seated, and the height terms are kept.
  released = square_reward_terms(_square_rep((*PEG, 0.01), (0.0, 0.0, 0.3), 0.08), cfg)
  assert (
    released["seat"] > 0.98 and released["lift"] > 0.98 and released["grasp"] < 0.01
  )


def test_square_reward_has_gradients_toward_each_next_stage():
  cfg = SquareRewardCfg()
  zero = torch.zeros(1, L.ACTION_DIM)
  stages = _square_stages()
  # Carrying: moving the nut toward the peg axis raises the reward.
  rep = stages["lifted"].clone().requires_grad_(True)
  square_reward(rep, zero, zero, cfg).sum().backward()
  toward_peg = torch.tensor([*PEG, 0.0])[:2] - stages["lifted"][0, REP_OBJ_POS][:2]
  assert (rep.grad[0, REP_OBJ_FROM_GOAL][:2] * toward_peg).sum() > 0
  # On the peg: moving the nut down raises the reward.
  rep = stages["half_down"].clone().requires_grad_(True)
  square_reward(rep, zero, zero, cfg).sum().backward()
  assert rep.grad[0, REP_OBJ_POS][2] < 0
