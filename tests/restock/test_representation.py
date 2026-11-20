"""The restock representation: layout, integration, kinematics."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.manip.fk import SerialChainFK
from supermimic.restock import layout as L
from supermimic.restock.kinematics import chain_fk, chain_from_link_poses
from supermimic.restock.models import RestockWorldModelLearnerCfg, prediction_floors
from supermimic.restock.representation import (
  PREDICTION_DIM,
  PREDICTION_GROUPS,
  REP_DIM,
  REP_SCALARS,
  REP_TOOL_POS,
)

from .conftest import make_chain, random_states


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
  w = np.sqrt(max(0.0, 1 + np.trace(rot))) / 2
  if w < 1e-6:
    raise ValueError("half-turn sample")
  return np.array(
    [w, (rot[2, 1] - rot[1, 2]) / (4 * w), (rot[0, 2] - rot[2, 0]) / (4 * w), (rot[1, 0] - rot[0, 1]) / (4 * w)]
  )  # fmt: skip


def test_layout_is_contiguous():
  slices = [
    L.Q, L.DQ, L.CAN_POS, L.CAN_QUAT, L.CAN_LIN_VEL, L.CAN_ANG_VEL, L.GOAL_POS,
    L.GOAL_QUAT, L.GOAL_LIN_VEL, L.GOAL_ANG_VEL, L.TARGET, L.GRIP_CMD, L.FLAGS,
    L.SLIDE_GAIN, L.SLIDE_MIN,
  ]  # fmt: skip
  assert slices[0].start == 0 and slices[-1].stop == L.STATE_DIM
  assert all(a.stop == b.start for a, b in zip(slices, slices[1:], strict=False))
  assert L.FLAGS.stop - L.FLAGS.start == len(L.FLAG_NAMES)


def test_chain_is_recovered_from_link_poses():
  """The fit used on the simulator's forward kinematics, on a known chain."""
  chain = make_chain(3)
  rng = np.random.default_rng(0)
  qs = rng.uniform(-2.5, 2.5, (40, L.NUM_ARM))
  pos = np.zeros((40, L.NUM_ARM + 1, 3))
  quat = np.zeros((40, L.NUM_ARM + 1, 4))
  quat[:, 0, 0] = 1.0
  for s, q in enumerate(qs):
    for j in range(L.NUM_ARM):
      sub = {k: v[: j + 1] for k, v in chain.items() if k not in ("tip_pos", "tip_rot")}
      sub |= {"tip_pos": np.zeros(3), "tip_rot": np.eye(3)}
      p, r = chain_fk(sub, q)
      pos[s, j + 1], quat[s, j + 1] = p, _mat_to_quat(r)
  fitted = chain_from_link_poses(
    qs[:30], pos[:30], quat[:30], list(range(1, L.NUM_ARM + 1)), chain["tip_pos"]
  )
  torch_fk = SerialChainFK(fitted)
  for q in qs[30:]:
    want, want_rot = chain_fk(chain, q)
    got, got_rot = chain_fk(fitted, q)
    assert np.abs(got - want).max() < 1e-9 and np.abs(got_rot - want_rot).max() < 1e-9
    tool, _ = torch_fk(torch.as_tensor(q, dtype=torch.float32)[None])
    assert np.abs(tool[0].numpy() - want).max() < 1e-5


def test_rep_state_round_trips(representation):
  state = random_states(64)
  rep = representation.represent(state)
  aux = representation.extract_aux_state(state)
  assert rep.shape == (64, REP_DIM) and aux.shape == (64, 14)
  assert torch.allclose(
    representation.reconstruct_full_state(aux, rep), state, atol=1e-6
  )
  lever = rep[:, REP_SCALARS][:, 0]
  can_xy = state[:, L.CAN_POS][:, :2]
  assert torch.allclose(
    lever, (rep[:, REP_TOOL_POS][:, :2] - can_xy).norm(dim=-1), atol=1e-6
  )


def test_prediction_target_inverts_the_integration(representation):
  state, next_state = random_states(128, 1), random_states(128, 2)
  action = torch.rand(128, L.ACTION_DIM) * 2 - 1
  next_state[:, L.GRIP_CMD] = (action[:, -1:] + 1) / 2
  floors = prediction_floors(RestockWorldModelLearnerCfg())
  scales = representation.fit_prediction_scale(state, next_state, action, floors)
  assert set(scales) == {name for name, _ in PREDICTION_GROUPS}
  target = representation.prediction_target(state, next_state, action)
  assert target.shape == (128, PREDICTION_DIM)
  next_aux, next_rep = representation.apply_prediction(
    representation.extract_aux_state(state),
    representation.represent(state),
    target / representation.pred_scale,
    representation.dt,
    action,
  )
  got = representation.reconstruct_full_state(next_aux, next_rep)
  # Quaternions may come back with the opposite sign; the loss blocks below
  # compare the rotations.
  for cols in (L.Q, L.DQ, L.CAN_POS, L.CAN_LIN_VEL, L.CAN_ANG_VEL, L.GOAL_POS, L.GOAL_LIN_VEL, L.GOAL_ANG_VEL, L.TARGET, L.GRIP_CMD):  # fmt: skip
    assert torch.allclose(got[:, cols], next_state[:, cols], atol=2e-4), cols
  blocks = representation.loss_blocks(next_rep, next_aux)
  want = representation.loss_blocks(
    representation.represent(next_state), representation.extract_aux_state(next_state)
  )
  assert set(blocks) == set(representation.LOSS_WEIGHT_CONFIG)
  for name in blocks:
    assert torch.allclose(blocks[name], want[name], atol=2e-4), name


def test_target_follows_the_action_inside_the_joint_limits(representation):
  state = random_states(8)
  state[:, L.TARGET] = 0.0
  action = torch.zeros(8, L.ACTION_DIM)
  action[:, 0], action[:, 1] = 1.0, -0.5
  zero = torch.zeros(8, PREDICTION_DIM)
  _, next_rep = representation.apply_prediction(
    representation.extract_aux_state(state),
    representation.represent(state),
    zero,
    representation.dt,
    action,
  )
  full = representation.reconstruct_full_state(torch.zeros(8, 14), next_rep)
  assert torch.allclose(full[:, L.TARGET][:, 0], torch.full((8,), 0.1), atol=1e-6)
  assert torch.allclose(full[:, L.TARGET][:, 1], torch.full((8,), -0.05), atol=1e-6)
  assert torch.allclose(full[:, L.GRIP_CMD], torch.full((8, 1), 0.5))
