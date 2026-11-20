from __future__ import annotations

import torch

from supermimic.manip import layout as L
from supermimic.manip.fk import SerialChainFK
from supermimic.manip.representation import (
  PREDICTION_DIM,
  REP_DIM,
  REP_DIM_WITH_GOAL,
  REP_EEF_POS,
  REP_GRASP_POS,
  REP_GRASP_POS_IN_EEF,
  REP_GRIP_CMD,
  REP_OBJ_FROM_GOAL,
  REP_OBJ_POS,
)
from supermimic.manip.rotations import mat_vec, matrix_from_quat

from .conftest import random_states


def test_fk_of_a_planar_two_link_arm():
  # Two z-axis hinges, links of length 1 along x.
  chain = {
    "pre_pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).numpy(),
    "pre_rot": torch.eye(3).expand(2, 3, 3).numpy(),
    "axes": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]).numpy(),
    "tip_pos": torch.tensor([1.0, 0.0, 0.0]).numpy(),
    "tip_rot": torch.eye(3).numpy(),
  }
  fk = SerialChainFK(chain)
  q = torch.tensor([[0.3, 0.5], [-1.0, 2.0]])
  pos, rot = fk(q)
  a, b = q[:, 0], q[:, 0] + q[:, 1]
  expected = torch.stack([a.cos() + b.cos(), a.sin() + b.sin(), torch.zeros(2)], -1)
  assert torch.allclose(pos, expected, atol=1e-6)
  assert torch.allclose(rot[:, 0, 0], b.cos(), atol=1e-6)
  assert torch.allclose(rot[:, 1, 0], b.sin(), atol=1e-6)


def test_fk_broadcasts_over_leading_dims(representation):
  q = torch.randn(4, 5, L.NUM_ARM)
  pos, rot = representation.fk(q)
  flat_pos, flat_rot = representation.fk(q.reshape(20, L.NUM_ARM))
  assert pos.shape == (4, 5, 3) and rot.shape == (4, 5, 3, 3)
  assert torch.allclose(pos.reshape(20, 3), flat_pos, atol=1e-6)
  assert torch.allclose(rot.reshape(20, 3, 3), flat_rot, atol=1e-6)


def test_represent_layout_and_round_trip(representation):
  state = random_states(32)
  rep = representation.represent(state)
  aux = representation.extract_aux_state(state)
  assert rep.shape == (32, REP_DIM)
  assert torch.allclose(
    representation.reconstruct_full_state(aux, rep), state, atol=1e-6
  )
  # Cube and grip point are stored relative to the table-top origin.
  eef_pos, eef_rot = representation.fk(state[:, : L.NUM_ARM])
  assert torch.allclose(
    rep[:, REP_OBJ_POS], state[:, L.OBJ_POS] - representation.origin
  )
  assert torch.allclose(rep[:, REP_EEF_POS], eef_pos - representation.origin, atol=1e-6)
  # The eef-frame offset maps back to the world offset.
  offset = mat_vec(eef_rot, rep[:, REP_GRASP_POS_IN_EEF])
  assert torch.allclose(offset, state[:, L.OBJ_POS] - eef_pos, atol=1e-5)


def test_represent_keeps_window_shape(representation):
  state = random_states(24).reshape(4, 6, L.STATE_DIM)
  rep = representation.represent(state)
  assert rep.shape == (4, 6, REP_DIM)
  assert torch.allclose(rep[2, 3], representation.represent(state[2, 3]), atol=1e-6)


def test_prediction_target_reproduces_the_next_state(representation):
  state, next_state = random_states(64, seed=1), random_states(64, seed=2)
  action = torch.rand(64, L.ACTION_DIM) * 2 - 1
  # Make the next gripper command the one the rule produces.
  grip = (state[:, L.GRIP_CMD] + 0.25 * action[:, -1:]).clamp(-1, 1)
  next_state[:, L.GRIP_CMD] = grip
  representation.pred_scale.copy_(torch.rand(PREDICTION_DIM) + 0.5)

  target = representation.prediction_target(state, next_state)
  next_aux, next_rep = representation.apply_prediction(
    representation.extract_aux_state(state),
    representation.represent(state),
    target / representation.pred_scale,
    representation.dt,
    action,
  )
  rebuilt = representation.reconstruct_full_state(next_aux, next_rep)
  quat, rebuilt_quat = next_state[:, L.OBJ_QUAT], rebuilt[:, L.OBJ_QUAT]
  assert torch.allclose(
    matrix_from_quat(rebuilt_quat), matrix_from_quat(quat), atol=1e-4
  )
  keep = [i for i in range(L.STATE_DIM) if not L.OBJ_QUAT.start <= i < L.OBJ_QUAT.stop]
  assert torch.allclose(rebuilt[:, keep], next_state[:, keep], atol=1e-4)
  assert torch.allclose(next_rep, representation.represent(rebuilt), atol=1e-4)


def test_zero_prediction_coasts(representation):
  state = random_states(8, seed=3)
  action = torch.zeros(8, L.ACTION_DIM)
  next_aux, next_rep = representation.apply_prediction(
    representation.extract_aux_state(state),
    representation.represent(state),
    torch.zeros(8, PREDICTION_DIM),
    representation.dt,
    action,
  )
  nxt = representation.reconstruct_full_state(next_aux, next_rep)
  dt = representation.dt
  assert torch.allclose(nxt[:, L.DQ], state[:, L.DQ])
  assert torch.allclose(nxt[:, L.Q], state[:, L.Q] + state[:, L.DQ] * dt, atol=1e-6)
  assert torch.allclose(
    nxt[:, L.OBJ_POS], state[:, L.OBJ_POS] + state[:, L.OBJ_LIN_VEL] * dt, atol=1e-6
  )
  assert torch.allclose(nxt[:, L.GRIP_CMD], state[:, L.GRIP_CMD])


def test_gripper_command_follows_the_action_and_saturates(representation):
  state = random_states(3, seed=4)
  state[:, L.GRIP_CMD] = torch.tensor([[0.0], [0.9], [-0.95]])
  action = torch.zeros(3, L.ACTION_DIM)
  action[:, -1] = torch.tensor([0.5, 1.0, -1.0])
  _, next_rep = representation.apply_prediction(
    representation.extract_aux_state(state),
    representation.represent(state),
    torch.zeros(3, PREDICTION_DIM),
    representation.dt,
    action,
  )
  assert torch.allclose(
    next_rep[:, REP_GRIP_CMD], torch.tensor([[0.125], [1.0], [-1.0]])
  )


def test_fit_prediction_scale_normalizes_each_group(representation):
  state, next_state = random_states(256, seed=5), random_states(256, seed=6)
  report = representation.fit_prediction_scale(state, next_state)
  target = (
    representation.prediction_target(state, next_state) / representation.pred_scale
  )
  assert set(report) == {
    "d_arm_vel", "d_finger_vel", "d_obj_lin_vel", "d_obj_ang_vel",
    "r_arm_pos", "r_finger_pos", "r_obj_pos", "r_obj_rot",
  }  # fmt: skip
  # Arm-velocity group: unit RMS after scaling.
  assert torch.allclose(
    target[:, : L.NUM_ARM].pow(2).mean().sqrt(), torch.tensor(1.0), atol=1e-4
  )


def test_loss_blocks_match_the_weight_config(representation):
  state = random_states(4)
  blocks = representation.loss_blocks(
    representation.represent(state), representation.extract_aux_state(state)
  )
  assert list(blocks) == list(representation.LOSS_WEIGHT_CONFIG)
  assert blocks["root_ori"].shape == (4, 9)
  assert blocks["arm_pos"].shape == (4, L.NUM_ARM)
  assert blocks["finger_vel"].shape == (4, L.NUM_FINGER)


def test_goal_task_appends_the_grasp_point_and_the_goal_offset(goal_representation):
  rep = goal_representation
  state = random_states(8, seed=3)
  out = rep.represent(state)
  assert out.shape == (8, REP_DIM_WITH_GOAL) and rep.rep_state_dim == REP_DIM_WITH_GOAL
  handle = state[:, L.OBJ_POS] + mat_vec(
    matrix_from_quat(state[:, L.OBJ_QUAT]), rep.grasp_offset
  )
  assert torch.allclose(out[:, REP_GRASP_POS], handle - rep.origin, atol=1e-6)
  assert torch.allclose(
    out[:, REP_OBJ_FROM_GOAL], state[:, L.OBJ_POS] - rep.goal_pos, atol=1e-6
  )
  # The grip-frame offset points at the handle, not at the object's centre.
  eef_pos, eef_rot = rep.fk(state[:, L.Q][:, : L.NUM_ARM])
  assert torch.allclose(
    mat_vec(eef_rot, out[:, REP_GRASP_POS_IN_EEF]), handle - eef_pos, atol=1e-5
  )
  # Predicted states go through the same assembly.
  aux = rep.extract_aux_state(state)
  target = rep.prediction_target(state, random_states(8, seed=4))
  _, next_rep = rep.apply_prediction(
    aux, out, target / rep.pred_scale, rep.dt, torch.zeros(8, L.ACTION_DIM)
  )
  assert next_rep.shape == (8, REP_DIM_WITH_GOAL)
