"""State representation of a fixed-base arm and one free rigid object.

Plays the role of :class:`supermimic.representations.StateRepresentation` for
the world-model and policy learners. The free object takes the place of the
humanoid's floating base: its world pose is the ``aux_state`` and is scored by
the ``root_pos`` / ``root_ori`` loss groups.

Full state: see :mod:`supermimic.manip.layout` (32)
Rep state:  ``[q(9), dq(9), obj_pos(3), obj_lin_vel(3), obj_ang_vel(3),
            grip_cmd(1) | obj_rot6d(6), eef_pos(3), eef_rot6d(6),
            grasp_pos_in_eef(3), obj_rot6d_in_eef(6)]`` (52), and for a task
            with a goal ``[grasp_pos(3), obj_pos_from_goal(3)]`` (58). The
            part after the bar is derived from the part before it and the
            object quaternion. The grasp point is a fixed point of the object
            (its centre unless the task's constants give a ``grasp_offset``).
Aux state:  ``[obj_pos(3), obj_quat(4)]``, world frame
Prediction: ``[d_dq(9), d_v(3), d_w(3), r_q(9), r_p(3), r_theta(3)]`` (30):
            velocity deltas integrated semi-implicitly as in the humanoid
            representations, plus pose residuals on top of that integration. A
            20 Hz control step spans 25 physics substeps in which contacts make
            and break, so pose changes are not a function of the velocity
            change alone.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from supermimic.manip import layout as L
from supermimic.manip.fk import SerialChainFK
from supermimic.manip.rotations import (
  mat_t_vec,
  mat_vec,
  matrix_from_quat,
  quat_integrate,
  rot6d_from_matrix,
  rotvec_between,
  small_matmul,
)

# Rep-state columns.
REP_Q = slice(0, 9)
REP_DQ = slice(9, 18)
REP_OBJ_POS = slice(18, 21)
REP_OBJ_LIN_VEL = slice(21, 24)
REP_OBJ_ANG_VEL = slice(24, 27)
REP_GRIP_CMD = slice(27, 28)
REP_OBJ_ROT6D = slice(28, 34)
REP_EEF_POS = slice(34, 37)
REP_EEF_ROT6D = slice(37, 43)
REP_GRASP_POS_IN_EEF = slice(43, 46)
REP_OBJ_ROT6D_IN_EEF = slice(46, 52)
REP_DIM = 52
# Only for tasks with a goal.
REP_GRASP_POS = slice(52, 55)
REP_OBJ_FROM_GOAL = slice(55, 58)
REP_DIM_WITH_GOAL = 58

# Prediction columns: (name, width), in order.
PREDICTION_GROUPS = (
  ("d_arm_vel", L.NUM_ARM),
  ("d_finger_vel", L.NUM_FINGER),
  ("d_obj_lin_vel", 3),
  ("d_obj_ang_vel", 3),
  ("r_arm_pos", L.NUM_ARM),
  ("r_finger_pos", L.NUM_FINGER),
  ("r_obj_pos", 3),
  ("r_obj_rot", 3),
)
PREDICTION_DIM = sum(width for _, width in PREDICTION_GROUPS)


class ArmObjectRepresentation(nn.Module):
  """Duck-types ``StateRepresentation`` for the learners."""

  LOSS_WEIGHT_CONFIG: dict[str, str] = {
    "root_pos": "nominal_error_root_pose_pos",
    "root_ori": "nominal_error_root_pose_ori",
    "arm_pos": "nominal_error_arm_pos",
    "arm_vel": "nominal_error_arm_vel",
    "finger_pos": "nominal_error_finger_pos",
    "finger_vel": "nominal_error_finger_vel",
    "obj_vel": "nominal_error_obj_vel",
    "obj_avel": "nominal_error_obj_avel",
  }

  full_state_dim = L.STATE_DIM
  prediction_dim = PREDICTION_DIM
  num_joints = L.NUM_JOINTS
  actor_obs_mode = "rep_state"

  def __init__(self, constants: dict[str, np.ndarray]):
    super().__init__()
    self.fk = SerialChainFK(
      {k[3:]: v for k, v in constants.items() if k.startswith("fk_")}
    )
    self.dt = float(constants["dt"])
    substeps = int(constants["substeps"])
    # Semi-implicit composition of the substeps under constant acceleration.
    self.substep_coeff = (substeps + 1) / (2 * substeps)
    self.grip_cmd_step = float(constants["grip_cmd_step"])
    origin = torch.tensor([0.0, 0.0, float(constants["table_z"])])
    self.register_buffer("origin", origin)
    self.register_buffer(
      "grasp_offset",
      torch.as_tensor(
        np.asarray(constants.get("grasp_offset", np.zeros(3))), dtype=torch.float32
      ),
    )
    self.has_goal = "goal_pos" in constants
    self.rep_state_dim = REP_DIM_WITH_GOAL if self.has_goal else REP_DIM
    if self.has_goal:
      self.register_buffer(
        "goal_pos", torch.as_tensor(constants["goal_pos"], dtype=torch.float32)
      )
    self.register_buffer("pred_scale", torch.ones(PREDICTION_DIM))

  # ------------------------------------------------------------------
  # State <-> representation
  # ------------------------------------------------------------------

  def _assemble(
    self,
    q: torch.Tensor,
    dq: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_quat: torch.Tensor,
    obj_lin_vel: torch.Tensor,
    obj_ang_vel: torch.Tensor,
    grip_cmd: torch.Tensor,
  ) -> torch.Tensor:
    """The one place the rep state is built, for real and predicted states."""
    eef_pos, eef_rot = self.fk(q[..., : L.NUM_ARM])
    obj_rot = matrix_from_quat(obj_quat)
    grasp_pos = obj_pos + mat_vec(obj_rot, self.grasp_offset)
    parts = [
      q,
      dq,
      obj_pos - self.origin,
      obj_lin_vel,
      obj_ang_vel,
      grip_cmd,
      rot6d_from_matrix(obj_rot),
      eef_pos - self.origin,
      rot6d_from_matrix(eef_rot),
      mat_t_vec(eef_rot, grasp_pos - eef_pos),
      rot6d_from_matrix(small_matmul(eef_rot.transpose(-1, -2), obj_rot)),
    ]
    if self.has_goal:
      parts += [grasp_pos - self.origin, obj_pos - self.goal_pos]
    return torch.cat(parts, dim=-1)

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    return self._assemble(
      full_state[..., L.Q],
      full_state[..., L.DQ],
      full_state[..., L.OBJ_POS],
      full_state[..., L.OBJ_QUAT],
      full_state[..., L.OBJ_LIN_VEL],
      full_state[..., L.OBJ_ANG_VEL],
      full_state[..., L.GRIP_CMD],
    )

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    return torch.cat([full_state[..., L.OBJ_POS], full_state[..., L.OBJ_QUAT]], -1)

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    return torch.cat(
      [
        rep_state[..., REP_Q],
        rep_state[..., REP_DQ],
        aux_state,
        rep_state[..., REP_OBJ_LIN_VEL],
        rep_state[..., REP_OBJ_ANG_VEL],
        rep_state[..., REP_GRIP_CMD],
      ],
      dim=-1,
    )

  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    return torch.cat([rep_state[..., REP_Q], rep_state[..., REP_DQ]], dim=-1)

  # ------------------------------------------------------------------
  # Dynamics integration
  # ------------------------------------------------------------------

  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
    action: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate one control step. ``action`` is the executed command (already
    clipped to ``[-1, 1]``); its last entry advances the gripper command by rule.
    """
    c = self.substep_coeff
    q = curr_rep_state[..., REP_Q]
    dq = curr_rep_state[..., REP_DQ]
    lin_vel = curr_rep_state[..., REP_OBJ_LIN_VEL]
    ang_vel = curr_rep_state[..., REP_OBJ_ANG_VEL]
    grip_cmd = curr_rep_state[..., REP_GRIP_CMD]
    obj_pos, obj_quat = aux_state[..., :3], aux_state[..., 3:7]

    d_dq, d_v, d_w, r_q, r_p, r_theta = (prediction * self.pred_scale).split(
      [L.NUM_JOINTS, 3, 3, L.NUM_JOINTS, 3, 3], dim=-1
    )

    next_dq = dq + d_dq
    next_q = q + (dq + c * d_dq) * dt + r_q
    next_lin_vel = lin_vel + d_v
    next_pos = obj_pos + (lin_vel + c * d_v) * dt + r_p
    next_ang_vel = ang_vel + d_w
    next_quat = quat_integrate(
      quat_integrate(obj_quat, ang_vel + c * d_w, dt), r_theta, 1.0
    )
    next_grip = (grip_cmd + self.grip_cmd_step * action[..., -1:]).clamp(-1.0, 1.0)

    next_aux = torch.cat([next_pos, next_quat], dim=-1)
    next_rep = self._assemble(
      next_q, next_dq, next_pos, next_quat, next_lin_vel, next_ang_vel, next_grip
    )
    return next_aux, next_rep

  def prediction_target(
    self, full_state: torch.Tensor, next_full_state: torch.Tensor
  ) -> torch.Tensor:
    """The unscaled prediction that takes ``full_state`` to ``next_full_state``:
    ``apply_prediction(target / pred_scale)`` reproduces the next state."""
    c, dt = self.substep_coeff, self.dt
    d_dq = next_full_state[..., L.DQ] - full_state[..., L.DQ]
    d_v = next_full_state[..., L.OBJ_LIN_VEL] - full_state[..., L.OBJ_LIN_VEL]
    d_w = next_full_state[..., L.OBJ_ANG_VEL] - full_state[..., L.OBJ_ANG_VEL]
    r_q = (
      next_full_state[..., L.Q]
      - full_state[..., L.Q]
      - (full_state[..., L.DQ] + c * d_dq) * dt
    )
    r_p = (
      next_full_state[..., L.OBJ_POS]
      - full_state[..., L.OBJ_POS]
      - (full_state[..., L.OBJ_LIN_VEL] + c * d_v) * dt
    )
    coasted = quat_integrate(
      full_state[..., L.OBJ_QUAT], full_state[..., L.OBJ_ANG_VEL] + c * d_w, dt
    )
    r_theta = rotvec_between(coasted, next_full_state[..., L.OBJ_QUAT])
    return torch.cat([d_dq, d_v, d_w, r_q, r_p, r_theta], dim=-1)

  @torch.no_grad()
  def fit_prediction_scale(
    self,
    full_state: torch.Tensor,
    next_full_state: torch.Tensor,
    floors: dict[str, float] | None = None,
  ) -> dict[str, float]:
    """Set ``pred_scale`` to each prediction group's RMS over real transitions,
    so the network's outputs are O(1). ``floors`` gives per-group lower bounds,
    for data in which a group hardly varies (an object nothing has touched
    yet). Returns the per-group scales."""
    floors = floors or {}
    target = self.prediction_target(full_state, next_full_state)
    target = target.reshape(-1, PREDICTION_DIM)
    scales, report, start = [], {}, 0
    for name, width in PREDICTION_GROUPS:
      rms = target[:, start : start + width].pow(2).mean().sqrt()
      rms = rms.clamp(min=max(1e-6, floors.get(name, 0.0)))
      scales.append(rms.expand(width))
      report[name] = float(rms)
      start += width
    self.pred_scale.copy_(torch.cat(scales))
    return report

  # ------------------------------------------------------------------
  # World-model loss
  # ------------------------------------------------------------------

  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """Channel groups the state loss scores, keyed as ``LOSS_WEIGHT_CONFIG``.

    The object rotation is scaled by sqrt(1/2) so a plain squared difference is
    ``theta^2 + O(theta^4)`` (see ``representations._root_loss_blocks``).
    """
    q, dq = rep_state[..., REP_Q], rep_state[..., REP_DQ]
    rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    return {
      "root_pos": aux_state[..., :3],
      "root_ori": rot * math.sqrt(0.5),
      "arm_pos": q[..., : L.NUM_ARM],
      "arm_vel": dq[..., : L.NUM_ARM],
      "finger_pos": q[..., L.NUM_ARM :],
      "finger_vel": dq[..., L.NUM_ARM :],
      "obj_vel": rep_state[..., REP_OBJ_LIN_VEL],
      "obj_avel": rep_state[..., REP_OBJ_ANG_VEL],
    }
