"""State representation of the restock task: a six-joint arm with a four-joint
gripper, the can to move and the goal can, both free bodies.

Plays the role of :class:`supermimic.representations.StateRepresentation` for
the world-model and policy learners, as
:class:`supermimic.manip.representation.ArmObjectRepresentation` does for
robosuite.

Full state: see :mod:`supermimic.restock.layout` (60)
Rep state:  ``[q(10), dq(10), can_pos(3), can_lin_vel(3), can_ang_vel(3),
            goal_pos(3), goal_lin_vel(3), goal_ang_vel(3), target_error(6),
            grip_cmd(1), flags(5), slide_gain(1), slide_min(1) |
            can_rot6d(6), goal_rot6d(6), tool_pos(3), tool_rot6d(6),
            can_in_tool(3), can_from_goal(3), scalars(6)]`` (85). The part
            after the bar is derived from the part before it and the two
            quaternions. ``target_error`` is the joint target minus the joint
            angle, the quantity the arm's position controller acts on.
            ``scalars`` are the six numbers the stage rules read
            (``SCALAR_NAMES``).
Aux state:  ``[can_pos(3), can_quat(4), goal_pos(3), goal_quat(4)]``
Prediction: velocity deltas of the joints and both bodies, integrated
            semi-implicitly, plus pose residuals on top of that integration and
            a residual on the joint target (50). The target advances by rule
            (``clip(target + step * action)``) except when the environment's
            leash to the measured joint angles cuts it short, which the
            residual learns. The gripper command is the action's last entry;
            the flags and the slide bookkeeping advance by
            :class:`supermimic.restock.predicates.StageRules`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from supermimic.manip.fk import SerialChainFK
from supermimic.manip.rotations import (
  mat_t_vec,
  mat_vec,
  matrix_from_quat,
  quat_integrate,
  rot6d_from_matrix,
  rotvec_between,
)
from supermimic.restock import layout as L
from supermimic.restock.predicates import Scene, SoftCfg, StageRules

REP_Q = slice(0, 10)
REP_DQ = slice(10, 20)
REP_CAN_POS = slice(20, 23)
REP_CAN_LIN_VEL = slice(23, 26)
REP_CAN_ANG_VEL = slice(26, 29)
REP_GOAL_POS = slice(29, 32)
REP_GOAL_LIN_VEL = slice(32, 35)
REP_GOAL_ANG_VEL = slice(35, 38)
REP_TARGET_ERROR = slice(38, 44)
REP_GRIP_CMD = slice(44, 45)
REP_FLAGS = slice(45, 50)
REP_SLIDE_GAIN = slice(50, 51)
REP_SLIDE_MIN = slice(51, 52)
REP_CAN_ROT6D = slice(52, 58)
REP_GOAL_ROT6D = slice(58, 64)
REP_TOOL_POS = slice(64, 67)
REP_TOOL_ROT6D = slice(67, 73)
REP_CAN_IN_TOOL = slice(73, 76)
REP_CAN_FROM_GOAL = slice(76, 79)
REP_SCALARS = slice(79, 85)
REP_DIM = 85

SCALAR_NAMES = ("lever", "dist", "cos_far", "can_up", "goal_up", "wrist_dist")

# Prediction columns: (name, width), in order.
PREDICTION_GROUPS = (
  ("d_arm_vel", L.NUM_ARM),
  ("d_finger_vel", L.NUM_FINGER),
  ("d_can_lin_vel", 3),
  ("d_can_ang_vel", 3),
  ("d_goal_lin_vel", 3),
  ("d_goal_ang_vel", 3),
  ("r_arm_pos", L.NUM_ARM),
  ("r_finger_pos", L.NUM_FINGER),
  ("r_can_pos", 3),
  ("r_can_rot", 3),
  ("r_goal_pos", 3),
  ("r_goal_rot", 3),
  ("r_target", L.NUM_ARM),
)
PREDICTION_DIM = sum(width for _, width in PREDICTION_GROUPS)


class RestockRepresentation(nn.Module):
  """Duck-types ``StateRepresentation`` for the learners."""

  LOSS_WEIGHT_CONFIG: dict[str, str] = {
    "root_pos": "nominal_error_root_pose_pos",
    "root_ori": "nominal_error_root_pose_ori",
    "goal_pos": "nominal_error_goal_pos",
    "goal_ori": "nominal_error_goal_ori",
    "arm_pos": "nominal_error_arm_pos",
    "arm_vel": "nominal_error_arm_vel",
    "finger_pos": "nominal_error_finger_pos",
    "finger_vel": "nominal_error_finger_vel",
    "can_vel": "nominal_error_can_vel",
    "can_avel": "nominal_error_can_avel",
    "goal_vel": "nominal_error_goal_vel",
    "goal_avel": "nominal_error_goal_avel",
    "target": "nominal_error_target",
  }

  full_state_dim = L.STATE_DIM
  rep_state_dim = REP_DIM
  prediction_dim = PREDICTION_DIM
  num_joints = L.NUM_JOINTS
  actor_obs_mode = "rep_state"

  def __init__(self, constants: dict[str, np.ndarray], soft: SoftCfg | None = None):
    super().__init__()
    chain = {k[3:]: v for k, v in constants.items() if k.startswith("fk_")}
    self.fk = SerialChainFK(chain)
    self.rules = StageRules(constants, soft)
    self.dt = float(constants["dt"])
    substeps = int(constants["substeps"])
    # Semi-implicit composition of the substeps under constant acceleration.
    self.substep_coeff = (substeps + 1) / (2 * substeps)
    self.target_step = float(constants["target_step"])

    def buf(name: str, value: np.ndarray) -> None:
      self.register_buffer(
        name, torch.as_tensor(np.asarray(value), dtype=torch.float32)
      )

    buf("arm_lo", constants["arm_lo"])
    buf("arm_hi", constants["arm_hi"])
    # The wrist link's origin in the tool frame (the tool frame is the wrist
    # frame moved to the tool point).
    buf("wrist_in_tool", -np.asarray(chain["tip_pos"]))
    self.register_buffer("pred_scale", torch.ones(PREDICTION_DIM))

  # ------------------------------------------------------------------
  # State <-> representation
  # ------------------------------------------------------------------

  def _assemble(
    self,
    q: torch.Tensor,
    dq: torch.Tensor,
    can: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    goal: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_error: torch.Tensor,
    grip_cmd: torch.Tensor,
    flags: torch.Tensor,
    slide_gain: torch.Tensor,
    slide_min: torch.Tensor,
  ) -> torch.Tensor:
    """The one place the rep state is built, for real and predicted states.
    ``can`` and ``goal`` are ``(pos, quat, lin_vel, ang_vel)``."""
    can_pos, can_quat, can_lin, can_ang = can
    goal_pos, goal_quat, goal_lin, goal_ang = goal
    tool_pos, tool_rot = self.fk(q[..., : L.NUM_ARM])
    can_rot = matrix_from_quat(can_quat)
    goal_rot = matrix_from_quat(goal_quat)

    to_tool = tool_pos[..., :2] - can_pos[..., :2]
    to_goal = goal_pos[..., :2] - can_pos[..., :2]
    lever = to_tool.norm(dim=-1, keepdim=True)
    dist = to_goal.norm(dim=-1, keepdim=True)
    cos_far = -(to_tool * to_goal).sum(-1, keepdim=True) / (lever * dist + 1e-9)
    wrist_pos = tool_pos + mat_vec(tool_rot, self.wrist_in_tool)
    wrist_dist = (wrist_pos - can_pos).norm(dim=-1, keepdim=True)
    scalars = [
      lever,
      dist,
      cos_far,
      can_rot[..., 2, 2:3],
      goal_rot[..., 2, 2:3],
      wrist_dist,
    ]
    return torch.cat(
      [
        q,
        dq,
        can_pos,
        can_lin,
        can_ang,
        goal_pos,
        goal_lin,
        goal_ang,
        target_error,
        grip_cmd,
        flags,
        slide_gain,
        slide_min,
        rot6d_from_matrix(can_rot),
        rot6d_from_matrix(goal_rot),
        tool_pos,
        rot6d_from_matrix(tool_rot),
        mat_t_vec(tool_rot, can_pos - tool_pos),
        can_pos - goal_pos,
        *scalars,
      ],
      dim=-1,
    )

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    s = full_state
    q = s[..., L.Q]
    return self._assemble(
      q,
      s[..., L.DQ],
      (
        s[..., L.CAN_POS],
        s[..., L.CAN_QUAT],
        s[..., L.CAN_LIN_VEL],
        s[..., L.CAN_ANG_VEL],
      ),
      (
        s[..., L.GOAL_POS],
        s[..., L.GOAL_QUAT],
        s[..., L.GOAL_LIN_VEL],
        s[..., L.GOAL_ANG_VEL],
      ),
      s[..., L.TARGET] - q[..., : L.NUM_ARM],
      s[..., L.GRIP_CMD],
      s[..., L.FLAGS],
      s[..., L.SLIDE_GAIN],
      s[..., L.SLIDE_MIN],
    )

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    s = full_state
    return torch.cat(
      [s[..., L.CAN_POS], s[..., L.CAN_QUAT], s[..., L.GOAL_POS], s[..., L.GOAL_QUAT]],
      -1,
    )

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    r = rep_state
    return torch.cat(
      [
        r[..., REP_Q],
        r[..., REP_DQ],
        aux_state[..., 0:7],
        r[..., REP_CAN_LIN_VEL],
        r[..., REP_CAN_ANG_VEL],
        aux_state[..., 7:14],
        r[..., REP_GOAL_LIN_VEL],
        r[..., REP_GOAL_ANG_VEL],
        r[..., REP_TARGET_ERROR] + r[..., REP_Q][..., : L.NUM_ARM],
        r[..., REP_GRIP_CMD],
        r[..., REP_FLAGS],
        r[..., REP_SLIDE_GAIN],
        r[..., REP_SLIDE_MIN],
      ],
      dim=-1,
    )

  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    return torch.cat([rep_state[..., REP_Q], rep_state[..., REP_DQ]], dim=-1)

  def scene(self, rep_state: torch.Tensor) -> Scene:
    """What the stage rules read of a rep state."""
    lever, dist, cos_far, can_up, goal_up, wrist_dist = rep_state[
      ..., REP_SCALARS
    ].unbind(-1)
    return Scene(
      can_pos=rep_state[..., REP_CAN_POS],
      can_speed_xy=rep_state[..., REP_CAN_LIN_VEL][..., :2].norm(dim=-1),
      can_up=can_up,
      goal_up=goal_up,
      lever=lever,
      dist=dist,
      cos_far=cos_far,
      wrist_dist=wrist_dist,
      grip_cmd=rep_state[..., REP_GRIP_CMD].squeeze(-1),
    )

  # ------------------------------------------------------------------
  # Dynamics integration
  # ------------------------------------------------------------------

  def _nominal_target(self, target: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    step = self.target_step * action[..., : L.NUM_ARM]
    return torch.maximum(torch.minimum(target + step, self.arm_hi), self.arm_lo)

  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
    action: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate one decision. ``action`` is the executed command (already
    clipped to ``[-1, 1]``)."""
    c = self.substep_coeff
    r = curr_rep_state
    q, dq = r[..., REP_Q], r[..., REP_DQ]
    n = L.NUM_JOINTS
    d_dq, d_cv, d_cw, d_gv, d_gw, r_q, r_cp, r_cr, r_gp, r_gr, r_target = (
      prediction * self.pred_scale
    ).split([n, 3, 3, 3, 3, n, 3, 3, 3, 3, L.NUM_ARM], dim=-1)

    next_dq = dq + d_dq
    next_q = q + (dq + c * d_dq) * dt + r_q

    def body(pos, quat, lin, ang, d_v, d_w, r_p, r_theta):
      next_pos = pos + (lin + c * d_v) * dt + r_p
      next_quat = quat_integrate(quat_integrate(quat, ang + c * d_w, dt), r_theta, 1.0)
      return next_pos, next_quat, lin + d_v, ang + d_w

    can = body(
      aux_state[..., 0:3],
      aux_state[..., 3:7],
      r[..., REP_CAN_LIN_VEL],
      r[..., REP_CAN_ANG_VEL],
      d_cv,
      d_cw,
      r_cp,
      r_cr,
    )
    goal = body(
      aux_state[..., 7:10],
      aux_state[..., 10:14],
      r[..., REP_GOAL_LIN_VEL],
      r[..., REP_GOAL_ANG_VEL],
      d_gv,
      d_gw,
      r_gp,
      r_gr,
    )
    target = r[..., REP_TARGET_ERROR] + q[..., : L.NUM_ARM]
    next_target = self._nominal_target(target, action) + r_target
    next_grip = (action[..., -1:] + 1.0) / 2.0

    # Assemble once with the old latches to read the scene, then advance them.
    next_rep = self._assemble(
      next_q,
      next_dq,
      can,
      goal,
      next_target - next_q[..., : L.NUM_ARM],
      next_grip,
      r[..., REP_FLAGS],
      r[..., REP_SLIDE_GAIN],
      r[..., REP_SLIDE_MIN],
    )
    flags, slide_gain, slide_min = self.rules.advance(
      r[..., REP_FLAGS],
      r[..., REP_SLIDE_GAIN].squeeze(-1),
      r[..., REP_SLIDE_MIN].squeeze(-1),
      self.scene(next_rep),
    )
    next_rep = torch.cat(
      [
        next_rep[..., : REP_FLAGS.start],
        flags,
        slide_gain.unsqueeze(-1),
        slide_min.unsqueeze(-1),
        next_rep[..., REP_SLIDE_MIN.stop :],
      ],
      dim=-1,
    )
    next_aux = torch.cat([can[0], can[1], goal[0], goal[1]], dim=-1)
    return next_aux, next_rep

  def prediction_target(
    self, full_state: torch.Tensor, next_full_state: torch.Tensor, action: torch.Tensor
  ) -> torch.Tensor:
    """The unscaled prediction that takes ``full_state`` to ``next_full_state``
    under ``action``: ``apply_prediction(target / pred_scale)`` reproduces the
    next physical state."""
    c, dt = self.substep_coeff, self.dt
    s, s1 = full_state, next_full_state
    d_dq = s1[..., L.DQ] - s[..., L.DQ]
    r_q = s1[..., L.Q] - s[..., L.Q] - (s[..., L.DQ] + c * d_dq) * dt

    def body(pos, quat, lin, ang):
      d_v = s1[..., lin] - s[..., lin]
      d_w = s1[..., ang] - s[..., ang]
      r_p = s1[..., pos] - s[..., pos] - (s[..., lin] + c * d_v) * dt
      coasted = quat_integrate(s[..., quat], s[..., ang] + c * d_w, dt)
      return d_v, d_w, r_p, rotvec_between(coasted, s1[..., quat])

    d_cv, d_cw, r_cp, r_cr = body(L.CAN_POS, L.CAN_QUAT, L.CAN_LIN_VEL, L.CAN_ANG_VEL)
    d_gv, d_gw, r_gp, r_gr = body(
      L.GOAL_POS, L.GOAL_QUAT, L.GOAL_LIN_VEL, L.GOAL_ANG_VEL
    )
    r_target = s1[..., L.TARGET] - self._nominal_target(
      s[..., L.TARGET], action.clamp(-1.0, 1.0)
    )
    return torch.cat(
      [d_dq, d_cv, d_cw, d_gv, d_gw, r_q, r_cp, r_cr, r_gp, r_gr, r_target], dim=-1
    )

  @torch.no_grad()
  def fit_prediction_scale(
    self,
    full_state: torch.Tensor,
    next_full_state: torch.Tensor,
    action: torch.Tensor,
    floors: dict[str, float] | None = None,
  ) -> dict[str, float]:
    """Set ``pred_scale`` to each prediction group's RMS over real transitions,
    so the network's outputs are O(1). ``floors`` gives per-group lower bounds,
    for data in which a group hardly varies. Returns the per-group scales."""
    floors = floors or {}
    target = self.prediction_target(full_state, next_full_state, action)
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
    The latches and the slide bookkeeping follow by rule and are not scored.

    Rotations are scaled by sqrt(1/2) so a plain squared difference is
    ``theta^2 + O(theta^4)`` (see ``representations._root_loss_blocks``).
    """
    r = rep_state
    q, dq = r[..., REP_Q], r[..., REP_DQ]
    can_rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    goal_rot = matrix_from_quat(aux_state[..., 10:14]).flatten(-2, -1)
    return {
      "root_pos": aux_state[..., 0:3],
      "root_ori": can_rot * math.sqrt(0.5),
      "goal_pos": aux_state[..., 7:10],
      "goal_ori": goal_rot * math.sqrt(0.5),
      "arm_pos": q[..., : L.NUM_ARM],
      "arm_vel": dq[..., : L.NUM_ARM],
      "finger_pos": q[..., L.NUM_ARM :],
      "finger_vel": dq[..., L.NUM_ARM :],
      "can_vel": r[..., REP_CAN_LIN_VEL],
      "can_avel": r[..., REP_CAN_ANG_VEL],
      "goal_vel": r[..., REP_GOAL_LIN_VEL],
      "goal_avel": r[..., REP_GOAL_ANG_VEL],
      "target": r[..., REP_TARGET_ERROR],
    }
