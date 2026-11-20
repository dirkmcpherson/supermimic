"""Learner config for the restock task (the dynamics and the actor are the
manipulation ones)."""

from __future__ import annotations

from dataclasses import dataclass

from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.manip.models import Actor, ManipDynamics

__all__ = ["Actor", "ManipDynamics", "RestockWorldModelLearnerCfg", "prediction_floors"]


@dataclass
class RestockWorldModelLearnerCfg(WorldModelLearnerCfg):
  """Bryson's-rule nominal errors of the restock loss groups."""

  # The can to move (the ``root_pos`` / ``root_ori`` groups) and the goal can:
  # metres, radians. Success is decided at the centimetre, so 2 mm.
  nominal_error_root_pose_pos: float = 0.002
  nominal_error_root_pose_ori: float = 0.02
  nominal_error_goal_pos: float = 0.002
  nominal_error_goal_ori: float = 0.02
  nominal_error_arm_pos: float = 0.005
  nominal_error_arm_vel: float = 0.1
  # Finger joints are hinges: radians, radians per second.
  nominal_error_finger_pos: float = 0.01
  nominal_error_finger_vel: float = 0.2
  nominal_error_can_vel: float = 0.02
  nominal_error_can_avel: float = 0.2
  nominal_error_goal_vel: float = 0.02
  nominal_error_goal_avel: float = 0.2
  nominal_error_target: float = 0.005


def prediction_floors(cfg: RestockWorldModelLearnerCfg) -> dict[str, float]:
  """Lower bounds on the prediction scales: a quantity that hardly varies in
  the data (the goal can is rarely touched) is still predicted in units of the
  error the loss tolerates for it."""
  return {
    "d_arm_vel": cfg.nominal_error_arm_vel,
    "d_finger_vel": cfg.nominal_error_finger_vel,
    "d_can_lin_vel": cfg.nominal_error_can_vel,
    "d_can_ang_vel": cfg.nominal_error_can_avel,
    "d_goal_lin_vel": cfg.nominal_error_goal_vel,
    "d_goal_ang_vel": cfg.nominal_error_goal_avel,
    "r_arm_pos": cfg.nominal_error_arm_pos,
    "r_finger_pos": cfg.nominal_error_finger_pos,
    "r_can_pos": cfg.nominal_error_root_pose_pos,
    "r_can_rot": cfg.nominal_error_root_pose_ori,
    "r_goal_pos": cfg.nominal_error_goal_pos,
    "r_goal_rot": cfg.nominal_error_goal_ori,
    "r_target": cfg.nominal_error_target,
  }
