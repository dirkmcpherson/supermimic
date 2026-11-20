"""Open-loop accuracy of the world model on held-out windows."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.manip import layout as L
from supermimic.manip.representation import (
  REP_EEF_POS,
  REP_Q,
  ArmObjectRepresentation,
)
from supermimic.manip.rotations import matrix_from_quat, rotation_angle
from supermimic.models import Dynamics

MOVED_THRESHOLD = 0.005
"""A window counts as a contact window if the cube ends up at least this far
(metres) from where it started."""


def _summary(values: torch.Tensor) -> dict[str, float]:
  if values.numel() == 0:
    return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
  q = torch.quantile(values, torch.tensor([0.5, 0.9], device=values.device))
  return {"mean": float(values.mean()), "median": float(q[0]), "p90": float(q[1])}


@torch.no_grad()
def open_loop_errors(
  dynamics: Dynamics,
  representation: ArmObjectRepresentation,
  windows: TensorDict,
  steps: tuple[int, ...],
) -> dict:
  """Roll the model out over ``windows`` ``(B, K+1)`` with the recorded actions.

  Reports, after each horizon in ``steps``, the error of the predicted cube
  position, cube rotation, grip-point position and finger opening, over all
  windows and over the contact windows (where the cube really moved by that
  horizon). ``hold`` is the error of predicting that the cube stays where it
  started: the bar a model has to clear.
  """
  states = windows["state"]
  rep_true = representation.represent(states)
  aux_true = representation.extract_aux_state(states)
  ctx = windows["sim_ctx"][:, 0]
  rep, aux = rep_true[:, 0], aux_true[:, 0]

  def gap(r: torch.Tensor) -> torch.Tensor:
    q = r[..., REP_Q]
    return q[..., L.NUM_ARM] - q[..., L.NUM_ARM + 1]

  out: dict = {"windows": int(states.shape[0])}
  for k in range(1, max(steps) + 1):
    aux, rep = dynamics(rep, windows["action"][:, k - 1], aux, ctx)
    if k not in steps:
      continue
    obj_err = (aux[:, :3] - aux_true[:, k, :3]).norm(dim=-1)
    hold_err = (aux_true[:, 0, :3] - aux_true[:, k, :3]).norm(dim=-1)
    rot_err = rotation_angle(
      matrix_from_quat(aux[:, 3:7]), matrix_from_quat(aux_true[:, k, 3:7])
    )
    eef_err = (rep[:, REP_EEF_POS] - rep_true[:, k, REP_EEF_POS]).norm(dim=-1)
    gap_err = (gap(rep) - gap(rep_true[:, k])).abs()
    moved = hold_err > MOVED_THRESHOLD
    out[f"k{k}"] = {
      "obj_pos_m": _summary(obj_err),
      "obj_pos_hold_m": _summary(hold_err),
      "obj_rot_rad": _summary(rot_err),
      "eef_pos_m": _summary(eef_err),
      "finger_gap_m": _summary(gap_err),
      "contact_fraction": float(moved.float().mean()),
      "contact_obj_pos_m": _summary(obj_err[moved]),
      "contact_obj_pos_hold_m": _summary(hold_err[moved]),
      "contact_obj_rot_rad": _summary(rot_err[moved]),
    }
  return out
