"""Load robomimic-format demonstrations into the :mod:`layout` state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from supermimic.manip import layout as L

FIRST_FRAME = 1
"""Recorded frames dropped from the start of every demo. The first recorded
transition is not one the simulator reproduces (the arm ends up about 4 mrad
off, in Lift and in Square alike; apparently an artefact of how the recorder
starts an episode) and, with delta actions, an open-loop replay never recovers
from it. From the second frame on, Lift demos replay bit for bit."""


@dataclass
class Demo:
  name: str
  states: np.ndarray
  """``(T, 32)``; ``states[t]`` is the state ``actions[t]`` was applied in."""
  actions: np.ndarray
  """``(T, 7)``."""
  variant: np.ndarray
  """``(3,)`` variant of the model the demo was recorded in (Lift: the cube's
  half-extents)."""


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Rotate ``vec`` ``(T, 3)`` by unit ``wxyz`` quaternions ``(T, 4)``."""
  w, xyz = quat[:, :1], quat[:, 1:]
  t = 2 * np.cross(xyz, vec)
  return vec + w * t + np.cross(xyz, t)


def integrate_grip_cmd(
  actions: np.ndarray, grip_cmd_step: float, initial: float = 0.0
) -> np.ndarray:
  """Gripper command before each action: ``g += step * clip(a, -1, 1)``, clipped."""
  out = np.empty(len(actions))
  grip = initial
  for t, action in enumerate(actions[:, -1]):
    out[t] = grip
    grip = float(np.clip(grip + grip_cmd_step * np.clip(action, -1.0, 1.0), -1.0, 1.0))
  return out


def convert_states(
  flat: np.ndarray, actions: np.ndarray, grip_cmd_step: float, flat_layout: dict
) -> np.ndarray:
  """robomimic flattened states ``[time, qpos, qvel]`` -> :mod:`layout` states.

  ``flat_layout`` is :meth:`ArmObjectSim.flat_layout`. The gripper command is
  not stored; it is rebuilt from the actions, starting from the reset value 0
  (demos are recorded from the reset on).
  """
  nq, nv = flat_layout["nq"], flat_layout["nv"]
  if flat.shape[1] != 1 + nq + nv:
    raise ValueError(f"unexpected flattened state width {flat.shape[1]}")
  qpos, qvel = flat[:, 1 : 1 + nq], flat[:, 1 + nq :]
  p, v = flat_layout["obj_qpos"], flat_layout["obj_qvel"]
  quat = qpos[:, p + 3 : p + 7]
  quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  out = np.empty((len(flat), L.STATE_DIM))
  out[:, L.Q] = qpos[:, flat_layout["q_idx"]]
  out[:, L.DQ] = qvel[:, flat_layout["dq_idx"]]
  out[:, L.OBJ_POS] = qpos[:, p : p + 3]
  out[:, L.OBJ_QUAT] = quat
  out[:, L.OBJ_LIN_VEL] = qvel[:, v : v + 3]
  out[:, L.OBJ_ANG_VEL] = _quat_rotate(quat, qvel[:, v + 3 : v + 6])
  out[:, L.GRIP_CMD.start] = integrate_grip_cmd(actions, grip_cmd_step)
  return out


def load_demos(
  path: str,
  grip_cmd_step: float,
  flat_layout: dict,
  variant_from_model_xml: Callable[[str], np.ndarray],
  limit: int | None = None,
) -> list[Demo]:
  import h5py

  demos = []
  with h5py.File(path, "r") as f:
    data = f["data"]
    names = sorted(data.keys(), key=lambda n: int(n.split("_")[1]))
    for name in names[:limit]:
      group = data[name]
      actions = np.asarray(group["actions"], dtype=np.float64)
      try:
        variant = variant_from_model_xml(group.attrs["model_file"])
      except ValueError as err:
        raise ValueError(f"{path}:{name}: {err}") from None
      states = convert_states(
        np.asarray(group["states"], dtype=np.float64),
        actions,
        grip_cmd_step,
        flat_layout,
      )
      demos.append(Demo(name, states[FIRST_FRAME:], actions[FIRST_FRAME:], variant))
  return demos
