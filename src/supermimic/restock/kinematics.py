"""Serial-chain parameters recovered from sampled link poses (NumPy only)."""

from __future__ import annotations

import numpy as np


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
  """Unit ``wxyz`` quaternions ``(..., 4)`` -> rotation matrices."""
  quat = np.asarray(quat, dtype=np.float64)
  w, x, y, z = np.moveaxis(quat / np.linalg.norm(quat, axis=-1, keepdims=True), -1, 0)
  return np.stack(
    [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ],
    axis=-1,
  ).reshape(*quat.shape[:-1], 3, 3)  # fmt: skip


def _axis_angle(rot: np.ndarray) -> tuple[np.ndarray, float]:
  angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0))
  axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
  return axis / np.linalg.norm(axis), float(angle)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
  x, y, z = axis
  skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
  return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def chain_from_link_poses(
  q: np.ndarray,
  link_pos: np.ndarray,
  link_quat: np.ndarray,
  link_ids: list[int],
  tip_pos: np.ndarray,
) -> dict[str, np.ndarray]:
  """Parameters of :class:`supermimic.manip.fk.SerialChainFK` for a chain of
  hinges, from forward-kinematics samples.

  ``q`` ``(S, J)`` joint angles, ``link_pos`` ``(S, L, 3)`` / ``link_quat``
  ``(S, L, 4)`` world poses of every link at those angles, ``link_ids`` the
  chain's ``J`` links in order (link ``j`` turns with joint ``j`` about an axis
  through its own origin, and its parent is the previous one; the first one's
  parent is fixed in the world at the origin with identity rotation removed by
  the fit). ``tip_pos`` is the tip in the last link's frame.
  """
  q = np.asarray(q, dtype=np.float64)
  rot = quat_to_mat(link_quat)
  pos = np.asarray(link_pos, dtype=np.float64)
  pre_pos, pre_rot, axes = [], [], []
  for j, link in enumerate(link_ids):
    if j == 0:
      rel_rot, rel_pos = rot[:, link], pos[:, link]
    else:
      parent = link_ids[j - 1]
      parent_rot_t = np.swapaxes(rot[:, parent], -1, -2)
      rel_rot = parent_rot_t @ rot[:, link]
      rel_pos = np.einsum("sij,sj->si", parent_rot_t, pos[:, link] - pos[:, parent])
    # The two samples whose joint angles differ most give the axis.
    a, b = int(np.argmin(q[:, j])), int(np.argmax(q[:, j]))
    axis, angle = _axis_angle(rel_rot[a].T @ rel_rot[b])
    delta = q[b, j] - q[a, j]
    # arccos returns the angle in [0, pi]; pick the sign and winding that match.
    best = min(
      ((s, k) for s in (1, -1) for k in (-1, 0, 1)),
      key=lambda sk: abs(sk[0] * angle + 2 * np.pi * sk[1] - delta),
    )
    axis = axis * best[0]
    axes.append(axis)
    pre_rot.append(rel_rot[a] @ _rodrigues(axis, -q[a, j]))
    pre_pos.append(rel_pos.mean(axis=0))
  return {
    "pre_pos": np.stack(pre_pos),
    "pre_rot": np.stack(pre_rot),
    "axes": np.stack(axes),
    "tip_pos": np.asarray(tip_pos, dtype=np.float64),
    "tip_rot": np.eye(3),
  }


def chain_fk(
  chain: dict[str, np.ndarray], q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Tip position and rotation for one joint vector (the reference the torch
  module is tested against)."""
  pos, rot = np.zeros(3), np.eye(3)
  for j in range(len(chain["axes"])):
    pos = pos + rot @ chain["pre_pos"][j]
    rot = rot @ chain["pre_rot"][j] @ _rodrigues(chain["axes"][j], float(q[j]))
  return pos + rot @ chain["tip_pos"], rot @ chain["tip_rot"]
