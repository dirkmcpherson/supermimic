"""Differentiable forward kinematics of a serial chain of hinge joints."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from supermimic.manip.rotations import mat_vec, small_matmul


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat / np.linalg.norm(quat)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def chain_from_mujoco(
  model, site_id: int, joint_ids: list[int]
) -> dict[str, np.ndarray]:
  """Kinematic parameters of the hinge chain from the world to ``site_id``.

  ``model`` is a raw ``mujoco.MjModel``. Returns, per joint, the fixed transform
  from the previous joint frame (the world for the first) to this joint's body
  frame and the joint axis in that frame, plus the fixed transform from the last
  joint frame to the site. Every joint on the path must be one of ``joint_ids``,
  in order, a hinge anchored at its body origin.
  """
  path = []
  body = int(model.site_bodyid[site_id])
  while body != 0:
    path.append(body)
    body = int(model.body_parentid[body])
  path.reverse()

  acc_pos, acc_rot = np.zeros(3), np.eye(3)
  pre_pos, pre_rot, axes, found = [], [], [], []
  for body in path:
    acc_pos = acc_pos + acc_rot @ model.body_pos[body]
    acc_rot = acc_rot @ _quat_to_mat(model.body_quat[body])
    num = int(model.body_jntnum[body])
    if num == 0:
      continue
    joint = int(model.body_jntadr[body])
    if num != 1 or int(model.jnt_type[joint]) != 3:  # mjJNT_HINGE
      raise ValueError(f"body {body} is not driven by a single hinge joint")
    if not np.allclose(model.jnt_pos[joint], 0.0):
      raise ValueError(f"joint {joint} is not anchored at its body origin")
    found.append(joint)
    pre_pos.append(acc_pos)
    pre_rot.append(acc_rot)
    axes.append(model.jnt_axis[joint] / np.linalg.norm(model.jnt_axis[joint]))
    acc_pos, acc_rot = np.zeros(3), np.eye(3)
  if found != list(joint_ids):
    raise ValueError(f"chain joints {found} do not match the requested {joint_ids}")
  tip_pos = acc_pos + acc_rot @ model.site_pos[site_id]
  tip_rot = acc_rot @ _quat_to_mat(model.site_quat[site_id])
  return {
    "pre_pos": np.stack(pre_pos),
    "pre_rot": np.stack(pre_rot),
    "axes": np.stack(axes),
    "tip_pos": tip_pos,
    "tip_rot": tip_rot,
  }


class SerialChainFK(nn.Module):
  """World pose of a chain's tip frame from its joint angles."""

  def __init__(self, chain: dict[str, np.ndarray]):
    super().__init__()

    def buf(name: str, value: np.ndarray) -> None:
      self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    axes = torch.as_tensor(chain["axes"], dtype=torch.float32)
    zero = torch.zeros(axes.shape[0])
    x, y, z = axes.unbind(-1)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).unflatten(
      -1, (3, 3)
    )
    buf("pre_pos", chain["pre_pos"])
    buf("pre_rot", chain["pre_rot"])
    buf("tip_pos", chain["tip_pos"])
    buf("tip_rot", chain["tip_rot"])
    self.register_buffer("skew", skew)
    self.register_buffer("skew_sq", skew @ skew)
    self.register_buffer("eye", torch.eye(3))
    self.num_joints = axes.shape[0]

  def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` ``(..., J)`` -> tip ``(pos (..., 3), rot (..., 3, 3))``."""
    sin, cos = torch.sin(q), torch.cos(q)
    pos = q.new_zeros(*q.shape[:-1], 3)
    rot = self.eye.expand(*q.shape[:-1], 3, 3)
    for j in range(self.num_joints):
      pos = pos + mat_vec(rot, self.pre_pos[j])
      rot = small_matmul(rot, self.pre_rot[j])
      # Rodrigues: rotation by q_j about the joint axis.
      s = sin[..., j, None, None]
      c = cos[..., j, None, None]
      rot = small_matmul(rot, self.eye + s * self.skew[j] + (1 - c) * self.skew_sq[j])
    pos = pos + mat_vec(rot, self.tip_pos)
    return pos, small_matmul(rot, self.tip_rot)
