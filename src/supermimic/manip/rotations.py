"""Rotation helpers (quaternions are ``wxyz``), free of mjlab.

Small matrix products are written element-wise so they keep FP32 precision when
TF32 matmuls are enabled for the MLPs, and fuse under ``torch.compile``.
"""

from __future__ import annotations

import torch


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """``a @ b`` for ``(..., n, k)`` and ``(..., k, m)`` without TF32 rounding."""
  return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def mat_vec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """``mat @ vec`` for ``(..., n, k)`` and ``(..., k)``."""
  return (mat * vec.unsqueeze(-2)).sum(-1)


def mat_t_vec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """``mat.T @ vec`` for ``(..., k, n)`` and ``(..., k)``."""
  return (mat * vec.unsqueeze(-1)).sum(-2)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product of two ``wxyz`` quaternions."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dim=-1,
  )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  """Rotation matrix ``(..., 3, 3)`` of a unit ``wxyz`` quaternion."""
  w, x, y, z = q.unbind(-1)
  rows = [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ]
  return torch.stack(rows, dim=-1).unflatten(-1, (3, 3))


def rot6d_from_matrix(mat: torch.Tensor) -> torch.Tensor:
  """First two columns of a rotation matrix, flattened row-major ``(..., 6)``."""
  return mat[..., :2].flatten(-2, -1)


def quat_integrate(
  quat: torch.Tensor, ang_vel: torch.Tensor, dt: float
) -> torch.Tensor:
  """Rotate ``quat`` by the world-frame angular velocity ``ang_vel`` over ``dt``."""
  omega = ang_vel * dt
  norm = torch.sqrt((omega * omega).sum(dim=-1, keepdim=True) + 1e-12)
  half = norm / 2
  delta = torch.cat([torch.cos(half), torch.sin(half) * omega / norm], dim=-1)
  out = quat_mul(delta, quat)
  return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True)


def rotvec_between(quat_from: torch.Tensor, quat_to: torch.Tensor) -> torch.Tensor:
  """World-frame rotation vector taking ``quat_from`` to ``quat_to``."""
  err = quat_mul(quat_to, quat_conjugate(quat_from))
  err = err * torch.where(err[..., :1] < 0, -1.0, 1.0)
  vec = err[..., 1:]
  sin_half = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
  angle = 2 * torch.atan2(sin_half, err[..., :1])
  return vec * angle / sin_half.clamp(min=1e-9)


def rotation_angle(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
  """Angle in radians between two rotation matrices."""
  trace = (mat_a * mat_b).sum(dim=(-2, -1))
  return torch.acos(((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))
