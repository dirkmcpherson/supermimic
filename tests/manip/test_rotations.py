from __future__ import annotations

import math

import torch

from supermimic.manip.rotations import (
  mat_t_vec,
  mat_vec,
  matrix_from_quat,
  quat_integrate,
  quat_mul,
  rotation_angle,
  rotvec_between,
  small_matmul,
)


def _unit_quats(count: int) -> torch.Tensor:
  q = torch.randn(count, 4, generator=torch.Generator().manual_seed(0))
  return q / q.norm(dim=-1, keepdim=True)


def test_matrix_from_quat_is_a_rotation_and_composes():
  a, b = _unit_quats(8), _unit_quats(8).flip(0)
  mat = matrix_from_quat(a)
  assert torch.allclose(
    mat @ mat.transpose(-1, -2), torch.eye(3).expand(8, 3, 3), atol=1e-5
  )
  assert torch.allclose(torch.linalg.det(mat), torch.ones(8), atol=1e-5)
  assert torch.allclose(
    matrix_from_quat(quat_mul(a, b)), mat @ matrix_from_quat(b), atol=1e-5
  )


def test_matrix_from_quat_z_rotation():
  angle = 0.3
  quat = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
  rotated = mat_vec(matrix_from_quat(quat), torch.tensor([1.0, 0.0, 0.0]))
  assert torch.allclose(
    rotated, torch.tensor([math.cos(angle), math.sin(angle), 0.0]), atol=1e-6
  )


def test_small_products_match_matmul():
  a, b, v = torch.randn(5, 3, 3), torch.randn(5, 3, 3), torch.randn(5, 3)
  assert torch.allclose(small_matmul(a, b), a @ b, atol=1e-6)
  assert torch.allclose(small_matmul(a, b[0]), a @ b[0], atol=1e-6)
  assert torch.allclose(mat_vec(a, v), (a @ v.unsqueeze(-1)).squeeze(-1), atol=1e-6)
  assert torch.allclose(
    mat_t_vec(a, v), (a.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1), atol=1e-6
  )


def test_quat_integrate_rotates_about_the_world_axis():
  quat = _unit_quats(6)
  omega = torch.tensor([0.0, 0.0, 2.0]).expand(6, 3)
  out = quat_integrate(quat, omega, 0.1)
  spin = torch.tensor([math.cos(0.1), 0.0, 0.0, math.sin(0.1)])
  # World-frame angular velocity: the increment multiplies on the left.
  assert torch.allclose(
    matrix_from_quat(out), matrix_from_quat(spin) @ matrix_from_quat(quat), atol=1e-5
  )


def test_rotvec_between_inverts_quat_integrate():
  quat = _unit_quats(16)
  rotvec = torch.randn(16, 3, generator=torch.Generator().manual_seed(1)) * 0.5
  moved = quat_integrate(quat, rotvec, 1.0)
  assert torch.allclose(rotvec_between(quat, moved), rotvec, atol=1e-5)
  # The double cover does not matter.
  assert torch.allclose(rotvec_between(quat, -moved), rotvec, atol=1e-5)
  assert torch.allclose(
    rotation_angle(matrix_from_quat(quat), matrix_from_quat(moved)),
    rotvec.norm(dim=-1),
    atol=1e-3,
  )
