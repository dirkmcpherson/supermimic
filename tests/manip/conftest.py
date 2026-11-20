"""Fixtures for the manipulation tests (no mjlab; robosuite only where marked)."""

from __future__ import annotations

import os

# The learners wrap their updates in torch.compile; skip it on the test CPU.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from supermimic.manip import layout as L  # noqa: E402
from supermimic.manip.representation import ArmObjectRepresentation  # noqa: E402


def _random_rotations(rng: np.random.Generator, count: int) -> np.ndarray:
  q, r = np.linalg.qr(rng.standard_normal((count, 3, 3)))
  q = q * np.sign(np.linalg.det(q))[:, None, None]
  return q


def make_constants(seed: int = 0, goal: bool = False) -> dict[str, np.ndarray]:
  """Constants of a made-up 7-joint arm with the real control timing; with
  ``goal``, of a task with a grasp offset and a goal (as Square has)."""
  rng = np.random.default_rng(seed)
  axes = rng.standard_normal((L.NUM_ARM, 3))
  extra = {}
  if goal:
    extra = {
      "grasp_offset": np.array([0.054, 0.0, 0.0]),
      "goal_pos": np.array([0.23, 0.1, 0.8]),
    }
  return {
    **extra,
    "fk_pre_pos": rng.uniform(-0.2, 0.2, (L.NUM_ARM, 3)),
    "fk_pre_rot": _random_rotations(rng, L.NUM_ARM),
    "fk_axes": axes / np.linalg.norm(axes, axis=1, keepdims=True),
    "fk_tip_pos": np.array([0.0, 0.0, 0.1]),
    "fk_tip_rot": np.eye(3),
    "table_z": np.array(0.8),
    "dt": np.array(0.05),
    "substeps": np.array(25),
    "grip_cmd_step": np.array(0.25),
  }


def random_states(count: int, seed: int = 0) -> torch.Tensor:
  """Random full states with unit cube quaternions and a valid gripper command."""
  gen = torch.Generator().manual_seed(seed)
  state = torch.randn(count, L.STATE_DIM, generator=gen) * 0.3
  quat = torch.randn(count, 4, generator=gen)
  state[:, L.OBJ_QUAT] = quat / quat.norm(dim=-1, keepdim=True)
  state[:, L.OBJ_POS] += torch.tensor([0.0, 0.0, 0.85])
  state[:, L.GRIP_CMD] = torch.rand(count, 1, generator=gen) * 2 - 1
  return state


def random_episodes(
  count: int, episode_len: int, seed: int = 0
) -> dict[str, np.ndarray]:
  """Back-to-back fake episodes in the offline-data layout."""
  rng = np.random.default_rng(seed)
  truncated = np.zeros(count, dtype=bool)
  truncated[episode_len - 1 :: episode_len] = True
  truncated[-1] = True
  return {
    "state": random_states(count, seed).numpy(),
    "action": rng.uniform(-1, 1, (count, L.ACTION_DIM)).astype(np.float32),
    "prev_action": rng.uniform(-1, 1, (count, L.ACTION_DIM)).astype(np.float32),
    "sim_ctx": rng.uniform(-1, 1, (count, L.CTX_DIM)).astype(np.float32),
    "truncated": truncated,
  }


@pytest.fixture
def constants() -> dict[str, np.ndarray]:
  return make_constants()


@pytest.fixture
def representation(constants) -> ArmObjectRepresentation:
  return ArmObjectRepresentation(constants)


@pytest.fixture
def goal_representation() -> ArmObjectRepresentation:
  return ArmObjectRepresentation(make_constants(goal=True))
