"""Fixtures for the restock tests (no simulator: made-up arm, real task numbers)."""

from __future__ import annotations

import os

# The learners wrap their updates in torch.compile; skip it on the test CPU.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from supermimic.restock import layout as L  # noqa: E402
from supermimic.restock.representation import RestockRepresentation  # noqa: E402


def make_chain(seed: int = 0) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  rot, _ = np.linalg.qr(rng.standard_normal((L.NUM_ARM, 3, 3)))
  rot = rot * np.sign(np.linalg.det(rot))[:, None, None]
  axes = rng.standard_normal((L.NUM_ARM, 3))
  return {
    "pre_pos": rng.uniform(-0.2, 0.2, (L.NUM_ARM, 3)),
    "pre_rot": rot,
    "axes": axes / np.linalg.norm(axes, axis=1, keepdims=True),
    "tip_pos": np.array([-0.08, 0.0, 0.13]),
    "tip_rot": np.eye(3),
  }


def make_constants(seed: int = 0) -> dict[str, np.ndarray]:
  """A made-up six-joint chain with the built environment's task numbers."""
  scalars = {
    "dt": 0.12,
    "target_step": 0.1,
    "leash": 0.125,
    "shelf_top_z": 0.17,
    "pick_z": 0.1505,
    "pick_grip": 0.3,
    "pick_wrist_dist": 0.2,
    "place_release": 0.45,
    "band_lo": 0.01,
    "band_hi": 0.07,
    "tilt_max_deg": 20.0,
    "tip_deg": 60.0,
    "held_lever": 0.025,
    "farside_reach": 0.10,
    "farside_cone_deg": 60.0,
    "slide_gain_min": 0.01,
    "nested_touch_dist": 0.081,
    "home_reward": 10.0,
  }
  return {
    **{f"fk_{k}": v for k, v in make_chain(seed).items()},
    **{k: np.array(v) for k, v in scalars.items()},
    "substeps": np.array(96),
    "arm_lo": np.full(L.NUM_ARM, -3.0),
    "arm_hi": np.full(L.NUM_ARM, 3.0),
    "shelf_lo": np.array([0.55, -0.5625]),
    "shelf_hi": np.array([0.95, 0.1875]),
  }


def random_states(count: int, seed: int = 0) -> torch.Tensor:
  """Random full states with unit quaternions and latches in range."""
  gen = torch.Generator().manual_seed(seed)
  state = torch.randn(count, L.STATE_DIM, generator=gen) * 0.3
  for quat in (L.CAN_QUAT, L.GOAL_QUAT):
    value = torch.randn(count, 4, generator=gen)
    state[:, quat] = value / value.norm(dim=-1, keepdim=True)
  state[:, L.GRIP_CMD] = torch.rand(count, 1, generator=gen)
  state[:, L.FLAGS] = (torch.rand(count, 5, generator=gen) > 0.5).float()
  state[:, L.SLIDE_GAIN] = torch.rand(count, 1, generator=gen) * 0.02
  state[:, L.SLIDE_MIN] = torch.rand(count, 1, generator=gen) * 0.3
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
    "sim_ctx": np.zeros((count, L.CTX_DIM), np.float32),
    "terminated": np.zeros(count, dtype=bool),
    "truncated": truncated,
  }


@pytest.fixture
def constants() -> dict[str, np.ndarray]:
  return make_constants()


@pytest.fixture
def representation(constants) -> RestockRepresentation:
  return RestockRepresentation(constants)
