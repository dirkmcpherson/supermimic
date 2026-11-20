from __future__ import annotations

import numpy as np
import pytest
import torch

from supermimic.manip import layout as L
from supermimic.manip.collect import split_episodes
from supermimic.manip.data import MixtureSampler, buffer_from_episodes
from supermimic.manip.demos import convert_states, integrate_grip_cmd

from .conftest import random_episodes


def _tagged_episodes(count: int, episode_len: int) -> dict[str, np.ndarray]:
  """Episodes whose state column 0 holds the episode index, column 1 the step."""
  data = random_episodes(count, episode_len)
  index = np.arange(count)
  data["state"][:, 0] = index // episode_len
  data["state"][:, 1] = index % episode_len
  return data


def test_buffer_windows_stay_inside_one_episode():
  data = _tagged_episodes(600, 12)
  buffer = buffer_from_episodes(data, num_rows=7, device="cpu")
  windows = buffer.sample(5, 512)["state"]
  episode, step = windows[..., 0], windows[..., 1]
  assert (episode == episode[:, :1]).all()
  assert (step[:, 1:] - step[:, :-1] == 1).all()


def test_buffer_keeps_every_transition():
  data = _tagged_episodes(100, 10)
  buffer = buffer_from_episodes(data, num_rows=8, device="cpu")
  singles = buffer.sample(1, 4096).squeeze(1)["state"]
  seen = {(int(e), int(s)) for e, s in singles[:, :2].tolist()}
  assert len(seen) == 100


def test_buffer_rejects_an_unfinished_last_episode():
  data = random_episodes(40, 10)
  data["truncated"][-1] = False
  with pytest.raises(ValueError):
    buffer_from_episodes(data, num_rows=4, device="cpu")


def test_split_episodes_cuts_on_an_episode_boundary():
  data = _tagged_episodes(100, 10)
  data["episode_success"] = np.arange(10) % 2 == 0
  train, val = split_episodes(data, 0.2)
  assert len(train["state"]) == 80 and len(val["state"]) == 20
  assert train["truncated"][-1] and val["truncated"][-1]
  assert set(val["state"][:, 0]) == {8.0, 9.0}
  assert len(train["episode_success"]) == 8 and len(val["episode_success"]) == 2


def test_mixture_sampler_shares_and_fallback():
  full_a = buffer_from_episodes(_tagged_episodes(200, 10), 4, "cpu")
  data_b = _tagged_episodes(200, 10)
  data_b["state"][:, 2] = 7.0
  full_b = buffer_from_episodes(data_b, 4, "cpu")
  sampler = MixtureSampler([(full_a, 0.25), (full_b, 0.75)])
  batch = sampler.sample(4, 64)
  assert batch.batch_size == torch.Size([64, 4])
  assert int((batch["state"][:, 0, 2] == 7.0).sum()) == 48

  from supermimic.manip.data import SCHEMA
  from supermimic.replay_buffer import ReplayBuffer

  empty = ReplayBuffer(SCHEMA, buffer_size=50, num_envs=4, device="cpu")
  sampler = MixtureSampler([(empty, 0.5), (full_b, 0.5)])
  batch = sampler.sample(4, 64)
  assert batch.batch_size == torch.Size([64, 4])
  assert (batch["state"][:, 0, 2] == 7.0).all()

  with pytest.raises(ValueError):
    MixtureSampler([(full_a, 0.5), (full_b, 0.6)])


def test_integrate_grip_cmd_is_the_command_before_each_action():
  actions = np.zeros((7, L.ACTION_DIM))
  actions[:, -1] = [1, 1, 1, 1, 1, -1, 0.5]
  cmd = integrate_grip_cmd(actions, 0.25)
  assert np.allclose(cmd, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75])


def test_convert_states_maps_the_flattened_layout():
  rng = np.random.default_rng(0)
  flat = rng.standard_normal((5, 32))
  # Cube rotated 90 degrees about z: body x-axis points along world y.
  flat[:, 13:17] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
  flat[:, 29:32] = [1.0, 0.0, 0.0]  # body-frame angular velocity
  actions = np.zeros((5, L.ACTION_DIM))
  flat_layout = {
    "nq": 16,
    "nv": 15,
    "q_idx": np.arange(9),
    "dq_idx": np.arange(9),
    "obj_qpos": 9,
    "obj_qvel": 9,
  }
  out = convert_states(flat, actions, 0.25, flat_layout)
  assert np.allclose(out[:, L.Q], flat[:, 1:10])
  assert np.allclose(out[:, L.DQ], flat[:, 17:26])
  assert np.allclose(out[:, L.OBJ_POS], flat[:, 10:13])
  assert np.allclose(out[:, L.OBJ_LIN_VEL], flat[:, 26:29])
  assert np.allclose(out[:, L.OBJ_ANG_VEL], [0.0, 1.0, 0.0], atol=1e-12)
  assert np.allclose(out[:, L.GRIP_CMD], 0.0)
