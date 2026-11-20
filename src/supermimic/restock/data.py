"""Replay buffers of restock transitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.manip.data import MixtureSampler
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.restock import layout as L

__all__ = [
  "SCHEMA",
  "MixtureSampler",
  "buffer_from_episodes",
  "load_npz",
  "with_final_states",
]

SCHEMA: dict[str, FieldSpec] = {
  "state": FieldSpec(L.STATE_DIM),
  "action": FieldSpec(L.ACTION_DIM),
  "prev_action": FieldSpec(L.ACTION_DIM),
  "sim_ctx": FieldSpec(L.CTX_DIM),
  "terminated": FieldSpec(dtype=torch.bool),
  "truncated": FieldSpec(dtype=torch.bool),
}


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as f:
    return {k: f[k] for k in f.files}


def with_final_states(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Give the state every episode ends in a row of its own.

  ``data`` has one row per transition (``state``, ``action``, ``next_state``)
  with the end flags on an episode's last transition. The result has one row
  per state: the last transition's ``next_state`` follows it as a row with a
  null action that carries the end flag instead. A window of the result that
  ends on such a row trains the world model on the transition into the
  episode's final state."""
  ended = data["terminated"] | data["truncated"]
  # Position of every original row in the result: one extra row after each end.
  shift = np.concatenate([[0], np.cumsum(ended)[:-1]])
  rows = np.arange(len(ended)) + shift
  final = rows[ended] + 1
  count = len(ended) + int(ended.sum())
  out = {}
  for name, spec in SCHEMA.items():
    value = data[name]
    column = np.zeros((count, *value.shape[1:]), dtype=value.dtype)
    if name in ("terminated", "truncated"):
      column[final] = value[ended]
    else:
      column[rows] = value
      if name == "state":
        column[final] = data["next_state"][ended]
      elif name == "prev_action":
        column[final] = data["action"][ended]
      elif name == "sim_ctx":
        column[final] = value[ended]
    assert spec.dim == 0 or column.shape[1] == spec.dim
    out[name] = column
  return out


def buffer_from_episodes(
  data: dict[str, np.ndarray], num_rows: int, device: str
) -> ReplayBuffer:
  """Pack back-to-back episodes into a full ``(num_rows, T)`` replay buffer.

  The flat sequence is cut into ``num_rows`` equal rows; a row boundary that
  falls inside an episode only splits it in two shorter ones. The tail is padded
  by wrapping to the start, which the last episode's end flag separates.
  """
  count = len(data["state"])
  ended = data["terminated"] | data["truncated"]
  if count == 0 or not ended[-1]:
    raise ValueError("episodes must be non-empty and the last row must end one")
  steps = -(-count // num_rows)
  index = np.arange(steps * num_rows) % count

  def rows(name: str, dtype: torch.dtype) -> torch.Tensor:
    value = torch.as_tensor(data[name][index], dtype=dtype, device=device)
    return value.reshape(num_rows, steps, *value.shape[1:])

  fields = {
    name: rows(name, torch.bool if spec.dtype == torch.bool else torch.float32)
    for name, spec in SCHEMA.items()
  }
  buffer = ReplayBuffer(SCHEMA, buffer_size=steps, num_envs=num_rows, device=device)
  for t in range(steps):
    buffer.insert(
      TensorDict({k: v[:, t] for k, v in fields.items()}, batch_size=[num_rows])
    )
  return buffer
