"""Replay buffers over offline episodes, and mixed sampling across buffers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.manip import layout as L
from supermimic.replay_buffer import FieldSpec, ReplayBuffer

SCHEMA: dict[str, FieldSpec] = {
  "state": FieldSpec(L.STATE_DIM),
  "action": FieldSpec(L.ACTION_DIM),
  "prev_action": FieldSpec(L.ACTION_DIM),
  "sim_ctx": FieldSpec(L.CTX_DIM),
  "terminated": FieldSpec(dtype=torch.bool),
  "truncated": FieldSpec(dtype=torch.bool),
}


def load_constants(data_dir: str | Path) -> dict[str, np.ndarray]:
  with np.load(Path(data_dir) / "constants.npz") as f:
    return {k: f[k] for k in f.files}


def task_of(constants: dict[str, np.ndarray]) -> str:
  """The task a dataset was built for (datasets from before tasks were named
  are Lift)."""
  return str(constants["task"]) if "task" in constants else "lift"


def load_episodes(path: str | Path) -> dict[str, np.ndarray]:
  with np.load(path) as f:
    return {k: f[k] for k in f.files}


def buffer_from_episodes(
  data: dict[str, np.ndarray], num_rows: int, device: str
) -> ReplayBuffer:
  """Pack back-to-back episodes into a full ``(num_rows, T)`` replay buffer.

  The flat sequence is cut into ``num_rows`` equal rows; a row boundary that
  falls inside an episode only splits it in two shorter ones. The tail is padded
  by wrapping to the start, which the last episode's ``truncated`` flag separates.
  """
  count = len(data["state"])
  if count == 0 or not data["truncated"][-1]:
    raise ValueError("episodes must be non-empty and end with a truncated row")
  steps = -(-count // num_rows)
  index = np.arange(steps * num_rows) % count

  def rows(name: str, dtype: torch.dtype) -> torch.Tensor:
    value = torch.as_tensor(data[name][index], dtype=dtype, device=device)
    return value.reshape(num_rows, steps, *value.shape[1:])

  fields = {
    "state": rows("state", torch.float32),
    "action": rows("action", torch.float32),
    "prev_action": rows("prev_action", torch.float32),
    "sim_ctx": rows("sim_ctx", torch.float32),
    "truncated": rows("truncated", torch.bool),
  }
  fields["terminated"] = torch.zeros_like(fields["truncated"])
  buffer = ReplayBuffer(SCHEMA, buffer_size=steps, num_envs=num_rows, device=device)
  for t in range(steps):
    buffer.insert(
      TensorDict({k: v[:, t] for k, v in fields.items()}, batch_size=[num_rows])
    )
  return buffer


class MixtureSampler:
  """Draws fixed shares of every batch from several replay buffers.

  A buffer that cannot yet supply a window (the online buffer early in a run)
  hands its share to the first buffer that can.
  """

  def __init__(self, sources: list[tuple[ReplayBuffer, float]], min_fill: int = 2):
    if not sources or abs(sum(w for _, w in sources) - 1.0) > 1e-6:
      raise ValueError("source weights must sum to 1")
    self.sources = sources
    self.min_fill = min_fill

  def _counts(self, batch_size: int, sequence_length: int) -> list[int]:
    counts, taken = [], 0
    for i, (_, weight) in enumerate(self.sources):
      last = i == len(self.sources) - 1
      n = batch_size - taken if last else int(round(batch_size * weight))
      counts.append(n)
      taken += n
    ready = [
      buf.num_transitions >= self.min_fill * sequence_length for buf, _ in self.sources
    ]
    if not any(ready):
      raise RuntimeError("no replay buffer can supply a window yet")
    fallback = ready.index(True)
    for i, ok in enumerate(ready):
      if not ok:
        counts[fallback] += counts[i]
        counts[i] = 0
    return counts

  def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
    parts = [
      buf.sample(sequence_length, n)
      for (buf, _), n in zip(
        self.sources, self._counts(batch_size, sequence_length), strict=True
      )
      if n > 0
    ]
    return torch.cat(parts, dim=0)
