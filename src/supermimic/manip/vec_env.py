"""Subprocess-vectorized task simulators with automatic resets.

NumPy only (the workers must not import torch). Each worker owns one
:class:`ArmObjectSim`; an episode that is truncated restarts at once, either with a
standard task reset or from a row of a bank of start states.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np

from supermimic.manip.sim import SimCfg, make_sim

_THREAD_VARS = (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMBA_NUM_THREADS",
)


@dataclass
class StartBank:
  """Start states to restore episodes from, with the model variant each needs."""

  states: np.ndarray
  """``(M, 32)``"""
  variants: np.ndarray
  """``(M, 3)``"""


def spawn_context() -> mp.context.BaseContext:
  return mp.get_context("spawn")


class single_threaded_children:
  """Make child processes started inside the block use one thread each."""

  def __enter__(self):
    self._saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))

  def __exit__(self, *exc):
    for var, value in self._saved.items():
      if value is None:
        os.environ.pop(var, None)
      else:
        os.environ[var] = value


def _worker(conn, cfg: SimCfg, seed: int, bank: StartBank | None, fraction: float):
  sim = make_sim(cfg, seed)
  rng = np.random.default_rng(seed + 7919)

  def start() -> tuple[np.ndarray, bool]:
    if bank is not None and rng.random() < fraction:
      row = int(rng.integers(len(bank.states)))
      return sim.restore(bank.states[row], bank.variants[row]), True
    return sim.reset(), False

  try:
    while True:
      cmd, payload = conn.recv()
      if cmd == "reset":
        state, from_bank = start()
        conn.send((state, sim.context(), from_bank))
      elif cmd == "step":
        next_state, success, truncated = sim.step(payload)
        state, from_bank = (next_state, False)
        if truncated:
          state, from_bank = start()
        conn.send((next_state, success, truncated, state, sim.context(), from_bank))
      elif cmd == "fraction":
        fraction = float(payload)
        conn.send(None)
      elif cmd == "close":
        break
  finally:
    conn.close()


class VecEnv:
  def __init__(
    self,
    num_envs: int,
    cfg: SimCfg,
    seed: int,
    bank: StartBank | None = None,
    bank_fraction: float = 0.0,
  ):
    self.num_envs = num_envs
    ctx = spawn_context()
    self._conns, self._procs = [], []
    with single_threaded_children():
      for i in range(num_envs):
        parent, child = ctx.Pipe()
        proc = ctx.Process(
          target=_worker,
          args=(child, cfg, seed + 1000 * i, bank, bank_fraction),
          daemon=True,
        )
        proc.start()
        child.close()
        self._conns.append(parent)
        self._procs.append(proc)

  def _call(self, cmd: str, payloads) -> list:
    for conn, payload in zip(self._conns, payloads, strict=True):
      conn.send((cmd, payload))
    return [conn.recv() for conn in self._conns]

  def set_bank_fraction(self, fraction: float) -> None:
    self._call("fraction", [fraction] * self.num_envs)

  def reset(self) -> dict[str, np.ndarray]:
    """Restart every episode. Returns ``state`` ``(N, 32)``, ``ctx`` ``(N, 3)``
    and ``from_bank`` ``(N,)``."""
    out = self._call("reset", [None] * self.num_envs)
    state, ctx, from_bank = zip(*out, strict=True)
    return {
      "state": np.stack(state),
      "ctx": np.stack(ctx),
      "from_bank": np.array(from_bank),
    }

  def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
    """Step every env.

    ``next_state`` is the state the action led to (before any reset), with its
    native ``success``; ``truncated`` marks episodes that ended on this step;
    ``state`` / ``ctx`` / ``from_bank`` describe the state to act in next,
    after the automatic restart of truncated episodes.
    """
    out = self._call("step", list(np.asarray(actions, dtype=np.float64)))
    next_state, success, truncated, state, ctx, from_bank = zip(*out, strict=True)
    return {
      "next_state": np.stack(next_state),
      "success": np.array(success),
      "truncated": np.array(truncated),
      "state": np.stack(state),
      "ctx": np.stack(ctx),
      "from_bank": np.array(from_bank),
    }

  def close(self) -> None:
    for conn in self._conns:
      try:
        conn.send(("close", None))
      except (BrokenPipeError, OSError):
        pass
    for proc in self._procs:
      proc.join(timeout=5)
      if proc.is_alive():
        proc.terminate()
