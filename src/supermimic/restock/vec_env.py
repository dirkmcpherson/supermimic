"""A pool of simulator worker processes, stepped together.

Each worker (:mod:`supermimic.restock.sim_worker`) is started with the
simulator's own Python interpreter, which differs from the learner's.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

from supermimic.restock import layout as L

SRC_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SimCfg:
  repo: str
  """The task repository to import the environment from (never written to)."""
  python: str
  """Interpreter of the environment the simulator runs in."""
  log_dir: str | None = None
  """Where each worker's output goes (default: discarded)."""


class WorkerPool:
  """``num`` simulators. ``call`` sends one request to each of the given
  workers and returns the answers in order; the workers run concurrently."""

  def __init__(self, num: int, cfg: SimCfg, seed: int):
    self.num = num
    self._dir = tempfile.TemporaryDirectory(prefix="restock_")
    address = os.path.join(self._dir.name, "sock")
    authkey = secrets.token_bytes(16)
    listener = Listener(address, family="AF_UNIX", authkey=authkey)
    env = dict(os.environ)
    env.update(
      PYTHONPATH=str(SRC_ROOT),
      PYTHONUNBUFFERED="1",
      PYTHONDONTWRITEBYTECODE="1",
      MUJOCO_GL="egl",
      OMP_NUM_THREADS="1",
      MKL_NUM_THREADS="1",
    )
    self._procs, self._logs = [], []
    for i in range(num):
      log = subprocess.DEVNULL
      if cfg.log_dir is not None:
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        log = open(Path(cfg.log_dir) / f"worker_{seed}_{i}.log", "w")  # noqa: SIM115
        self._logs.append(log)
      command = [
        cfg.python,
        "-m",
        "supermimic.restock.sim_worker",
        *("--address", address),
        *("--authkey", authkey.hex()),
        *("--repo", cfg.repo),
        *("--seed", str(seed + i)),
      ]
      self._procs.append(
        subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
      )
    # Connections arrive in any order; a worker is whichever connection it got.
    self._conns = [listener.accept() for _ in range(num)]
    listener.close()
    self.build_seconds = [self._receive(c)["build_seconds"] for c in self._conns]

  @staticmethod
  def _receive(conn):
    status, value = conn.recv()
    if status != "ok":
      raise RuntimeError(f"simulator worker failed:\n{value}")
    return value

  def call(
    self, name: str, args_per_worker: list[tuple], workers: list[int] | None = None
  ):
    workers = list(range(self.num)) if workers is None else workers
    for i, args in zip(workers, args_per_worker, strict=True):
      self._conns[i].send((name, args, {}))
    return [self._receive(self._conns[i]) for i in workers]

  def call_one(self, worker: int, name: str, *args, **kwargs):
    self._conns[worker].send((name, args, kwargs))
    return self._receive(self._conns[worker])

  def constants(self) -> dict[str, np.ndarray]:
    return self.call_one(0, "constants")

  def close(self) -> None:
    for conn in self._conns:
      try:
        conn.send(("close", (), {}))
        conn.recv()
      except (OSError, EOFError):
        pass
      conn.close()
    for proc in self._procs:
      try:
        proc.wait(timeout=30)
      except subprocess.TimeoutExpired:
        proc.kill()
    for log in self._logs:
      log.close()
    self._dir.cleanup()


class VecEnv:
  """Training environments: every worker runs episode after episode from the
  task's own training starts."""

  def __init__(self, pool: WorkerPool):
    self.pool = pool
    self.num_envs = pool.num

  def set_demo_starts(
    self, paths: list[str], lengths: list[int], fraction: float
  ) -> None:
    self.pool.call("set_demo_starts", [(paths, lengths, fraction)] * self.num_envs)

  def reset(self) -> tuple[np.ndarray, np.ndarray]:
    """Start a training episode in every env. Returns the states and the
    decisions spent reaching each (0 for a task start)."""
    outs = self.pool.call("training_start", [()] * self.num_envs)
    return np.stack([state for state, _ in outs]), np.array(
      [steps for _, steps in outs]
    )

  def step(
    self, actions: np.ndarray, workers: list[int] | None = None
  ) -> dict[str, np.ndarray]:
    """Step ``workers`` (default: all) with one action each; every entry of
    the result has one row per stepped worker.

    ``next_state`` is the state the action led to, ``state`` the one the next
    action is chosen in (a fresh start where an episode ended), ``stages`` the
    outcome of the episodes that ended, ``from_demo`` whether those had started
    along a demonstration, ``prefix_decisions`` the decisions simulated to
    reach the new starts."""
    outs = self.pool.call("step_auto", [(a,) for a in actions], workers)
    stages = np.zeros((len(outs), len(L.STAGES)), dtype=bool)
    for i, out in enumerate(outs):
      if "stages" in out:
        stages[i] = out["stages"]
    return {
      "next_state": np.stack([o["next_state"] for o in outs]),
      "state": np.stack([o["state"] for o in outs]),
      "terminated": np.array([o["terminated"] for o in outs]),
      "truncated": np.array([o["truncated"] for o in outs]),
      "reward": np.array([o["reward"] for o in outs]),
      "from_demo": np.array([o["from_demo"] for o in outs]),
      "prefix_decisions": np.array([o["prefix_decisions"] for o in outs]),
      "stages": stages,
    }
