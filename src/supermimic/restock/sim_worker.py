"""One restock simulator in its own process.

Runs under the task's own simulator environment (Genesis
allows one world per process, and that environment has no recent torch), so
this module imports NumPy and the task's own code only. The parent talks to it
over a :mod:`multiprocessing.connection` socket; see
:class:`supermimic.restock.vec_env.WorkerPool`.

The environment is built by the task repository's own constructor
(``baselines/rl/relabel_reward.py::build_env``) with the task's standard
settings; nothing in that repository is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Client

import numpy as np

from supermimic.restock import layout as L
from supermimic.restock.kinematics import chain_from_link_poses

SIM_VARIANT = "gc_kp4_riser3_shelf6"
LADDER = "nested_sparse10"
TIP_GUARD = "not_in_hand"
FRAMES_PER_EPISODE = 1200
ACTION_REPEAT = 4
DELTA_CAP = 0.025


def _np(value) -> np.ndarray:
  if hasattr(value, "detach"):
    value = value.detach().cpu().numpy()
  return np.asarray(value, dtype=np.float64).reshape(-1)


class RestockSim:
  def __init__(self, repo: str, seed: int):
    os.environ["GENESIS_PICKAPLACE_ROOT"] = repo
    os.environ["GENESIS_SIM_VARIANT"] = SIM_VARIANT
    for sub in ("baselines/rl", "baselines", "can_pos_recovery"):
      sys.path.insert(0, os.path.join(repo, sub))
    import relabel_reward

    self._task = relabel_reward
    self.env = relabel_reward.build_env(
      SIM_VARIANT, FRAMES_PER_EPISODE, LADDER, far_release=False, tip_guard=TIP_GUARD
    )
    env = self.env
    assert env.tracker is not None and env.terminal_stages == ("home",)
    assert env.action_repeat == ACTION_REPEAT and abs(env.delta_cap - DELTA_CAP) < 1e-12
    assert env.tip_guard == TIP_GUARD and env.max_steps == FRAMES_PER_EPISODE
    self.genv = env.genv
    self.world = self.genv.w
    self.rng = np.random.default_rng(seed)
    self.uids = [int(u) for u in env.success_uids]
    self._grip_cmd = 0.0
    self._demo_starts: tuple[list[str], list[int], float] | None = None
    self._tape_actions: dict[str, np.ndarray] = {}
    self._from_demo = False
    self.reset()

  # ------------------------------------------------------------------
  # Constants of the built world
  # ------------------------------------------------------------------

  def constants(self) -> dict[str, np.ndarray]:
    import full_env
    import replay_harness
    import torch

    world, env = self.world, self.env
    kin = world["kinova"]
    names = [link.name for link in kin.links]
    chain_links = [names.index(n) for n in names[1 : 1 + L.NUM_ARM]]
    assert world["eef"].idx_local == chain_links[-1], "tool is not on the last arm link"
    rng = np.random.default_rng(0)
    q0 = _np(kin.get_dofs_position())
    qs, poss, quats = [], [], []
    for _ in range(48):
      q = q0.copy()
      q[: L.NUM_ARM] = rng.uniform(full_env.ARM_LO, full_env.ARM_HI)
      # The entity's forward kinematics is a pure function; the world is untouched.
      pos, quat = kin.forward_kinematics(qpos=torch.as_tensor(q, dtype=torch.float32))
      qs.append(q[: L.NUM_ARM])
      poss.append(pos.detach().cpu().numpy())
      quats.append(quat.detach().cpu().numpy())
    if self.genv._tool_offset is None:
      self.genv._calib_tool_offset()
    chain = chain_from_link_poses(
      np.stack(qs), np.stack(poss), np.stack(quats), chain_links, self.genv._tool_offset
    )
    box_pos, box_size = replay_harness.BOX_POS, replay_harness.BOX_SIZE
    tracker = env.tracker
    scene = world["scene"]
    frame_dt = 3 * float(scene.dt)
    return {
      **{f"fk_{k}": v for k, v in chain.items()},
      "dt": np.array(env.action_repeat * frame_dt),
      "substeps": np.array(env.action_repeat * 3 * int(scene.substeps)),
      "arm_lo": np.asarray(full_env.ARM_LO, dtype=np.float64),
      "arm_hi": np.asarray(full_env.ARM_HI, dtype=np.float64),
      "target_step": np.array(env.action_repeat * env.delta_cap),
      "leash": np.array(env.delta_leash),
      "shelf_top_z": np.array(env.shelf_top_z),
      "shelf_lo": np.array([box_pos[i] - box_size[i] / 2 for i in (0, 1)]),
      "shelf_hi": np.array([box_pos[i] + box_size[i] / 2 for i in (0, 1)]),
      "pick_z": np.array(float(world["pick_z"])),
      "pick_grip": np.array(replay_harness.GP_CLOSE / 100.0),
      "pick_wrist_dist": np.array(_pick_eef_dist()),
      "place_release": np.array(env.PLACE_RELEASE),
      "band_lo": np.array(tracker.band_lo_m),
      "band_hi": np.array(tracker.band_hi_m),
      "tilt_max_deg": np.array(tracker.tilt_max_deg),
      "tip_deg": np.array(env.TIP_DEG),
      "held_lever": np.array(tracker.held_lever_m),
      "at_rest_m": np.array(tracker.at_rest_m),
      "at_rest_seconds": np.array(tracker.at_rest_frames * frame_dt),
      "farside_reach": np.array(tracker.farside_reach_m),
      "farside_cone_deg": np.array(tracker.farside_cone_deg),
      "slide_gain_min": np.array(tracker.slide_gain_min_m),
      "nested_touch_dist": np.array(tracker.nested_touch_dist),
      "home_reward": np.array(float(env.stage_reward["home"])),
      "episode_decisions": np.array(env.max_steps // env.action_repeat),
      "uids": np.asarray(self.uids, dtype=np.int64),
      "sustain_frames": np.array(
        [_pick_sustain(), env.PLACE_SUSTAIN, env.tip_guard_sustain]
      ),
      "sim_variant": np.array(SIM_VARIANT),
      "ladder": np.array(env.ladder),
      "tip_guard": np.array(env.tip_guard),
      "task_code": np.array(json.dumps(_code_stamp())),
    }

  # ------------------------------------------------------------------
  # State
  # ------------------------------------------------------------------

  def state(self) -> np.ndarray:
    world, env = self.world, self.env
    kin, can, goal, tracker = (
      world["kinova"],
      world["bottle"],
      world["goal"],
      env.tracker,
    )
    out = np.empty(L.STATE_DIM)
    out[L.Q] = _np(kin.get_dofs_position())
    out[L.DQ] = _np(kin.get_dofs_velocity())
    for body, pos, quat, lin, ang in (
      (can, L.CAN_POS, L.CAN_QUAT, L.CAN_LIN_VEL, L.CAN_ANG_VEL),
      (goal, L.GOAL_POS, L.GOAL_QUAT, L.GOAL_LIN_VEL, L.GOAL_ANG_VEL),
    ):
      out[pos] = _np(body.get_pos())
      out[quat] = _np(body.get_quat())
      out[lin] = _np(body.get_vel())
      out[ang] = _np(body.get_ang())
    out[L.TARGET] = env._dj_target
    out[L.GRIP_CMD] = self._grip_cmd
    out[L.FLAGS] = [
      self.genv._picked,
      tracker.released,
      tracker.settled_after_release,
      tracker.farside,
      tracker.slide_event,
    ]
    out[L.SLIDE_GAIN] = tracker.slide_gain_m
    if tracker._slide_min is None:
      out[L.SLIDE_MIN] = np.hypot(*(out[L.CAN_POS][:2] - out[L.GOAL_POS][:2]))
    else:
      out[L.SLIDE_MIN] = tracker._slide_min
    return out.astype(np.float32)

  def reset(self, uid: int | None = None, ic: dict | None = None) -> np.ndarray:
    """Start an episode from a training start (``uid``; random when ``None``)
    or from an explicit pose (``ic``, the evaluator's convention: the inner
    reset is given the pose, every other piece of bookkeeping is the
    environment's own)."""
    env = self.env
    if ic is None:
      if uid is None:
        uid = int(self.rng.choice(self.uids))
      env.reset(options={"uid": int(uid)})
    else:
      inner = self.genv.reset
      self.genv.reset = lambda *a, **kw: inner(**ic)
      try:
        env.reset(options={"uid": self.uids[0]})
      finally:
        self.genv.reset = inner
    self._grip_cmd = 0.0
    return self.state()

  def step(self, action: np.ndarray) -> dict:
    action = np.asarray(action, dtype=np.float32)
    _, reward, terminated, truncated, info = self.env.step(action)
    self._grip_cmd = float((np.clip(action[6], -1.0, 1.0) + 1.0) / 2.0)
    out = {
      "next_state": self.state(),
      "reward": float(reward),
      "terminated": bool(terminated),
      "truncated": bool(truncated),
    }
    if terminated or truncated:
      granted = set(self.env._granted)
      out["stages"] = np.array(
        [bool(s in granted or info.get(s)) for s in L.STAGES], dtype=bool
      )
    return out

  def step_auto(self, action: np.ndarray) -> dict:
    """``step`` that starts the next training episode when one ends.

    ``from_demo`` says whether the episode the action belonged to had started
    part-way along a demonstration, ``prefix_decisions`` how many decisions
    were simulated to reach the next episode's start (0 for a task start)."""
    out = self.step(action)
    out["from_demo"] = self._from_demo
    out["prefix_decisions"] = 0
    if out["terminated"] or out["truncated"]:
      out["state"], out["prefix_decisions"] = self.training_start()
    else:
      out["state"] = out["next_state"]
    return out

  def set_demo_starts(
    self, paths: list[str], lengths: list[int], fraction: float
  ) -> None:
    """Start ``fraction`` of the training episodes part-way along a tape:
    tape ``i`` at a uniformly drawn decision below ``lengths[i]``."""
    self._demo_starts = (list(paths), [int(n) for n in lengths], float(fraction))

  def training_start(self) -> tuple[np.ndarray, int]:
    """Begin a training episode; returns its first state and the decisions
    spent reaching it."""
    self._from_demo = False
    if self._demo_starts is not None:
      paths, lengths, fraction = self._demo_starts
      if self.rng.random() < fraction:
        # Every (tape, decision) pair is equally likely.
        tape = int(self.rng.choice(len(paths), p=np.asarray(lengths) / sum(lengths)))
        steps = int(self.rng.integers(0, lengths[tape]))
        state = self._replay_prefix(paths[tape], steps)
        if state is not None:
          self._from_demo = True
          return state, steps
        # The tape ended early this time; a task start, the prefix still paid.
        return self.reset(), steps
    return self.reset(), 0

  def _replay_prefix(self, path: str, steps: int) -> np.ndarray | None:
    """Reach a demonstration state the only way there is one: from the tape's
    start, by its own actions. The episode's clock restarts there. ``None`` if
    the environment ended the episode on the way."""
    task, env = self._task, self.env
    tape = np.load(path, allow_pickle=True)
    if path not in self._tape_actions:
      self._tape_actions[path] = task.tape_stream(tape, task.tape_layout(tape))[1]
    task.reset_to_tape_ic(env, tape)
    self._grip_cmd = 0.0
    limit, env.max_steps = env.max_steps, 10**9
    try:
      for action in self._tape_actions[path][:steps]:
        if self.step(action)["terminated"]:
          return None
    finally:
      env.max_steps = limit
    env._t = 0
    return self.state()

  # ------------------------------------------------------------------
  # Demonstration tapes
  # ------------------------------------------------------------------

  def replay(self, path: str) -> dict:
    """Re-execute a recorded tape from its start and return the full states.

    A tape stores the 17 observed numbers only; the full state exists nowhere
    but in a simulator that runs the tape's actions again. The tape runs to its
    own end or to the environment's terminal (home, or a tipped can), never to
    the episode time limit."""
    task, env = self._task, self.env
    tape = np.load(path, allow_pickle=True)
    _, actions, _, want = task.tape_stream(tape, task.tape_layout(tape))
    how, can_gap, goal_gap = task.reset_to_tape_ic(env, tape)
    self._grip_cmd = 0.0
    limit, env.max_steps = env.max_steps, 10**9
    states, flags = [self.state()], []
    can_dev = joint_dev = 0.0
    stages = np.zeros(len(L.STAGES), dtype=bool)
    try:
      for t, action in enumerate(actions):
        step = self.step(action)
        states.append(step["next_state"])
        flags.append(step["terminated"])
        if want is not None and t < len(want):
          got = step["next_state"]
          can_dev = max(can_dev, float(np.linalg.norm(got[L.CAN_POS] - want[t][8:11])))
          joint_dev = max(
            joint_dev, float(np.abs(got[L.Q][: L.NUM_ARM] - want[t][:6]).max())
          )
        if step["terminated"]:
          stages = step["stages"]
          break
      else:
        granted = set(env._granted)
        stages = np.array([s in granted for s in L.STAGES], dtype=bool)
    finally:
      env.max_steps = limit
    steps = len(flags)

    def tape_scalar(key: str):
      return tape[key].item() if key in tape.files else None

    return {
      "name": os.path.basename(path),
      "states": np.stack(states[:-1]),
      "next_states": np.stack(states[1:]),
      "actions": np.asarray(actions[:steps], dtype=np.float32),
      "terminated": np.asarray(flags, dtype=bool),
      "stages": stages,
      "tape_decisions": len(actions),
      "start": how,
      "start_gap_m": max(can_gap, goal_gap),
      "can_dev_m": can_dev,
      "joint_dev_rad": joint_dev,
      "tape_end_reason": tape_scalar("rz_end_reason"),
      "tape_end_decision": tape_scalar("rz_end_decision"),
    }

  # ------------------------------------------------------------------
  # Scoring, the task evaluator's way
  # ------------------------------------------------------------------

  def finish_episode(self) -> None:
    """The settle the task's evaluator runs after every episode (it steps
    the world, so a sequence of episodes scored in one process sees it)."""
    self.genv.end_of_episode()


def _pick_eef_dist() -> float:
  import genesis_can_env

  return float(genesis_can_env.PICK_EEF_DIST)


def _pick_sustain() -> int:
  import genesis_can_env

  return int(genesis_can_env.PICK_SUSTAIN)


def _code_stamp() -> dict:
  """First 12 hex digits of the SHA-256 of the task files that define the MDP."""
  import hashlib

  import full_env
  import genesis_can_env
  import stage_predicates

  def digest(module) -> str:
    with open(module.__file__, "rb") as f:
      return hashlib.sha256(f.read()).hexdigest()[:12]

  return {
    "full_env": digest(full_env),
    "genesis_can_env": digest(genesis_can_env),
    "stage_predicates": digest(stage_predicates),
  }


def serve(sim: RestockSim, conn) -> None:
  handlers = {
    "constants": sim.constants,
    "state": sim.state,
    "reset": sim.reset,
    "step": sim.step,
    "step_auto": sim.step_auto,
    "set_demo_starts": sim.set_demo_starts,
    "training_start": sim.training_start,
    "replay": sim.replay,
    "finish_episode": sim.finish_episode,
  }
  while True:
    try:
      name, args, kwargs = conn.recv()
    except EOFError:
      return
    if name == "close":
      conn.send(("ok", None))
      return
    try:
      conn.send(("ok", handlers[name](*args, **kwargs)))
    except Exception:  # noqa: BLE001 - reported to the parent, which raises
      conn.send(("error", traceback.format_exc()))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--address", required=True)
  parser.add_argument("--authkey", required=True)
  parser.add_argument("--repo", required=True, help="task repository (read only)")
  parser.add_argument("--seed", type=int, required=True)
  args = parser.parse_args()

  conn = Client(args.address, family="AF_UNIX", authkey=bytes.fromhex(args.authkey))
  start = time.time()
  try:
    # The task code prints its build banners; keep them out of the parent's way.
    sim = RestockSim(args.repo, args.seed)
  except Exception:  # noqa: BLE001
    conn.send(("error", traceback.format_exc()))
    raise
  conn.send(("ok", {"build_seconds": time.time() - start, "pid": os.getpid()}))
  serve(sim, conn)


if __name__ == "__main__":
  main()
