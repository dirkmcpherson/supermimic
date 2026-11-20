"""Single robosuite simulators exposing the full state of :mod:`layout`.

One class per task (:data:`SIMS`), all a Panda under delta operational-space
control with one free rigid object. NumPy only: this module is what the
collection worker processes import.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass

import numpy as np

from supermimic.manip import layout as L

CUBE_HALF_SIZE_LO = 0.020
CUBE_HALF_SIZE_HI = 0.022
"""robosuite draws each Lift cube half-extent uniformly from this range (metres)."""
CUBE_HALF_SIZE_MEAN = (CUBE_HALF_SIZE_LO + CUBE_HALF_SIZE_HI) / 2
CUBE_HALF_SIZE_SCALE = (CUBE_HALF_SIZE_HI - CUBE_HALF_SIZE_LO) / 2
_CUBE_SIZE = re.compile(r'<geom name="cube_g0" size="([^"]+)"')


def normalize_cube_size(half_size: np.ndarray) -> np.ndarray:
  return (np.asarray(half_size) - CUBE_HALF_SIZE_MEAN) / CUBE_HALF_SIZE_SCALE


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def _proportional_format_action(self, action):
  """Gripper command that integrates the action's value, not only its sign.

  Identical to robosuite's ``PandaGripper.format_action`` for actions of +-1
  (what teleoperated demonstrations hold), and continuous in between, so the
  command update ``g += speed * clip(a, -1, 1)`` is differentiable.
  """
  assert len(action) == self.dof
  rate = self.speed * np.clip(action, -1.0, 1.0)
  self.current_action = np.clip(
    self.current_action + np.array([-1.0, 1.0]) * rate, -1.0, 1.0
  )
  return self.current_action


@dataclass(frozen=True)
class SimCfg:
  task: str = "lift"
  """Key of :data:`SIMS`."""
  horizon: int = 100
  """Control steps before an episode is truncated."""
  proportional_gripper: bool = True
  """Integrate the gripper action's value (see ``_proportional_format_action``);
  ``False`` keeps robosuite's sign-only rule."""
  randomize_variant: bool = True
  """Redraw the per-episode model variant on every task reset, where the task
  has one (Lift: the cube's half-extents, which robosuite draws once per model
  build)."""
  control_freq: int = 20
  nominal_nullspace: bool = True
  """Pin the controller's null-space posture reference to the nominal home
  joints. robosuite sets it to each episode's noised initial joints, a
  per-episode constant that is not part of the state; ``False`` keeps that."""


class ArmObjectSim:
  """A robosuite task with a Panda and one free object.

  ``variant`` ``(3,)`` holds the model parameters that differ between episodes
  (zeros for a task without any); ``context()`` is its normalized form, the
  world model's privileged input.
  """

  ENV_NAME: str
  TASK: str

  def __init__(self, cfg: SimCfg, seed: int):
    import robosuite
    from robosuite.controllers import load_controller_config

    self.cfg = cfg
    self.rng = np.random.default_rng(seed)
    # robosuite's placement and joint-noise samplers draw from the global RNG.
    np.random.seed(seed)
    self.env = robosuite.make(
      self.ENV_NAME,
      robots="Panda",
      controller_configs=load_controller_config(default_controller="OSC_POSE"),
      has_renderer=False,
      has_offscreen_renderer=False,
      use_camera_obs=False,
      use_object_obs=False,
      control_freq=cfg.control_freq,
      horizon=10**9,
      ignore_done=True,
      hard_reset=False,
      reward_shaping=True,
    )
    self.substeps = int(round(self.env.control_timestep / self.env.model_timestep))
    self.dt = float(self.env.control_timestep)
    self.variant = np.zeros(L.CTX_DIM)
    self._bind()
    self.steps = 0

  # ------------------------------------------------------------------
  # Task hooks
  # ------------------------------------------------------------------

  def _object(self):
    """The robosuite model of the free object."""
    raise NotImplementedError

  def _task_constants(self) -> dict[str, np.ndarray]:
    return {}

  def _sample_variant(self) -> np.ndarray | None:
    return None

  def set_variant(self, variant: np.ndarray) -> None:
    """Change the model to ``variant``. Call before writing a state."""

  def context(self) -> np.ndarray:
    return np.zeros(L.CTX_DIM)

  @staticmethod
  def variant_from_model_xml(xml: str) -> np.ndarray:
    """The variant of the model a demonstration was recorded in."""
    return np.zeros(L.CTX_DIM)

  # ------------------------------------------------------------------
  # Model bindings
  # ------------------------------------------------------------------

  def _bind(self) -> None:
    env = self.env
    robot = env.robots[0]
    model = env.sim.model
    self.robot = robot
    self.gripper = robot.gripper
    if self.cfg.proportional_gripper:
      self.gripper.format_action = types.MethodType(
        _proportional_format_action, self.gripper
      )
    self._q_idx = np.array(
      list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)
    )
    self._dq_idx = np.array(
      list(robot._ref_joint_vel_indexes) + list(robot._ref_gripper_joint_vel_indexes)
    )
    obj = self._object()
    qpos0, _ = model.get_joint_qpos_addr(obj.joints[0])
    qvel0, _ = model.get_joint_qvel_addr(obj.joints[0])
    self._obj_qpos = slice(qpos0, qpos0 + 7)
    self._obj_qvel = slice(qvel0, qvel0 + 6)
    self._obj_body = int(model.body_name2id(obj.root_body))
    self.table_z = float(env.model.mujoco_arena.table_offset[2])
    self.grip_cmd_step = self.substeps * float(self.gripper.speed)

  def constants(self) -> dict[str, np.ndarray]:
    """Model constants the learned model and reward need."""
    from supermimic.manip.fk import chain_from_mujoco

    model = self.env.sim.model._model
    names = [f"robot0_joint{i}" for i in range(1, L.NUM_ARM + 1)]
    joint_ids = [self.env.sim.model.joint_name2id(n) for n in names]
    chain = chain_from_mujoco(model, self.robot.eef_site_id, joint_ids)
    return {
      **{f"fk_{k}": v for k, v in chain.items()},
      "task": np.array(self.TASK),
      "table_z": np.array(self.table_z),
      "dt": np.array(self.dt),
      "substeps": np.array(self.substeps),
      "grip_cmd_step": np.array(self.grip_cmd_step),
      **self._task_constants(),
    }

  def flat_layout(self) -> dict:
    """Where this wrapper's state sits in robomimic's flattened MuJoCo state
    ``[time, qpos(nq), qvel(nv)]``."""
    model = self.env.sim.model
    return {
      "nq": int(model.nq),
      "nv": int(model.nv),
      "q_idx": self._q_idx.copy(),
      "dq_idx": self._dq_idx.copy(),
      "obj_qpos": self._obj_qpos.start,
      "obj_qvel": self._obj_qvel.start,
    }

  # ------------------------------------------------------------------
  # State
  # ------------------------------------------------------------------

  def state(self) -> np.ndarray:
    data = self.env.sim.data
    obj_qpos = data.qpos[self._obj_qpos]
    obj_qvel = data.qvel[self._obj_qvel]
    quat = obj_qpos[3:7] / np.linalg.norm(obj_qpos[3:7])
    out = np.empty(L.STATE_DIM)
    out[L.Q] = data.qpos[self._q_idx]
    out[L.DQ] = data.qvel[self._dq_idx]
    out[L.OBJ_POS] = obj_qpos[:3]
    out[L.OBJ_QUAT] = quat
    out[L.OBJ_LIN_VEL] = obj_qvel[:3]
    # MuJoCo keeps a free joint's angular velocity in the body frame.
    out[L.OBJ_ANG_VEL] = _quat_to_mat(quat) @ obj_qvel[3:6]
    # Shape (1,) after a reset, (2,) = [-g, g] once a command has been applied.
    out[L.GRIP_CMD] = self.gripper.current_action[-1]
    return out

  def _write_state(self, state: np.ndarray) -> None:
    sim = self.env.sim
    quat = state[L.OBJ_QUAT] / np.linalg.norm(state[L.OBJ_QUAT])
    sim.data.qpos[self._q_idx] = state[L.Q]
    sim.data.qvel[self._dq_idx] = state[L.DQ]
    sim.data.qpos[self._obj_qpos] = np.concatenate([state[L.OBJ_POS], quat])
    sim.data.qvel[self._obj_qvel] = np.concatenate(
      [state[L.OBJ_LIN_VEL], _quat_to_mat(quat).T @ state[L.OBJ_ANG_VEL]]
    )
    sim.data.qacc_warmstart[:] = 0.0
    sim.forward()
    grip = float(state[L.GRIP_CMD][0])
    self.gripper.current_action = np.array([-grip, grip])

  def _set_nullspace_reference(self, joints: np.ndarray | None = None) -> None:
    """Reset the controller's posture reference (and its goal to the current pose)."""
    if joints is None:
      if not self.cfg.nominal_nullspace:
        return
      joints = self.robot.init_qpos
    self.robot.controller.update_initial_joints(np.array(joints, dtype=np.float64))

  def success(self) -> bool:
    """robosuite's own success test for the task."""
    return bool(self.env._check_success())

  # ------------------------------------------------------------------
  # Episode control
  # ------------------------------------------------------------------

  def reset(self) -> np.ndarray:
    """Standard task reset: arm near home, object placed at random."""
    if self.cfg.randomize_variant:
      variant = self._sample_variant()
      if variant is not None:
        self.set_variant(variant)
    self.env.reset()
    self._set_nullspace_reference()
    self.steps = 0
    return self.state()

  def restore(
    self,
    state: np.ndarray,
    variant: np.ndarray | None = None,
    nullspace_ref: np.ndarray | None = None,
  ) -> np.ndarray:
    """Start an episode at ``state``, in the model ``variant`` (default: keep
    the current one). ``nullspace_ref`` overrides the controller's posture
    reference (default: per ``cfg.nominal_nullspace``, else the arm joints of
    ``state``).
    """
    state = np.array(state, dtype=np.float64)
    if variant is not None:
      self.set_variant(variant)
    self.env.reset()
    self._write_state(state)
    if nullspace_ref is None and not self.cfg.nominal_nullspace:
      nullspace_ref = state[L.Q][: L.NUM_ARM]
    self._set_nullspace_reference(nullspace_ref)
    self.steps = 0
    return self.state()

  def _advance(self, action: np.ndarray) -> None:
    """One control step: robosuite's ``step`` loop without its observables."""
    env = self.env
    policy_step = True
    for _ in range(self.substeps):
      env.sim.forward()
      env._pre_action(action, policy_step)
      env.sim.step()
      policy_step = False

  def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    """Apply ``action``; returns ``(next_state, success, truncated)``."""
    self._advance(np.clip(action, -1.0, 1.0))
    self.steps += 1
    return self.state(), self.success(), self.steps >= self.cfg.horizon


class LiftSim(ArmObjectSim):
  """robosuite ``Lift``. The variant is the cube's half-extents (metres)."""

  ENV_NAME = "Lift"
  TASK = "lift"

  def _object(self):
    return self.env.cube

  def _bind(self) -> None:
    super()._bind()
    model = self.env.sim.model
    self._cube_geoms = [model.geom_name2id(n) for n in ("cube_g0", "cube_g0_vis")]
    raw = model._model
    size = np.array(raw.geom_size[self._cube_geoms[0]], dtype=np.float64)
    self._cube_density = float(raw.body_mass[self._obj_body] / (8 * size.prod()))
    self.variant = size

  def _sample_variant(self) -> np.ndarray:
    return self.rng.uniform(CUBE_HALF_SIZE_LO, CUBE_HALF_SIZE_HI, 3)

  def set_variant(self, variant: np.ndarray) -> None:
    """Resize the cube in place: geometry, bounds, mass and inertia, as a model
    compiled with that size would have them."""
    import mujoco

    half_size = np.array(variant, dtype=np.float64)
    sim = self.env.sim
    raw = sim.model._model
    x, y, z = half_size
    mass = self._cube_density * 8 * x * y * z
    for geom in self._cube_geoms:
      raw.geom_size[geom] = half_size
      raw.geom_rbound[geom] = np.linalg.norm(half_size)
      raw.geom_aabb[geom, 3:] = half_size
    raw.body_mass[self._obj_body] = mass
    raw.body_inertia[self._obj_body] = (
      mass / 3 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    )
    # Recompute the constants derived from mass (subtree mass, inverse weights).
    mujoco.mj_setConst(raw, sim.data._data)
    self.variant = half_size

  def context(self) -> np.ndarray:
    return normalize_cube_size(self.variant)

  @staticmethod
  def variant_from_model_xml(xml: str) -> np.ndarray:
    match = _CUBE_SIZE.search(xml)
    if match is None:
      raise ValueError("no cube geom in the stored model")
    return np.array([float(v) for v in match.group(1).split()])


class SquareSim(ArmObjectSim):
  """robosuite ``NutAssemblySquare``: fit the square nut over the square peg.

  The model also holds the round nut, which robosuite parks far from the
  workspace; it is not part of the state. Nothing varies between episodes but
  the start state, so the variant stays zero.
  """

  ENV_NAME = "NutAssemblySquare"
  TASK = "square"

  def _object(self):
    return self.env.nuts[self.env.nut_id]

  def _task_constants(self) -> dict[str, np.ndarray]:
    env = self.env
    model, data = env.sim.model, env.sim.data
    env.sim.forward()
    # The handle (where the nut is grasped), in the nut's body frame.
    site = model.site_name2id(self._object().important_sites["handle"])
    rot = data.body_xmat[self._obj_body].reshape(3, 3)
    grasp_offset = rot.T @ (data.site_xpos[site] - data.body_xpos[self._obj_body])
    peg = env.peg1_body_id
    peg_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == peg)
    peg_pos = np.array(data.body_xpos[peg])
    peg_half = np.array(model._model.geom_size[peg_geom])
    return {
      "grasp_offset": grasp_offset,
      # Peg axis at table height: where the nut's centre ends up, in x and y.
      "goal_pos": np.array([peg_pos[0], peg_pos[1], self.table_z]),
      "peg_top_z": np.array(peg_pos[2] + peg_half[2]),
      "peg_half_width": np.array(peg_half[0]),
    }


SIMS: dict[str, type[ArmObjectSim]] = {"lift": LiftSim, "square": SquareSim}


def make_sim(cfg: SimCfg, seed: int) -> ArmObjectSim:
  return SIMS[cfg.task](cfg, seed)
