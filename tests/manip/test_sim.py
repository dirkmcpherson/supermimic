"""Simulator-wrapper tests against real robosuite environments."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("robosuite")

from supermimic.manip import layout as L  # noqa: E402
from supermimic.manip.fk import SerialChainFK  # noqa: E402
from supermimic.manip.representation import (  # noqa: E402
  REP_GRASP_POS,
  REP_GRASP_POS_IN_EEF,
  REP_OBJ_FROM_GOAL,
  ArmObjectRepresentation,
)
from supermimic.manip.reward import SquareRewardCfg, square_reward_terms  # noqa: E402
from supermimic.manip.sim import (  # noqa: E402
  CUBE_HALF_SIZE_HI,
  CUBE_HALF_SIZE_LO,
  LiftSim,
  SimCfg,
  SquareSim,
  make_sim,
)
from supermimic.manip.vec_env import StartBank, VecEnv  # noqa: E402


@pytest.fixture(scope="module")
def sim() -> LiftSim:
  return make_sim(SimCfg(horizon=5), seed=0)


def _actions(count: int, seed: int = 1) -> np.ndarray:
  return np.random.default_rng(seed).uniform(-1, 1, (count, L.ACTION_DIM))


def test_timing_and_layout(sim):
  state = sim.reset()
  assert state.shape == (L.STATE_DIM,)
  assert sim.dt == 0.05 and sim.substeps == 25 and sim.grip_cmd_step == 0.25
  assert np.isclose(np.linalg.norm(state[L.OBJ_QUAT]), 1.0)
  assert state[L.GRIP_CMD] == 0.0
  # The cube starts just above the table, the fingers half open.
  assert sim.table_z < state[L.OBJ_POS][2] < sim.table_z + 0.05
  assert np.allclose(state[L.Q][7:], [0.0208, -0.0208], atol=1e-3)


def test_step_matches_robosuite_step(sim):
  actions = _actions(15)
  start = sim.reset().copy()
  for action in actions:
    ours, _, _ = sim.step(action)
  sim.restore(start)
  for action in actions:
    sim.env.step(action)
  assert np.array_equal(ours, sim.state())


def test_restore_reproduces_a_trajectory(sim):
  actions = _actions(12, seed=2)
  sim.reset()
  for action in actions[:4]:
    sim.step(action)
  middle, variant = sim.state().copy(), sim.variant.copy()
  for action in actions[4:]:
    first, _, _ = sim.step(action)
  # Other episodes in between (with another cube size) must leave no trace.
  sim.reset()
  sim.step(actions[0])
  # Exact up to rounding in the quaternion and angular-velocity frame changes.
  assert np.allclose(sim.restore(middle, variant), middle, rtol=0, atol=1e-12)
  for action in actions[4:]:
    second, _, _ = sim.step(action)
  # The contact solver's warm start is not part of the state.
  assert np.allclose(first, second, rtol=0, atol=1e-6)


def test_truncation_after_the_horizon(sim):
  sim.reset()
  flags = [sim.step(np.zeros(L.ACTION_DIM))[2] for _ in range(5)]
  assert flags == [False, False, False, False, True]


def test_resized_cube_matches_a_compiled_model():
  a, b = make_sim(SimCfg(), seed=3), make_sim(SimCfg(), seed=4)
  assert not np.allclose(a.variant, b.variant)
  b.set_variant(a.variant)
  raw_a, raw_b = a.env.sim.model._model, b.env.sim.model._model
  for name in (
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "body_mass",
    "body_inertia",
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
  ):
    assert np.allclose(getattr(raw_a, name), getattr(raw_b, name), rtol=1e-12), name


def test_task_reset_redraws_the_cube_size(sim):
  sizes = np.array([(sim.reset(), sim.variant.copy())[1] for _ in range(4)])
  assert (sizes >= CUBE_HALF_SIZE_LO).all() and (sizes <= CUBE_HALF_SIZE_HI).all()
  assert len(np.unique(sizes[:, 0])) == 4
  assert np.all(np.abs(sim.context()) <= 1.0)
  # A cube at rest sits on the table at its own half-height.
  sim.reset()
  for _ in range(5):
    state, _, _ = sim.step(np.zeros(L.ACTION_DIM))
  assert np.isclose(state[L.OBJ_POS][2] - sim.table_z, sim.variant[2], atol=5e-4)


def test_gripper_command_rules():
  close, half = np.zeros(L.ACTION_DIM), np.zeros(L.ACTION_DIM)
  close[-1], half[-1] = 1.0, 0.4
  ours = make_sim(SimCfg(), seed=5)
  stock = make_sim(SimCfg(proportional_gripper=False), seed=5)
  ours.reset()
  stock.reset()
  # Identical for a saturated command...
  assert np.isclose(ours.step(close)[0][L.GRIP_CMD][0], 0.25)
  assert np.isclose(stock.step(close)[0][L.GRIP_CMD][0], 0.25)
  # ...proportional versus sign-only in between.
  assert np.isclose(ours.step(half)[0][L.GRIP_CMD][0], 0.35)
  assert np.isclose(stock.step(half)[0][L.GRIP_CMD][0], 0.50)
  for _ in range(6):
    state, _, _ = ours.step(close)
  assert state[L.GRIP_CMD][0] == 1.0
  # Fully commanded closed with nothing between them, the fingers meet.
  assert state[L.Q][7] - state[L.Q][8] < 0.005


def test_fk_and_representation_match_the_simulator(sim):
  constants = sim.constants()
  fk = SerialChainFK(
    {k[3:]: v for k, v in constants.items() if k.startswith("fk_")}
  ).double()
  representation = ArmObjectRepresentation(constants).double()
  sim.reset()
  for action in _actions(4, seed=6):
    state, _, _ = sim.step(action)
  sim.env.sim.forward()  # positions lag the last integration step until refreshed
  data = sim.env.sim.data
  site = sim.robot.eef_site_id
  pos, rot = fk(torch.tensor(state[L.Q][:7]))
  assert np.allclose(pos.numpy(), data.site_xpos[site], atol=1e-6)
  assert np.allclose(rot.numpy(), data.site_xmat[site].reshape(3, 3), atol=1e-6)
  # The representation's cube-in-eef offset agrees with robosuite's reach distance.
  rep = representation.represent(torch.tensor(state))
  cube = data.body_xpos[sim.env.cube_body_id]
  assert np.isclose(
    rep[43:46].norm().item(), np.linalg.norm(cube - data.site_xpos[site]), atol=1e-6
  )


def test_world_angular_velocity_integrates_the_obj_quaternion(sim):
  from supermimic.manip.rotations import (
    matrix_from_quat,
    quat_integrate,
    rotation_angle,
  )

  state = sim.reset().copy()
  state[L.OBJ_POS][2] += 0.3  # in free fall, away from every contact
  state[L.OBJ_ANG_VEL] = [1.0, -2.0, 0.5]
  sim.restore(state)
  after, _, _ = sim.step(np.zeros(L.ACTION_DIM))
  predicted = quat_integrate(
    torch.tensor(state[L.OBJ_QUAT]), torch.tensor(state[L.OBJ_ANG_VEL]), sim.dt
  )
  error = rotation_angle(
    matrix_from_quat(predicted), matrix_from_quat(torch.tensor(after[L.OBJ_QUAT]))
  )
  # A torque-free near-cube keeps its world angular velocity.
  assert error < 2e-3
  assert np.allclose(after[L.OBJ_ANG_VEL], state[L.OBJ_ANG_VEL], atol=2e-2)


def test_vec_env_auto_resets_and_restores_from_the_bank():
  probe = make_sim(SimCfg(), seed=7)
  start = probe.reset().copy()
  start[L.GRIP_CMD] = 0.5
  bank = StartBank(states=start[None], variants=probe.variant[None])
  env = VecEnv(2, SimCfg(horizon=3), seed=8, bank=bank, bank_fraction=1.0)
  try:
    out = env.reset()
    assert out["state"].shape == (2, L.STATE_DIM) and out["ctx"].shape == (2, L.CTX_DIM)
    assert out["from_bank"].all()
    assert np.allclose(out["state"], start)
    for step in range(3):
      out = env.step(np.zeros((2, L.ACTION_DIM)))
      assert out["truncated"].tolist() == [step == 2] * 2
    # After the horizon the next state to act in is a fresh bank start.
    assert np.allclose(out["state"], start)
    assert not np.allclose(out["next_state"], start)
    env.set_bank_fraction(0.0)
    assert not env.reset()["from_bank"].any()
  finally:
    env.close()


# ----------------------------------------------------------------------
# Square
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def square() -> SquareSim:
  return make_sim(SimCfg(task="square", horizon=5), seed=0)


def test_square_step_matches_robosuite_step(square):
  start = square.reset()
  ours = [square.step(a)[0] for a in _actions(6, seed=11)]
  square.restore(start)
  for action, expected in zip(_actions(6, seed=11), ours, strict=True):
    square.env.step(action)
    assert np.array_equal(square.state(), expected)


def test_square_restore_reproduces_a_trajectory(square):
  square.reset()
  for action in _actions(3, seed=12):
    square.step(action)
  middle = square.state().copy()
  first = [square.step(a)[0] for a in _actions(4, seed=13)]
  square.reset()
  assert np.allclose(square.restore(middle), middle, rtol=0, atol=1e-12)
  second = [square.step(a)[0] for a in _actions(4, seed=13)]
  assert np.allclose(first, second, rtol=0, atol=1e-6)


def test_square_constants_match_the_model(square):
  constants = square.constants()
  assert str(constants["task"]) == "square"
  # The handle sits 5.4 cm from the nut's centre; the peg stands 13 cm tall.
  assert np.allclose(constants["grasp_offset"], [0.054, 0.0, 0.0], atol=1e-9)
  assert np.isclose(constants["peg_top_z"] - constants["table_z"], 0.13)
  representation = ArmObjectRepresentation(constants).double()
  square.reset()
  for action in _actions(4, seed=14):
    state, _, _ = square.step(action)
  env = square.env
  env.sim.forward()
  data = env.sim.data
  rep = representation.represent(torch.tensor(state))
  handle = data.site_xpos[env.object_site_ids[env.nut_id]]
  origin = np.array([0.0, 0.0, square.table_z])
  assert np.allclose(rep[REP_GRASP_POS].numpy(), handle - origin, atol=1e-6)
  # robosuite's reach reward measures the same grip-point-to-handle distance.
  reach = env._gripper_to_target(
    gripper=env.robots[0].gripper,
    target=env.nuts[env.nut_id].important_sites["handle"],
    target_type="site",
    return_distance=True,
  )
  assert np.isclose(rep[REP_GRASP_POS_IN_EEF].norm().item(), reach, atol=1e-6)
  peg = data.body_xpos[env.peg1_body_id]
  nut = data.body_xpos[square._obj_body]
  assert np.allclose(rep[REP_OBJ_FROM_GOAL][:2].numpy(), (nut - peg)[:2], atol=1e-6)


def test_square_success_agrees_with_the_seat_term(square):
  constants = square.constants()
  representation = ArmObjectRepresentation(constants)
  state = square.reset().copy()
  assert not square.success()
  # Put the nut over the peg, down on the table; the arm stays at home.
  state[L.OBJ_POS] = constants["goal_pos"] + np.array([0.003, -0.002, 0.01])
  state[L.OBJ_QUAT] = [1.0, 0.0, 0.0, 0.0]
  state[L.OBJ_LIN_VEL] = state[L.OBJ_ANG_VEL] = 0.0
  square.restore(state)
  assert square.success()
  terms = square_reward_terms(
    representation.represent(torch.tensor(state, dtype=torch.float32)),
    SquareRewardCfg(),
  )
  assert terms["seat"] > 0.95
  # Beside the peg at the same height it is neither.
  state[L.OBJ_POS][0] += 0.08
  square.restore(state)
  assert not square.success()
  terms = square_reward_terms(
    representation.represent(torch.tensor(state, dtype=torch.float32)),
    SquareRewardCfg(),
  )
  assert terms["seat"] < 0.01


def test_square_vec_env_reports_no_context():
  env = VecEnv(2, SimCfg(task="square", horizon=3), seed=9)
  try:
    out = env.reset()
    assert out["state"].shape == (2, L.STATE_DIM)
    assert np.array_equal(out["ctx"], np.zeros((2, L.CTX_DIM)))
    for _ in range(3):
      out = env.step(np.zeros((2, L.ACTION_DIM)))
    assert out["truncated"].all()
  finally:
    env.close()
