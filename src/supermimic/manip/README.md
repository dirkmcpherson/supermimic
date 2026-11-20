# SuperMimic on robosuite manipulation tasks

A test of this repo's learning machinery on fixed-base manipulation tasks, at
single-desktop scale. The world-model learner, the actor-critic learner, the
replay buffer and the dynamics model are the repo's own; this package adds what
a manipulation task needs around them. Nothing here imports mjlab.

Tasks (`sim.py`, `tasks.py`, `reward.py`): `lift` (robosuite `Lift`) and
`square` (robosuite `NutAssemblySquare`), both a Panda under delta
operational-space control at 20 Hz with one free object.

## Setup

The tasks run in robosuite 1.4.1 (CPU MuJoCo), in their own environment:

```bash
python3.12 -m venv .venv-lift
.venv-lift/bin/pip install torch==2.10.0 tensordict==0.11.0 rsl-rl-lib==5.4.0 \
  numpy==1.26.4 mujoco==3.1.6 robosuite==1.4.1 numba h5py pytest ruff
```

Demonstrations are robomimic-format datasets (`low_dim_v141_<task>_ph.hdf5`,
`low_dim_v141_<task>_mh.hdf5`).

## Running

```bash
scripts/run_manip_overnight.sh square     # everything, unattended; see its header
```

or stage by stage (`PYTHONPATH=src .venv-lift/bin/python -m ...`):

| Stage | Module | Output |
| --- | --- | --- |
| 0 | `supermimic.manip.scripts.prepare_data` | recorded and replayed demos, exploration branches, start-state bank |
| 1 | `supermimic.manip.scripts.train_wm` | `world_model.pt`, open-loop errors on held-out data (`wm_eval.json`) |
| baseline | `supermimic.manip.scripts.train_bc` | behaviour cloning of the demos with the policy's actor (`bc_report.json`) |
| 2 | `supermimic.manip.scripts.train_policy` | checkpoints, `policy_eval.jsonl`, `policy_report.json` |
| - | `supermimic.manip.scripts.summarize` | `REPORT.md` |

Stage 2 controls: `--bank-fraction 0 --start-mix 1 0 0` never starts an episode
from a demo state; adding `--wm scratch --wm-mix 1 0 0` uses no offline data at
all.

Tests: `PYTHONPATH=src .venv-lift/bin/python -m pytest tests/manip --confcutdir=tests/manip`

## What is different from the humanoid setup

- **State.** Arm and finger joints, the object's world pose and twist, and the
  gripper's integrated command (controller memory that MuJoCo's state does not
  hold). The object takes the floating base's place as the `aux_state`
  (`representation.py`).
- **Prediction.** Velocity deltas integrated as in the humanoid
  representations, plus pose residuals: a 20 Hz control step spans 25 physics
  substeps, so contact changes the pose by more than the velocity change
  explains.
- **Context.** Lift: the cube's half-extents, which robosuite randomizes, in
  place of the humanoid's randomized friction, centre of mass and encoder
  bias. Square has nothing of the kind (zeros).
- **Reward.** Analytic and differentiable, on the predicted state
  (`reward.py`): robosuite's shaped reward for the task, changed where that is
  not a smooth function of the state (see the module docstring). Success is
  always robosuite's own test.
- **Data.** Three replay buffers: the policy's own rollouts, replayed demos,
  and exploration branches (noisy rollouts restarted from demo states). Each
  world-model batch and each batch of imagination start states mixes them.

## Departures from stock robosuite

Both are off in the `final_stock_env` evaluation.

- The gripper command integrates the action's value, not only its sign, so it
  is a differentiable function of the action. The two rules agree for the +-1
  commands demonstrations contain.
- The controller's null-space posture reference is pinned to the nominal home
  joints. robosuite sets it to each episode's noised initial joints, a
  per-episode constant that is not part of the state. (The demonstrations
  replay exactly only with the pinned reference, so they appear to have been
  recorded that way.)

## Demonstrations

Each demo's first recorded transition is dropped (`demos.FIRST_FRAME`): it is
not one the simulator reproduces, and an open-loop replay with delta actions
never recovers from it. Replayed from the second frame, Lift demos reproduce
bit for bit; 94% of the Square demos still succeed, the nut within 2.3 mm of
the recording at the median.
