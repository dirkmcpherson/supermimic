"""Column layout of the restock full state, shared by the simulator worker
(NumPy) and the torch side. Nothing here may import torch.

Full state (60), read once per decision (4 env frames, 0.12 s):
  ``[q(10), dq(10), can_pos(3), can_quat(4), can_lin_vel(3), can_ang_vel(3),
  goal_pos(3), goal_quat(4), goal_lin_vel(3), goal_ang_vel(3), target(6),
  grip_cmd(1), flags(5), slide_gain(1), slide_min(1)]``

``q``/``dq`` are the six arm joints followed by the four finger joints, in the
robot's own degree-of-freedom order. Poses and twists are in the world frame
(quaternions ``wxyz``). The rest is memory the environment keeps outside the
physics state:

``target``      the persistent joint target the delta actions integrate onto
``grip_cmd``    the gripper command of the last decision, 0 (open) to 1
``flags``       the environment's own latches, 0 or 1 (see ``FLAG_NAMES``)
``slide_gain``  metres the can has been pushed goalward from behind since it
                was set down (the stage tracker's ``slide_gain_m``)
``slide_min``   smallest can-to-goal distance since the release; before the
                release, the current distance
"""

from __future__ import annotations

NUM_ARM = 6
NUM_FINGER = 4
NUM_JOINTS = NUM_ARM + NUM_FINGER

Q = slice(0, 10)
DQ = slice(10, 20)
CAN_POS = slice(20, 23)
CAN_QUAT = slice(23, 27)
CAN_LIN_VEL = slice(27, 30)
CAN_ANG_VEL = slice(30, 33)
GOAL_POS = slice(33, 36)
GOAL_QUAT = slice(36, 40)
GOAL_LIN_VEL = slice(40, 43)
GOAL_ANG_VEL = slice(43, 46)
TARGET = slice(46, 52)
GRIP_CMD = slice(52, 53)
FLAGS = slice(53, 58)
SLIDE_GAIN = slice(58, 59)
SLIDE_MIN = slice(59, 60)

FLAG_NAMES = ("picked", "released", "settled", "farside", "slide_event")
"""``picked``: the can was held in the air for 10 frames. ``released``: after
that, it stood on the shelf with the gripper open for 10 frames. ``settled``:
it has been at rest since. ``farside``: the tool has been behind the freed can,
within reach. ``slide_event``: the can was pushed at least 1 cm goalward from
there. The episode ends with ``home`` when a slide event has happened and the
can rests against the goal can."""

STATE_DIM = 60
ACTION_DIM = 7
CTX_DIM = 1
"""Unused per-episode context (always zero); the learners expect a column."""

STAGES = ("picked", "placed_v2", "farside", "slide_event", "home", "tipped")
"""Episode outcomes the worker reports for logging."""
