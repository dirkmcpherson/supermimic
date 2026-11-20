"""Column layout of the Lift full state, shared by the simulator and torch sides.

Full state (32):
  ``[q(9), dq(9), obj_pos(3), obj_quat(4), obj_lin_vel(3), obj_ang_vel(3),
  grip_cmd(1)]``

``q``/``dq`` are the seven arm joints followed by the two finger joints. The
cube pose and twist are in the world frame (quaternion ``wxyz``; MuJoCo stores
the free joint's angular velocity in the body frame, the simulator wrapper
rotates it). ``grip_cmd`` is the gripper's integrated command in ``[-1, 1]``
(+1 closed), controller memory that is not part of the MuJoCo state.
"""

from __future__ import annotations

NUM_ARM = 7
NUM_FINGER = 2
NUM_JOINTS = NUM_ARM + NUM_FINGER

Q = slice(0, 9)
DQ = slice(9, 18)
OBJ_POS = slice(18, 21)
OBJ_QUAT = slice(21, 25)
OBJ_LIN_VEL = slice(25, 28)
OBJ_ANG_VEL = slice(28, 31)
GRIP_CMD = slice(31, 32)

STATE_DIM = 32
ACTION_DIM = 7
CTX_DIM = 3
"""Per-episode context: the cube half-extents, normalized to [-1, 1]."""
