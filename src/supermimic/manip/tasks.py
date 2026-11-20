"""Per-task defaults that are not properties of the simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
  name: str
  episode_steps: int
  """Control steps of a training episode (20 Hz)."""
  eval_episode_steps: int
  """Control steps of an evaluation episode."""
  demo_files: tuple[str, ...]
  """robomimic file names of the human demonstrations (proficient, mixed)."""


TASKS: dict[str, TaskSpec] = {
  "lift": TaskSpec(
    name="lift",
    episode_steps=100,
    eval_episode_steps=150,
    demo_files=("low_dim_v141_lift_ph.hdf5", "low_dim_v141_lift_mh.hdf5"),
  ),
  # robomimic evaluates Square with a 400-step horizon.
  "square": TaskSpec(
    name="square",
    episode_steps=250,
    eval_episode_steps=400,
    demo_files=("low_dim_v141_square_ph.hdf5", "low_dim_v141_square_mh.hdf5"),
  ),
}
