"""Roll a deterministic actor out on standard Lift episodes."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.manip.representation import ArmObjectRepresentation
from supermimic.manip.vec_env import VecEnv


@torch.no_grad()
def act(
  actor: torch.nn.Module,
  representation: ArmObjectRepresentation,
  states: np.ndarray,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Noise-free actions for simulator states. Returns ``(actions, rep_state)``."""
  state = torch.as_tensor(states, dtype=torch.float32, device=device)
  rep = representation.represent(state)
  obs = TensorDict({"actor": rep}, batch_size=[rep.shape[0]], device=device)
  return actor(obs), rep


def evaluate_policy(
  actor: torch.nn.Module,
  representation: ArmObjectRepresentation,
  env: VecEnv,
  episodes: int,
  horizon: int,
  device: str,
) -> dict[str, float]:
  """Success over at least ``episodes`` task-reset episodes of ``horizon`` steps
  (the env's own horizon must equal ``horizon`` so all episodes end together).

  ``success_rate`` counts an episode in which robosuite's success test passed
  on any step (the robomimic convention); ``held_rate`` those where it still
  passed on the last step.
  """
  was_training = actor.training
  actor.eval()
  success, held, first = [], [], []
  while len(success) < episodes:
    out = env.reset()
    any_success = np.zeros(env.num_envs, dtype=bool)
    first_step = np.full(env.num_envs, np.nan)
    for t in range(horizon):
      actions, _ = act(actor, representation, out["state"], device)
      out = env.step(actions.cpu().numpy())
      new = out["success"] & ~any_success
      first_step[new] = t + 1
      any_success |= out["success"]
    assert out["truncated"].all(), "env horizon differs from the evaluation horizon"
    success += any_success.tolist()
    held += out["success"].tolist()
    first += first_step[any_success].tolist()
  actor.train(was_training)
  return {
    "episodes": len(success),
    "success_rate": float(np.mean(success)),
    "held_rate": float(np.mean(held)),
    "steps_to_success": float(np.mean(first)) if first else float("nan"),
  }
