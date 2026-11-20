"""How well the smooth stage rules reproduce the environment's own decisions
on real transitions."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.restock import layout as L
from supermimic.restock.representation import REP_FLAGS, RestockRepresentation


@torch.no_grad()
def rule_agreement(
  representation: RestockRepresentation, data: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
  """Compare, transition by transition, the model's rule for each latch and
  each episode end with what the environment did.

  ``data`` holds ``state``, ``next_state`` and the environment's ``home`` and
  ``tipped`` flags of each transition. Every rule is applied to the true state
  and the true previous latches, so the numbers isolate the rules from the
  learned dynamics. For each quantity: how often the environment set it
  (``env``), how often the rule does (``rule``, at 0.5), and the transitions on
  which only one of them does.
  """
  state = torch.as_tensor(data["state"], dtype=torch.float32)
  next_state = torch.as_tensor(data["next_state"], dtype=torch.float32)
  rules = representation.rules
  scene = representation.scene(representation.represent(next_state))
  flags, _, _ = rules.advance(
    state[:, L.FLAGS], state[:, L.SLIDE_GAIN.start], state[:, L.SLIDE_MIN.start], scene
  )
  true_flags = next_state[:, L.FLAGS] > 0.5
  was_set = state[:, L.FLAGS] > 0.5

  def compare(env: torch.Tensor, rule: torch.Tensor) -> dict[str, float]:
    return {
      "env": int(env.sum()),
      "rule": int(rule.sum()),
      "env_only": int((env & ~rule).sum()),
      "rule_only": int((rule & ~env).sum()),
    }

  report = {}
  for i, name in enumerate(L.FLAG_NAMES):
    # Score the transitions on which the latch could still be set.
    open_ = ~was_set[:, i]
    report[name] = compare(true_flags[open_, i], flags[open_, i] > 0.5)
  rep_next = representation.represent(next_state)
  home = rules.home(scene, rep_next[:, REP_FLAGS])
  tipped = rules.tipped(scene)
  env_home = torch.as_tensor(data["home"], dtype=torch.bool)
  env_tipped = torch.as_tensor(data["tipped"], dtype=torch.bool)
  report["home"] = compare(env_home, home > 0.5)
  report["home"]["rule_mean_on_env"] = (
    float(home[env_home].mean()) if env_home.any() else float("nan")
  )
  report["home"]["rule_max_elsewhere"] = float(home[~env_home].max())
  report["tipped"] = compare(env_tipped, tipped > 0.5)
  report["transitions"] = {"count": len(state)}
  return report
