"""What real episodes went on to return, state by state.

The task pays once, ``home_reward`` when an episode ends in ``home``. A state
``k`` decisions before that ending was therefore followed by a discounted
return of ``home_reward * gamma**(k - 1)``; a state of an episode that ended any
other way by 0. These pairs are a model-free estimate of how good a state is
under the recent policies, and the critic can be fitted to them.
"""

from __future__ import annotations

import numpy as np
import torch


def discounted_outcomes(length: int, paid: float, gamma: float) -> np.ndarray:
  """Return that followed each of an episode's ``length`` states (the final
  state excluded), when its last transition paid ``paid``."""
  return (paid * gamma ** np.arange(length - 1, -1, -1)).astype(np.float32)


def demo_outcomes(
  data: dict[str, np.ndarray], gamma: float, home_reward: float
) -> tuple[np.ndarray, np.ndarray]:
  """States and following returns of back-to-back demonstration episodes
  (one row per transition, end flags and ``home`` on each last one)."""
  ended = data["terminated"] | data["truncated"]
  returns = np.zeros(len(ended), dtype=np.float32)
  start = 0
  for end in np.flatnonzero(ended):
    paid = home_reward if data["home"][end] else 0.0
    returns[start : end + 1] = discounted_outcomes(end + 1 - start, paid, gamma)
    start = end + 1
  return data["state"][:start].astype(np.float32), returns[:start]


class OutcomeBuffer:
  """The most recent ``capacity`` (state, return) pairs, on the device."""

  def __init__(self, capacity: int, state_dim: int, device: str):
    self.capacity = capacity
    self.state = torch.zeros(capacity, state_dim, device=device)
    self.returns = torch.zeros(capacity, device=device)
    self.size = 0
    self._next = 0

  def add(self, states: np.ndarray | torch.Tensor, returns: np.ndarray | torch.Tensor):
    device = self.state.device
    states = torch.as_tensor(states, dtype=torch.float32, device=device)[
      -self.capacity :
    ]
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)[
      -self.capacity :
    ]
    index = (self._next + torch.arange(len(states), device=device)) % self.capacity
    self.state[index] = states
    self.returns[index] = returns
    self._next = (self._next + len(states)) % self.capacity
    self.size = min(self.size + len(states), self.capacity)

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if self.size == 0:
      raise ValueError("no outcomes stored yet")
    index = torch.randint(self.size, (count,), device=self.state.device)
    return self.state[index], self.returns[index]
