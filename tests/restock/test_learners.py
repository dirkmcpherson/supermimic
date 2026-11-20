"""The repo's learners run unchanged on the restock representation."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.models import build_mlp
from supermimic.restock import layout as L
from supermimic.restock.calibration import rule_agreement
from supermimic.restock.data import MixtureSampler, buffer_from_episodes
from supermimic.restock.models import Actor, ManipDynamics, RestockWorldModelLearnerCfg
from supermimic.restock.representation import REP_DIM
from supermimic.restock.wm_env import RestockWorldModelEnv

from .conftest import random_episodes, random_states


def _setup(representation):
  dynamics = ManipDynamics(
    representation, L.ACTION_DIM, L.CTX_DIM, hidden_dims=(32, 32)
  )
  buffer = buffer_from_episodes(random_episodes(400, 20), 4, "cpu")
  return dynamics, MixtureSampler([(buffer, 1.0)])


def test_world_model_learner_reduces_its_loss(representation):
  dynamics, sampler = _setup(representation)
  cfg = RestockWorldModelLearnerCfg(batch_size=64, num_batches=2, horizon=4)
  learner = WorldModelLearner(dynamics, representation, cfg)
  batch = sampler.sample(1, 256).squeeze(1)
  learner.update_normalization(batch["state"], batch["action"])
  torch.manual_seed(0)
  fixed = [sampler.sample(cfg.horizon, cfg.batch_size) for _ in range(cfg.num_batches)]
  first = learner.update(iter(fixed))
  assert set(first) == {
    "Loss/state",
    *(f"Loss/{k}" for k in representation.LOSS_WEIGHT_CONFIG),
  }
  for _ in range(20):
    last = learner.update(iter(fixed))
  assert last["Loss/state"] < first["Loss/state"]


def test_actor_critic_learner_updates_through_the_world_model(representation):
  dynamics, sampler = _setup(representation)
  wm_env = RestockWorldModelEnv(dynamics, representation, sampler, 32, gamma=0.99)
  actor = Actor(REP_DIM, L.ACTION_DIM, (32, 32))
  critic = build_mlp(REP_DIM + L.CTX_DIM, (32,), output_dim=1)
  cfg = ActorCriticPolicyLearnerCfg(batch_size=32, horizon=4)
  learner = ActorCriticPolicyLearner(actor, critic, wm_env, cfg)

  obs = wm_env.reset()
  assert obs["actor"].shape == (32, REP_DIM)
  _, reward, term_prob, _ = wm_env.step(actor(obs))
  assert reward.shape == (32,) and (reward >= 0).all() and (reward <= 10).all()
  assert (term_prob >= 0).all() and (term_prob <= 1).all()
  # The value ceiling is the one terminal reward.
  assert abs(wm_env.max_step_reward / (1 - cfg.gamma) - 10.0) < 1e-9

  before = [p.detach().clone() for p in actor.parameters()]
  logs = learner.update()
  assert all(torch.isfinite(v).all() for v in logs.values())
  assert any(
    not torch.equal(a, b) for a, b in zip(before, actor.parameters(), strict=True)
  )
  assert not any(p.requires_grad for p in dynamics.parameters())


def test_critic_fits_real_outcomes(representation):
  from supermimic.restock.outcomes import OutcomeBuffer, demo_outcomes
  from supermimic.restock.policy import RestockPolicyLearner, RestockPolicyLearnerCfg

  dynamics, sampler = _setup(representation)
  wm_env = RestockWorldModelEnv(dynamics, representation, sampler, 32, gamma=0.99)
  outcomes = OutcomeBuffer(64, L.STATE_DIM, "cpu")
  states = random_states(64, 3)
  outcomes.add(states, torch.linspace(0.0, 10.0, 64))

  def learner(targets: str):
    torch.manual_seed(0)
    actor = Actor(REP_DIM, L.ACTION_DIM, (32, 32))
    critic = build_mlp(REP_DIM + L.CTX_DIM, (64,), output_dim=1)
    cfg = RestockPolicyLearnerCfg(batch_size=32, horizon=4, critic_targets=targets)
    return RestockPolicyLearner(actor, critic, wm_env, cfg, outcomes)

  real = learner("real")
  first = real.update()
  assert "Policy/value_function" not in first
  for _ in range(60):
    last = real.update()
  assert last["Policy/value_function_real"] < 0.5 * first["Policy/value_function_real"]
  # The lambda-return at lam -> 1 is the stock return; below, it still trains.
  torch.manual_seed(1)
  start = wm_env.sample_initial_states()
  real_state, real_return = outcomes.sample(32 * 4)
  start["real_state"] = real_state.view(32, 4, -1)
  start["real_return"] = real_return.view(32, 4)
  returns = {}
  for lam in (1.0, 1.0 - 1e-6, 0.5):
    probe = learner("real")
    probe.cfg.actor_lam = lam
    returns[lam] = probe._train_batch(start.clone())["return"]
  assert torch.allclose(returns[1.0], returns[1.0 - 1e-6], atol=1e-3)
  assert torch.isfinite(returns[0.5])
  mixed = learner("mixed").update()
  assert {"Policy/value_function", "Policy/value_function_real"} <= set(mixed)
  assert "Policy/value_function_real" not in learner("imagined").update()

  # Returns of back-to-back demonstration episodes.
  data = {
    "state": states[:5].numpy(),
    "terminated": np.array([0, 0, 1, 0, 0], dtype=bool),
    "truncated": np.array([0, 0, 0, 0, 1], dtype=bool),
    "home": np.array([0, 0, 1, 0, 0], dtype=bool),
  }
  kept, returns = demo_outcomes(data, 0.5, 10.0)
  assert len(kept) == 5 and returns.tolist() == [2.5, 5.0, 10.0, 0.0, 0.0]

  # The buffer keeps the most recent pairs.
  small = OutcomeBuffer(4, 1, "cpu")
  small.add(np.arange(3, dtype=np.float32)[:, None], np.arange(3, dtype=np.float32))
  small.add(
    np.arange(3, 6, dtype=np.float32)[:, None], np.arange(3, 6, dtype=np.float32)
  )
  assert small.size == 4 and sorted(small.returns.tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_rule_agreement_reports_every_rule(representation):
  data = {
    "state": random_states(200, 0).numpy(),
    "next_state": random_states(200, 1).numpy(),
    "home": torch.zeros(200, dtype=torch.bool).numpy(),
    "tipped": torch.zeros(200, dtype=torch.bool).numpy(),
  }
  data["home"][:3] = True
  report = rule_agreement(representation, data)
  assert set(report) == {*L.FLAG_NAMES, "home", "tipped", "transitions"}
  assert report["home"]["env"] == 3


def test_final_states_get_their_own_rows():
  from supermimic.restock.data import with_final_states

  count = 7
  state = np.zeros((count, L.STATE_DIM), np.float32)
  state[:, 0] = np.arange(count)
  data = {
    "state": state,
    "next_state": state + 0.5,
    "action": np.full((count, L.ACTION_DIM), 1.0, np.float32),
    "prev_action": np.zeros((count, L.ACTION_DIM), np.float32),
    "sim_ctx": np.zeros((count, L.CTX_DIM), np.float32),
    "terminated": np.array([0, 0, 1, 0, 0, 0, 0], dtype=bool),
    "truncated": np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool),
  }
  out = with_final_states(data)
  assert out["state"][:, 0].tolist() == [0, 1, 2, 2.5, 3, 4, 5, 6, 6.5]
  assert out["terminated"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0]
  assert out["truncated"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1]
  assert out["action"][:, 0].tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 0]
  buffer = buffer_from_episodes(out, 1, "cpu")
  window = buffer.sample(3, 64)["state"][:, :, 0]
  # No window runs across an episode end, and some end on a final state.
  assert bool(((window[:, 1:] - window[:, :-1]) > 0).all())
  assert bool((window[:, -1] == 2.5).any())
