"""The repo's learners run unchanged on the manipulation representation."""

from __future__ import annotations

import torch

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.manip import layout as L
from supermimic.manip.data import MixtureSampler, buffer_from_episodes
from supermimic.manip.models import Actor, ManipDynamics, ManipWorldModelLearnerCfg
from supermimic.manip.representation import REP_DIM
from supermimic.manip.reward import LiftRewardCfg
from supermimic.manip.wm_env import ManipWorldModelEnv
from supermimic.manip.wm_eval import open_loop_errors
from supermimic.models import build_mlp

from .conftest import random_episodes


def _setup(representation):
  dynamics = ManipDynamics(
    representation, L.ACTION_DIM, L.CTX_DIM, hidden_dims=(32, 32)
  )
  buffer = buffer_from_episodes(random_episodes(400, 20), 4, "cpu")
  return dynamics, MixtureSampler([(buffer, 1.0)])


def test_dynamics_clips_the_command_and_advances_the_state(representation):
  dynamics, sampler = _setup(representation)
  batch = sampler.sample(1, 16).squeeze(1)
  rep = representation.represent(batch["state"])
  aux = representation.extract_aux_state(batch["state"])
  big = torch.full((16, L.ACTION_DIM), 5.0)
  next_aux, next_rep = dynamics(rep, big, aux, batch["sim_ctx"])
  clipped_aux, clipped_rep = dynamics(rep, big.clamp(-1, 1), aux, batch["sim_ctx"])
  assert next_rep.shape == (16, REP_DIM) and next_aux.shape == (16, 7)
  assert torch.allclose(next_rep, clipped_rep) and torch.allclose(next_aux, clipped_aux)
  assert torch.allclose(next_aux[:, 3:].norm(dim=-1), torch.ones(16), atol=1e-5)


def test_world_model_learner_reduces_its_loss(representation):
  dynamics, sampler = _setup(representation)
  cfg = ManipWorldModelLearnerCfg(batch_size=64, num_batches=2, horizon=4)
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
  wm_env = ManipWorldModelEnv(
    dynamics, representation, sampler, 32, "lift", LiftRewardCfg()
  )
  actor = Actor(REP_DIM, L.ACTION_DIM, (32, 32))
  critic = build_mlp(REP_DIM + L.CTX_DIM, (32,), output_dim=1)
  cfg = ActorCriticPolicyLearnerCfg(batch_size=32, horizon=4)
  learner = ActorCriticPolicyLearner(actor, critic, wm_env, cfg)

  obs = wm_env.reset()
  assert obs["actor"].shape == (32, REP_DIM)
  assert obs["critic"].shape == (32, REP_DIM + L.CTX_DIM)
  action = actor(obs)
  assert action.abs().max() <= 1.0
  _, reward, term_prob, _ = wm_env.step(action)
  assert reward.shape == (32,) and (term_prob == 0).all()
  assert reward.max() <= wm_env.max_step_reward

  before = [p.detach().clone() for p in actor.parameters()]
  logs = learner.update()
  assert {"Policy/reward", "Policy/grad_norm", "Policy/value_function"} <= set(logs)
  assert all(torch.isfinite(v).all() for v in logs.values())
  assert any(
    not torch.equal(a, b) for a, b in zip(before, actor.parameters(), strict=True)
  )
  # The policy update must not touch the world model.
  assert not any(p.requires_grad for p in dynamics.parameters())


def test_open_loop_errors_reports_each_horizon(representation):
  dynamics, sampler = _setup(representation)
  windows = sampler.sample(9, 64)
  report = open_loop_errors(dynamics, representation, windows, (1, 8))
  assert report["windows"] == 64 and set(report) == {"windows", "k1", "k8"}
  assert report["k8"]["obj_pos_m"]["mean"] >= 0
  assert 0.0 <= report["k8"]["contact_fraction"] <= 1.0
