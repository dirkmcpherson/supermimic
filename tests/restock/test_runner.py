"""The collection loop, against a stand-in for the simulator pool."""

from __future__ import annotations

import json

import numpy as np
import torch

from supermimic.restock import layout as L
from supermimic.restock.data import buffer_from_episodes
from supermimic.restock.models import ManipDynamics, RestockWorldModelLearnerCfg
from supermimic.restock.policy import RestockPolicyLearnerCfg
from supermimic.restock.runner import Runner, RunnerCfg

from .conftest import random_episodes, random_states


class FakePool:
  """Env ``i`` runs episodes of ``3 + i`` decisions; odd-numbered episodes are
  terminated in ``home``, even ones truncated. ``state[0]`` counts decisions,
  ``state[1]`` episodes, ``state[2]`` is the env."""

  def __init__(self, num: int, lengths: np.ndarray | None = None):
    self.num = num
    self.lengths = 3 + np.arange(num) if lengths is None else lengths
    self.t = np.zeros(num, dtype=int)
    self.episode = np.zeros(num, dtype=int)
    self.base = random_states(num).numpy()
    self.stepped = 0

  def _state(self, i: int) -> np.ndarray:
    state = self.base[i].copy()
    state[0], state[1], state[2] = self.t[i], self.episode[i], i
    return state

  def _start(self, i: int) -> np.ndarray:
    self.episode[i] += 1
    self.t[i] = 0
    return self._state(i)

  def call(self, name, args_per_worker, workers=None):
    workers = list(range(self.num)) if workers is None else workers
    assert len(args_per_worker) == len(workers)
    if name == "training_start":
      return [(self._start(i), 0) for i in workers]
    assert name == "step_auto"
    outs = []
    for i in workers:
      self.stepped += 1
      self.t[i] += 1
      next_state = self._state(i)
      ended = self.t[i] >= self.lengths[i]
      terminated = bool(ended and self.episode[i] % 2 == 1)
      out = {
        "next_state": next_state,
        "reward": 0.0,
        "terminated": terminated,
        "truncated": bool(ended and not terminated),
        "from_demo": False,
        "prefix_decisions": 0,
      }
      if ended:
        out["stages"] = np.array([name == "home" and terminated for name in L.STAGES])
        out["state"] = self._start(i)
      else:
        out["state"] = next_state
      outs.append(out)
    return outs


def small_cfg() -> RunnerCfg:
  return RunnerCfg(
    buffer_size=64,
    dynamics_hidden_dims=(16,),
    actor_hidden_dims=(16,),
    critic_hidden_dims=(16,),
    wm=RestockWorldModelLearnerCfg(batch_size=8, num_batches=1, horizon=3),
    policy=RestockPolicyLearnerCfg(batch_size=8, horizon=2, critic_targets="real"),
    actor_ema=0.5,
    compile=False,
  )


def build_runner(representation, pool: FakePool, tmp_path) -> Runner:
  dynamics = ManipDynamics(representation, L.ACTION_DIM, L.CTX_DIM, (16,))
  demo_buffer = buffer_from_episodes(random_episodes(60, 10), 2, "cpu")
  cfg = small_cfg()
  return Runner(cfg, representation, dynamics, demo_buffer, pool, [], tmp_path, "cpu")


def test_collect_when_every_episode_ends_together(representation, tmp_path):
  torch.manual_seed(0)
  pool = FakePool(4, lengths=np.full(4, 3))
  runner = build_runner(representation, pool, tmp_path)
  runner.collect(9)
  # Two episodes and two waiting rows per env, with no env stepped on the latter.
  ended = runner.replay_buffer._storage["truncated"][:, :9]
  ended = ended | runner.replay_buffer._storage["terminated"][:, :9]
  assert ended.sum(1).tolist() == [2] * 4
  assert runner.policy_decisions == pool.stepped == 4 * 7


def test_collect_stores_every_final_state_once(representation, tmp_path):
  torch.manual_seed(0)
  pool = FakePool(4)
  runner = build_runner(representation, pool, tmp_path)
  runner.collect(12)
  runner.collect(9)

  steps = 21
  store = runner.replay_buffer._storage
  waits = 0
  for i in range(pool.num):
    t = store["state"][i, :steps, 0].numpy()
    episode = store["state"][i, :steps, 1].numpy()
    ended = (store["terminated"][i, :steps] | store["truncated"][i, :steps]).numpy()
    action = store["action"][i, :steps].numpy()
    prev_action = store["prev_action"][i, :steps].numpy()
    length = 3 + i
    assert t[0] == 0 and episode[0] == 1
    for k in range(steps):
      # A row ends an episode exactly when it holds the episode's final state.
      assert ended[k] == (t[k] == length)
      if ended[k]:
        waits += 1
        assert not action[k].any()
        assert bool(store["terminated"][i, k]) == (episode[k] % 2 == 1)
      if k + 1 < steps:
        if ended[k]:
          assert t[k + 1] == 0 and episode[k + 1] == episode[k] + 1
          assert not prev_action[k + 1].any()
        else:
          assert t[k + 1] == t[k] + 1 and episode[k + 1] == episode[k]
          assert np.array_equal(prev_action[k + 1], action[k])
  # Waiting rows are not decisions.
  assert runner.policy_decisions == pool.stepped == steps * pool.num - waits
  assert waits > 0

  # A cut marks every env's last row, so no window runs into the next round.
  runner.collect(3, cut=True)
  last = store["terminated"][:, steps + 2] | store["truncated"][:, steps + 2]
  assert bool(last.all()) and runner._needs_reset
  check = runner._model_check()
  assert 0 < check["Model/end_share"] < 1 and 0 <= check["Model/end_prob_at_ends"] <= 1

  # Every state of a finished episode is stored with the return that followed:
  # the discounted terminal reward of an episode that ended in home, else 0.
  outcomes = runner.outcomes
  assert outcomes.size > 0 and "Value/abs_error" in check
  stored = outcomes.state[: outcomes.size].numpy()
  t, episode, env = stored[:, 0], stored[:, 1], stored[:, 2]
  expected = np.where(episode % 2 == 1, 10.0 * 0.99 ** (3 + env - 1 - t), 0.0)
  assert np.allclose(outcomes.returns[: outcomes.size].numpy(), expected, atol=1e-5)
  assert (t < 3 + env).all()

  # The averaged actor: one noise-free env acts with it, and it follows the
  # trained actor at the configured rate.
  assert runner._ema_env.sum() == 1 and not (runner._ema_env & ~runner._clean).any()
  assert len(runner._episodes["ema"]) > 0
  runner._update_ema(0.0)
  trained = next(runner.actor.parameters())
  averaged = next(runner.ema_actor.parameters())
  assert torch.equal(trained, averaged)
  with torch.no_grad():
    trained.add_(1.0)
  runner._update_ema(0.5)
  assert torch.allclose(averaged, trained - 0.5)
  assert "actor_ema" in runner.state_dict()
  logs = runner.policy_learner.update()
  assert torch.isfinite(logs["Policy/value_function_real"])


def test_guard_holds_the_policy_back_on_a_sudden_model_loss(representation, tmp_path):
  runner = build_runner(representation, FakePool(4), tmp_path)
  # Without a guard every round is trusted.
  assert all(runner._model_trusted(loss) for loss in (0.1, 5.0, 0.1))
  runner.cfg.wm_guard = 3.0
  runner._wm_level = None
  assert all(runner._model_trusted(0.1) for _ in range(20))
  assert not runner._model_trusted(0.5)
  assert runner._model_trusted(0.25)
  # A lasting rise becomes the level: after enough rounds it is trusted again.
  held = 0
  while not runner._model_trusted(0.5):
    held += 1
  assert 10 < held < 200

  runner.collect(9)
  assert set(runner._round_extremes) == set(runner._extremes)
  assert all(v >= 0 for v in runner._round_extremes.values())


def test_run_writes_metrics_and_a_round_trace(representation, tmp_path):
  torch.manual_seed(0)
  runner = build_runner(representation, FakePool(4), tmp_path)
  cfg = runner.cfg
  cfg.max_decisions, cfg.eval_interval, cfg.log_interval = 300, 10**9, 2
  cfg.trace, cfg.wm_guard = True, 1e-6
  summary = runner.run()
  trace = [json.loads(line) for line in (tmp_path / "round_trace.jsonl").open()]
  assert len(trace) == summary["rounds"]
  # With a guard this tight only the first round updates the policy.
  assert [row["policy_updated"] for row in trace] == [True] + [False] * (len(trace) - 1)
  metrics = [json.loads(line) for line in (tmp_path / "policy_metrics.jsonl").open()]
  assert metrics[-1]["Policy/skipped_rounds"] == 2
  assert metrics[-1]["Collect/max_joint_speed"] >= 0
