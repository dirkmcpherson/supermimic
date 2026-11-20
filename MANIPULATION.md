# SuperMimic on manipulation tasks: changes and results

This branch tests SuperMimic's learning machinery (the differentiable world
model and the policy trained through it) on fixed-base manipulation. Nothing in
the existing packages is modified; everything is added next to them.

## What was added

| Path | What it is |
|---|---|
| `src/supermimic/manip/` | Adapter for robosuite tasks (`lift`, `square`): simulator wrapper, state representation, soft task reward, data collection, runner. See its README. |
| `src/supermimic/restock/` | Adapter for a task in the Genesis simulator: simulator workers in a separate Python environment, state representation with the task's stage rules, runner, scoring, diagnosis. See its README. |
| `src/supermimic/restock/policy.py`, `outcomes.py` | The two changes to policy learning described below. |
| `tests/manip/`, `tests/restock/` | Tests that need no simulator (66 in total). |
| `scripts/run_manip_overnight.sh`, `scripts/restock_cluster.sh` | Launchers (desktop and Slurm). |

Both adapters reuse the repo's `WorldModelLearner`, `ActorCriticPolicyLearner`,
replay buffer, optimizers and captured updates unchanged. Both learn from
simulator state, not images.

## Robosuite Lift

Three policy runs (two settings, one extra seed) reached 100% success over 200
evaluation episodes in 0.22 to 0.26 hours each on one desktop GPU, after
640k to 770k real transitions. Lift is also solved by cloning the
demonstrations, so this only shows that the pipeline works. Square
(`NutAssemblySquare`) is implemented and tested but has not been trained.

## A sparse-reward task with contact

The second adapter's task is a multi-stage pick, place and push with one
reward, +10 when the episode ends in success. An episode also ends, without
reward, when the object is knocked over. Training is 250k decisions (1M
simulator frames) with 74 demonstrations, 11 of them successful. A run is
scored by its final checkpoint: one episode with deterministic actions from
each of 30 fixed evaluation starts.

### Stock learner: 0.27

With the stock actor-critic learner (critic fitted to imagined returns, actor
climbing imagined rewards plus the critic's value of the last imagined state),
five seeds succeeded on 0.37, 0.27, 0.13, 0.47 and 0.13 of the evaluation
starts: mean 0.273. About half of the evaluation episodes knocked the object
over before picking it up.

`scripts/diagnose.py` measures why, on real episodes of the trained policies:

- Fed the real actions, the world model predicts the fall one decision ahead
  (0.92) but not two or more ahead (0.03 to 0.18, then about 0). It does not
  foresee the contact that starts the fall.
- Imagined rollouts from real starts therefore succeed in 0.94 to 0.97 of
  cases, including for a policy that knocks the object over in 90% of real
  episodes.
- The critic says 7 to 8 of 10 at the start of episodes that go on to fail.

So the policy is given no reason to approach the object carefully.

### Two changes to policy learning: 0.62

- `--critic real`: the critic is fitted to the discounted return that real
  episodes went on to collect (10 discounted back from a successful ending, 0
  for every state of an episode that ended otherwise), from the most recent
  60k real states and the demonstrations. No imagined return goes into it.
- `--actor-lam 0.8`: the actor climbs the lambda-return, which weighs the
  critic's value of every imagined state, not only the last one.

There is still one critic network; it is trained on real states and evaluated
on imagined ones. Neither change works alone in the one test made: the real
critic with the stock return (`--actor-lam 1`) scored 0.30.

Share of successful episodes, final checkpoint:

| Run | 30 evaluation starts | 61 training starts |
|---|---|---|
| `rc_s0` | 0.70 | 0.95 |
| `rc_s1` | 0.63 | 0.93 |
| `rc_s2` | 0.63 | 0.97 |
| `rc_s3` | 0.73 | 0.97 |
| `rc_s4` | 0.13 | 0.38 |
| `diag_s4` | 0.70 | 0.95 |
| `guard_s0` | 0.60 | 0.97 |
| `guard_s1` | 0.63 | 0.93 |
| `guard_s2` | 0.73 | 0.95 |
| `guard_s3` | 0.67 | 0.80 |
| `guard_s4` | 0.63 | 0.95 |

| Set | Runs | Mean | Range |
|---|---|---|---|
| stock learner | 5 | 0.273 | 0.13 to 0.47 |
| `rc` (the five seeds fixed in advance) | 5 | 0.567 | 0.13 to 0.73 |
| `guard` | 5 | 0.653 | 0.60 to 0.73 |
| all eleven with the two changes | 11 | 0.618 (median 0.633) | 0.13 to 0.73 |

The `guard` and `diag` runs are the same algorithm as `rc`: they add a guard
that never triggered and per-round logging (see below). Training is not
repeatable from a seed, so every run is a fresh sample.

Several evaluation starts fail before the pick in every run, which puts the
practical limit of this method on these starts near 0.73.

Other settings tried, one run each: `--actor-lam 0.5` scored 0.40 (stable but
too slow in the budget). An averaged copy of the actor (`--actor-ema 0.98`)
scored the same as the actor itself within one episode.

### The collapse

`rc_s4` was at 0.70 in the in-training evaluations up to 114k decisions. Then,
over about 40 rounds, the world model's loss rose 50 times (all loss groups,
mostly the objects' angular velocities), the actor's gradient norms rose 100
to 1000 times, and the policy never recovered. The model's input normalization
did not shift. It happened once in eleven runs and could not be replayed.

`--wm-guard 3` skips a round's policy update when the model's loss exceeds
three times its recent level. It would have triggered in that run. In the five
runs made with it the loss never rose that far, so whether it saves a run is
untested. `--trace` logs every round for the next occurrence.

### What this result is, and is not

The two changes are standard techniques: a lambda-return backpropagated
through a learned model, and Monte Carlo value regression on a sparse terminal
reward. What reached 0.62 is SuperMimic's world model together with a
model-free critic on real outcomes. The stock method, in which everything the
policy learns from comes out of the model, reached 0.27 here, because the
model misses the one contact event that decides most episodes.

Not tested: the stock critic with the lambda-return, the `mixed` critic
option, the guard under an actual collapse, and learning from images.

## Reproducing

```bash
# Tests (no simulator needed)
pytest tests/manip --confcutdir=tests/manip
pytest tests/restock --confcutdir=tests/restock

# Robosuite
DEMO_DIR=/path/to/robomimic/low_dim/datasets scripts/run_manip_overnight.sh lift
```

The second adapter needs its task repository and simulator environment; its
README has the commands.
