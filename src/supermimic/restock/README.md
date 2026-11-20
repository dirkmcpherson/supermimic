# SuperMimic on the restock task

A Kinova Gen3 Lite picks up a can, sets it on a shelf, moves behind it and
slides it against a goal can. The task, its simulator (Genesis) and its
demonstrations live in another repository; this package adapts SuperMimic to
it without changing anything there.

## Setting

The environment is built by the task repository's own constructor: world
`gc_kp4_riser3_shelf6`, delta joint actions (4 frames per decision), ladder
`nested_sparse10`, tip guard `not_in_hand`, 1200 frames (300 decisions) per
episode, training starts drawn from the environment's own start table, and 30
`rnd` evaluation starts scored by the `home` stage.

| | |
|---|---|
| Policy input | the simulator state (85 numbers, see `representation.py`), including velocities, both cans' poses and the environment's stage latches |
| Reward in the world model | the `home` event with soft edges (`predicates.py`); nothing is paid for earlier stages |
| Demonstrations | each tape re-executed in the simulator to get the full state, cut at the environment's terminal (rows after a tip are dropped) |
| Evaluation actions | deterministic |

With the stock learner (`--critic imagined`) the real environment's reward is
never used for learning; its `home` flag is only counted.

## The critic and real outcomes

The stock learner fits the critic to imagined returns and the actor climbs the
imagined rewards plus the critic's value of the rollout's last state. On this
task that fails: the world model predicts a tipped can one decision ahead but
not the contact that starts the fall, so imagined rollouts almost never tip,
the critic says 7 to 8 of 10 at the start of episodes that go on to tip, and
the policy gets no reason to approach the can carefully
(`scripts/diagnose.py` measures all three on real episodes).

Two options in `policy.py` change that, and together they are what works:

- `--critic real`: the critic is fitted to the discounted return real episodes
  went on to collect (`outcomes.py`; the task's one terminal reward, so this
  does use the environment's `home` event).
- `--actor-lam 0.8`: the actor climbs the lambda-return, which weighs the
  critic's value of every imagined state, not only the last. Real experience
  can then veto a state the model wrongly expects to pass through safely.
  With `--critic real` and the stock return (`--actor-lam 1`) nothing improves.

`--actor-ema 0.98` keeps an averaged copy of the actor, evaluated alongside.

Training is not repeatable from a seed (two runs with the same seed differ
within ten rounds), so every run is a fresh sample. One of eleven runs of this
configuration collapsed: over about 40 rounds the world model's loss rose 50
times, the actor's gradient norms 100 to 1000 times, and the policy never
recovered. Two options exist for that case:

- `--trace` writes every round's losses, gradient norm and the largest
  collected velocities to `round_trace.jsonl`.
- `--wm-guard 3` skips a round's policy update when the world model's loss
  exceeds three times its recent level. It would have triggered in the
  collapsed run; in the five runs made with it the loss never rose that far,
  so whether it saves a run is untested.

## Layout

- `sim_worker.py`: one simulator per process, NumPy only, run with the
  simulator environment's Python. `vec_env.py` starts and steps a pool of them.
- `layout.py`: the 60-number full state read once per decision.
- `representation.py`, `predicates.py`: the state representation, the
  integration step, and the smooth versions of the environment's stage rules.
- `wm_env.py`, `runner.py`: the differentiable environment and the online loop.
- `policy.py`, `outcomes.py`: the learner options above and the store of real
  episode outcomes.
- `calibration.py`: how often the smooth rules agree with the environment on
  real transitions.
- `scripts/prepare_data.py`, `scripts/train.py`, `scripts/score.py`,
  `scripts/diagnose.py`, `scripts/summarize.py`, and the
  launcher `scripts/restock_cluster.sh` at the repository root.

## Running (Slurm cluster)

Set `LAB` to the directory that holds the two Python environments
(`condaenv/rl` for the learner, `condaenv/genesis` for the simulator), the
task repository and this checkout; the launcher's header lists the other
paths it derives from it.

```
export LAB=/path/to/lab/dir
scripts/restock_cluster.sh data
scripts/restock_cluster.sh train s0 --seed 0 --critic real --actor-lam 0.8 --actor-ema 0.98
scripts/restock_cluster.sh score s0
scripts/restock_cluster.sh diagnose s0
```

A training job runs from a copy of the code taken when it is submitted
(`runs/<name>/src`).

Tests that need no simulator: `pytest tests/restock --confcutdir=tests/restock`.
