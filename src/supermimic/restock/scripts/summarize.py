"""Write ``REPORT.md`` for the restock runs under one directory.

python -m supermimic.restock.scripts.summarize --data <dir> --runs <dir> [--out REPORT.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supermimic.restock import layout as L


def _rows(path: Path) -> list[dict]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(value: float | None) -> str:
  return "n/a" if value is None or value != value else f"{100 * value:.0f}%"


def data_section(data: Path) -> list[str]:
  meta = json.loads((data / "meta.json").read_text())
  tapes = meta["tapes_detail"]
  recorded = [t for t in tapes if t["tape_end_decision"] is not None]
  same_end = sum(abs(t["tape_end_decision"] - t["decisions"]) <= 1 for t in recorded)
  lines = [
    "## Demonstrations",
    "",
    f"- {meta['tapes']} tapes re-executed in the simulator: {meta['transitions']} transitions "
    f"({meta['simulator_frames']} simulator frames), of {meta['tape_decisions']} decisions on the tapes; "
    "the rest lie after the environment's terminal and are dropped.",
    f"- {same_end} of {len(recorded)} tapes end at the decision the dataset records (within one).",
    "- Tapes reaching each stage: "
    + ", ".join(f"{name} {meta['stages'][name]}" for name in L.STAGES)
    + ".",
    f"- Can position against the tape's stored state: median {1000 * meta['can_dev_m_median']:.1f} mm, "
    f"largest {1000 * meta['can_dev_m_max']:.0f} mm.",
    "",
    "Smooth rule against the environment's own decision, on the true states of these transitions "
    "(a latch is compared only while it is still unset, so a rule that fires a few decisions early "
    "counts once per early decision):",
    "",
    "| Rule | Environment sets it | Rule sets it | Environment only | Rule only |",
    "|---|---|---|---|---|",
  ]
  for name, row in meta["rule_agreement"].items():
    if name != "transitions":
      lines.append(
        f"| {name} | {row['env']} | {row['rule']} | {row['env_only']} | {row['rule_only']} |"
      )
  return [*lines, ""]


def overview_section(runs: list[Path]) -> list[str]:
  """One row per scored run, start set and checkpoint, then the mean over each
  group of runs that differ only by seed (``<group>_s<seed>``)."""
  rows: dict[tuple[str, str], list[dict]] = {}
  lines = [
    "## Overview",
    "",
    "One episode per start, deterministic actions. `final` is the checkpoint at the end of training; `latest` is the "
    "checkpoint saved at the last evaluation during training, a few thousand decisions before the end. "
    "`ended early` is the share of episodes that tipped the can before picking it up; the last column "
    "is the noise-free training environments' last 200 episodes.",
    "",
    "| Run | Starts | Checkpoint | Episodes | "
    + " | ".join(L.STAGES)
    + " | ended early | home on training starts |",
    "|---|---|---|---|" + "---|" * (len(L.STAGES) + 2),
  ]
  for run in runs:
    summary_path = run / "train_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    train_home = summary.get("final_training_stages", {}).get("Train/clean/home")
    for path in sorted(run.glob("score_*.json")):
      s = json.loads(path.read_text())
      if not s.get("settle", True):
        continue
      checkpoint = s["checkpoint"] + (
        ", averaged actor" if s.get("actor") == "ema" else ""
      )
      early = sum(e["tipped"] and not e["picked"] for e in s["per_episode"]) / max(
        s["episodes"], 1
      )
      row = {**s["stages"], "early": early, "train_home": train_home}
      group = run.name.rsplit("_s", 1)[0]
      rows.setdefault((group, s["set"], checkpoint), []).append(row)
      lines.append(
        f"| {run.name} | {s['set']} | {checkpoint} | {s['episodes']} | "
        + " | ".join(_pct(s["stages"][name]) for name in L.STAGES)
        + f" | {_pct(early)} | {_pct(train_home)} |"
      )
  lines += [
    "",
    "| Group | Starts | Checkpoint | Runs | home mean | home range | home std. dev. | picked mean | tipped mean | ended early mean |",
    "|---|---|---|---|---|---|---|---|---|---|",
  ]
  for (group, starts, checkpoint), group_rows in rows.items():
    home = [r["home"] for r in group_rows]
    n = len(home)
    mean = sum(home) / n
    std = (sum((h - mean) ** 2 for h in home) / (n - 1)) ** 0.5 if n > 1 else None
    lines.append(
      f"| {group} | {starts} | {checkpoint} | {n} | {mean:.3f} | {min(home):.3f} to {max(home):.3f} | "
      + ("n/a" if std is None else f"{std:.3f}")
      + " | "
      + " | ".join(
        f"{sum(r[k] for r in group_rows) / n:.3f}"
        for k in ("picked", "tipped", "early")
      )
      + " |"
    )
  return [*lines, ""]


def run_section(run: Path) -> list[str]:
  setup = (
    json.loads((run / "setup.json").read_text())
    if (run / "setup.json").exists()
    else {}
  )
  summary = (
    json.loads((run / "train_summary.json").read_text())
    if (run / "train_summary.json").exists()
    else None
  )
  args = setup.get("args", {})
  lines = [f"## Run `{run.name}`", ""]
  if args:
    keys = ("seed", "num_envs", "decisions", "wm_updates", "steps_per_round", "wm_batches", "policy_batches", "policy_horizon", "gamma", "explore_scale", "demo_mix", "demo_starts", "critic", "actor_lam", "actor_ema")  # fmt: skip
    lines += [
      "Settings: " + ", ".join(f"{k} {args[k]}" for k in keys if k in args) + ".",
      "",
    ]
  if summary is None:
    lines += ["Still running, or it ended without a summary.", ""]
  else:
    lines += [
      f"Ended on `{summary['stop_reason']}` after {summary['rounds']} rounds, {summary['env_steps']:,} decisions "
      f"({4 * summary['env_steps']:,} simulator frames; {summary.get('prefix_decisions', 0):,} of the decisions were tape prefixes), "
      f"{summary['hours']:.2f} h, peak GPU memory {summary.get('gpu_gb', float('nan')):.1f} GiB.",
      "",
    ]
  scores = sorted(run.glob("score_*.json"))
  if scores:
    lines += [
      "Scored with one episode per start, in order, in one simulator process, with deterministic actions:",
      "",
      "| Starts | Checkpoint | Decisions trained | Episodes | "
      + " | ".join(L.STAGES)
      + " |",
      "|---|---|---|---|" + "---|" * len(L.STAGES),
    ]
    for path in scores:
      s = json.loads(path.read_text())
      lines.append(
        f"| {s['set']}{'' if s.get('settle', True) else ' (no settle)'} | {s['checkpoint']}{', averaged actor' if s.get('actor') == 'ema' else ''} | {s['env_steps']:,} | {s['episodes']} | "
        + " | ".join(_pct(s["stages"][name]) for name in L.STAGES)
        + " |"
      )
    lines.append("")
  evals = _rows(run / "policy_eval.jsonl")
  metrics = {m["round"]: m for m in _rows(run / "policy_metrics.jsonl")}
  if evals:
    lines += [
      "During training (evaluation starts spread over the training simulators; training columns are the "
      "noise-free environments' last 200 episodes):",
      "",
      "| Decisions | Hours | Eval home | Eval picked | Eval placed | Eval tipped | Train picked | Train placed | Train home | Imagined return |",
      "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in evals:
      m = metrics.get(e["round"], {})
      imagined = m.get("Policy/reward")
      horizon = args.get("policy_horizon", 16)
      lines.append(
        f"| {e['env_steps']:,} | {e['hours']:.2f} | {_pct(e['home'])} | {_pct(e['picked'])} | {_pct(e['placed_v2'])} | "
        f"{_pct(e['tipped'])} | {_pct(m.get('Train/clean/picked'))} | {_pct(m.get('Train/clean/placed_v2'))} | "
        f"{_pct(m.get('Train/clean/home'))} | "
        + ("n/a" if imagined is None else f"{imagined * horizon * 0.12:.2f}")
        + " |"
      )
    lines.append("")
  return lines


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True)
  parser.add_argument("--runs", required=True)
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  runs = sorted(
    p for p in Path(args.runs).iterdir() if (p / "runner_cfg.json").exists()
  )
  lines = [
    "# Restock: SuperMimic runs",
    "",
    *overview_section(runs),
    *data_section(Path(args.data)),
  ]
  for run in runs:
    lines += run_section(run)
  out = Path(args.out) if args.out else Path(args.runs) / "REPORT.md"
  out.write_text("\n".join(lines) + "\n")
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
