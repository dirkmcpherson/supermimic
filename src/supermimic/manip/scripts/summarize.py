"""Write ``REPORT.md`` for a run directory from the stages' JSON outputs.

python -m supermimic.manip.scripts.summarize --data data/square --run logs/square/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
  return json.loads(path.read_text()) if path.exists() else None


def _rows(path: Path) -> list[dict]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cm(value: float) -> str:
  return f"{100 * value:.2f}"


def _pct(value: float) -> str:
  return f"{100 * value:.1f}%"


def _data_section(meta: dict) -> list[str]:
  replay, branches = meta["demo_replay"], meta["branches"]
  kinds = ", ".join(
    f"{name}: {count} episodes "
    f"({100 * branches['success_rate_by_kind'][name]:.0f}% succeed)"
    for name, count in branches["episodes_by_kind"].items()
  )
  return [
    "## Data",
    "",
    f"- Demos replayed in this simulator: {replay['demos']} "
    f"({replay['transitions']} transitions), {_pct(replay['success_rate'])} "
    "succeed; the object deviates from the recording by "
    f"{1000 * replay['obj_error_m_median']:.1f} mm (median) / "
    f"{1000 * replay['obj_error_m_max']:.1f} mm (max).",
    f"- Exploration branches: {branches['transitions']} transitions in "
    f"{branches['seconds'] / 60:.1f} min ({branches['steps_per_second']:.0f} steps/s). "
    f"{kinds}.",
    "",
  ]


def _wm_section(report: dict) -> list[str]:
  lines = [
    "## Stage 1: world model alone",
    "",
    f"Best validation loss {report['best_val_loss']:.3f} at update "
    f"{report['best_update']}; trained for {report['train_seconds'] / 60:.1f} min, "
    f"peak GPU memory {report.get('gpu_gb', 0):.1f} GiB.",
    "",
    "Open-loop object-position error on held-out windows, in cm. *hold* is the "
    "error of predicting that the object stays where the window started. "
    "*contact* rows keep only windows in which the object really moved by at "
    "least 0.5 cm.",
    "",
    "| Held-out set | Steps | Windows | Model median | Model p90 | Hold median | "
    "Contact share | Contact model median | Contact model p90 | Contact hold median |",
    "|---|---|---|---|---|---|---|---|---|---|",
  ]
  sets = {**report["open_loop"]}
  for name, result in report.get("open_loop_long", {}).items():
    sets[f"{name} (long)"] = result
  for name, result in sets.items():
    for key in sorted(
      (k for k in result if k.startswith("k")), key=lambda k: int(k[1:])
    ):
      r = result[key]
      lines.append(
        f"| {name} | {key[1:]} | {result['windows']} | "
        f"{_cm(r['obj_pos_m']['median'])} | {_cm(r['obj_pos_m']['p90'])} | "
        f"{_cm(r['obj_pos_hold_m']['median'])} | {100 * r['contact_fraction']:.0f}% | "
        f"{_cm(r['contact_obj_pos_m']['median'])} | "
        f"{_cm(r['contact_obj_pos_m']['p90'])} | "
        f"{_cm(r['contact_obj_pos_hold_m']['median'])} |"
      )
  lines.append("")
  return lines


_SCORE_HEADER = [
  "| Policy | Environment | Episodes | Success | Still successful at the end |",
  "|---|---|---|---|---|",
]


def _score_rows(report: dict) -> list[str]:
  rows = []
  for label, key, env in (
    ("final", "final", "training simulator"),
    ("final", "final_stock_env", "stock robosuite"),
    ("best checkpoint", "best", "training simulator"),
  ):
    if key in report:
      r = report[key]
      rows.append(
        f"| {label} | {env} | {r['episodes']} | {_pct(r['success_rate'])} | "
        f"{_pct(r['held_rate'])} |"
      )
  return rows


def _bc_section(run: Path) -> list[str]:
  report = _load(run / "bc_report.json")
  if report is None:
    return []
  evals = _rows(run / "bc_metrics.jsonl")
  during = ", ".join(f"{100 * e['success_rate']:.0f}% at {e['step']}" for e in evals)
  return [
    "## Baseline: behaviour cloning",
    "",
    f"The policy's own actor regressed on {report['demo_steps']} recorded demo "
    f"steps for {report['steps']} updates ({report['train_seconds'] / 60:.1f} min), "
    "with no environment interaction and no world model. Success on "
    f"{evals[0]['episodes'] if evals else 0} episodes during training: {during}.",
    "",
    *_SCORE_HEADER,
    *_score_rows(report),
    "",
  ]


def _policy_section(name: str, run: Path) -> list[str]:
  report = _load(run / "policy_report.json")
  evals = _rows(run / "policy_eval.jsonl")
  metrics = _rows(run / "policy_metrics.jsonl")
  lines = [f"## Stage 2: policy ({name})", ""]
  if report is None:
    lines.append(
      "No final report: the run did not finish. Evaluations logged so far are below."
    )
  else:
    lines.append(
      f"Stopped on `{report['stop_reason']}` after {report['rounds']} rounds, "
      f"{report['env_steps']:,} real transitions, {report['hours']:.2f} h."
    )
    lines += ["", *_SCORE_HEADER, *_score_rows(report)]
  if metrics:
    last = metrics[-1]
    lines += [
      "",
      f"Per round at the end: collect {last['Time/collect']:.2f} s, world model "
      f"{last['Time/wm']:.2f} s, policy {last['Time/policy']:.2f} s; peak GPU memory "
      f"{last.get('gpu_gb', 0):.1f} GiB.",
    ]
  if evals:
    lines += [
      "",
      "| Hours | Real transitions | Eval success | Still successful at the end |",
      "|---|---|---|---|",
    ]
    stride = max(1, len(evals) // 20)
    shown = evals[::stride]
    if shown[-1] is not evals[-1]:
      shown.append(evals[-1])
    for e in shown:
      lines.append(
        f"| {e['hours']:.2f} | {e['env_steps']:,} | {100 * e['success_rate']:.0f}% | "
        f"{100 * e['held_rate']:.0f}% |"
      )
  lines.append("")
  return lines


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", required=True)
  parser.add_argument("--run", required=True)
  args = parser.parse_args()
  run = Path(args.run)

  meta = _load(Path(args.data) / "meta.json")
  task = meta["args"].get("task", "lift") if meta is not None else "task"
  lines = [f"# Run report: {task}", ""]
  if meta is not None:
    lines += _data_section(meta)
  wm = _load(run / "wm_eval.json")
  if wm is not None:
    lines += _wm_section(wm)
  lines += _bc_section(run / "bc")
  for policy_dir in sorted(
    p for p in run.iterdir() if (p / "runner_cfg.json").exists()
  ):
    lines += _policy_section(policy_dir.name, policy_dir)
  (run / "REPORT.md").write_text("\n".join(lines))
  print(f"[INFO] wrote {run / 'REPORT.md'}")


if __name__ == "__main__":
  main()
