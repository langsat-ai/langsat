#!/usr/bin/env python3
"""Rebuild the results table in README.md from every examples/**/results/metrics.json.

    python scripts/build_results_table.py

The numbers in the README are whatever the notebooks last wrote — nothing is typed by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


# what the platform calls an architecture → what a reader recognises
MODEL_LABELS = {"Baseline": "GraphSAGE (default, labelled *Baseline*)", "SAGE": "GraphSAGE (full)", "GAT": "GATv2",
                "GIN": "GIN", "unsupervised": "GraphMAE (self-supervised)"}


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items())
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def main() -> None:
    rows = []
    for f in sorted(ROOT.glob("examples/*/*/results/metrics.json")):
        m = json.loads(f.read_text())
        nb_dir = f.parents[1]
        nb = next(nb_dir.glob("*.ipynb"))
        link = f"[{m.get('notebook', nb.stem)}]({nb.relative_to(ROOT).as_posix()})"
        headline = fmt(m.get("headline"))
        baseline = fmt(m.get("baseline")) if m.get("baseline") else "—"
        train = m.get("training_duration_sec")
        train_s = f"{int(train) // 60} min {int(train) % 60} s" if train else "—"
        credits = m.get("credits_charged_this_run")
        model = MODEL_LABELS.get(str(m.get("model")), m.get("model", ""))
        rows.append(f"| {link} | {m.get('task', '')} | {model} | {headline} | {baseline} | {train_s} | {fmt(credits) if credits is not None else '—'} |")
    table = "\n".join([
        "| notebook | task | model / lane | result (test split) | baseline | training time | credits charged |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ])
    text = README.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"README.md needs the markers {START} … {END}")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), f"{START}\n{table}\n{END}", text, flags=re.S)
    README.write_text(new)
    print(f"wrote {len(rows)} rows into README.md")


if __name__ == "__main__":
    main()
