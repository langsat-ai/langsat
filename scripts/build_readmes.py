"""Per-notebook README.md from the notebook's intro cell + results/metrics.json."""
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1] / "examples" / "amazon_reviews"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return "—" if v is None else str(v)


for d in sorted(ROOT.glob("[0-9][0-9]_*/")):
    nb_path = next(d.glob("*.ipynb"))
    nb = nbformat.read(nb_path, as_version=4)
    intro = nb.cells[0].source.strip()
    m_path = d / "results" / "metrics.json"
    m = json.loads(m_path.read_text()) if m_path.exists() else {}
    lines = [intro, "", f"Notebook: [`{nb_path.name}`]({nb_path.name}) · outputs are from a real run on {m.get('generated_at', '—')[:10]}.", ""]
    if m:
        lines += ["## Result", "", "| | |", "|---|---|",
                  f"| task | {m.get('task')} |", f"| model / lane | {({'Baseline': 'GraphSAGE (default, labelled *Baseline*)', 'SAGE': 'GraphSAGE (full)', 'GAT': 'GATv2', 'GIN': 'GIN', 'TabNet': 'TabNet', 'unsupervised': 'GraphMAE (self-supervised)'}).get(str(m.get('model')), m.get('model'))} |"]
        for k, v in (m.get("headline") or {}).items():
            lines.append(f"| {k} | {fmt(v)} |")
        for k, v in (m.get("baseline") or {}).items():
            lines.append(f"| baseline {k} | {fmt(v)} |")
        if m.get("training_duration_sec"):
            t = int(m["training_duration_sec"]); lines.append(f"| training time | {t // 60} min {t % 60} s |")
        if m.get("credits_charged_this_run") is not None:
            lines.append(f"| credits charged (this run) | {m['credits_charged_this_run']:,} |")
        if m.get("project_id"):
            lines.append(f"| project | `{m['project_id']}` |")
        if m.get("model_id"):
            lines.append(f"| model | `{m['model_id']}` |")
        lines.append("")
        lines.append("Full numbers: [`results/metrics.json`](results/metrics.json).")
    lines += ["", "## Run it", "", "```bash", "pip install -r ../../../requirements.txt && langsat login",
              f"jupyter lab {nb_path.name}", "```", ""]
    (d / "README.md").write_text("\n".join(lines))
    print("wrote", d / "README.md")
