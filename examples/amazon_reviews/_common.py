"""Shared helpers for the Amazon-reviews notebooks.

Every notebook is self-contained and re-runnable: projects are found by name and reused, a model
that already finished is reused instead of retrained, and each notebook writes the numbers it ends
on to `results/metrics.json` (the README table is generated from those files).

    import sys; sys.path.insert(0, "..")
    from _common import connect, get_or_create_project, train_or_reuse, save_metrics
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langsat import Langsat

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "amazon_reviews_10k"
TABLES = ("customer", "product", "review")
PROJECT_PREFIX = "amazon-reviews"


def connect() -> Langsat:
    """A client from `LANGSAT_API_KEY` or the token saved by `langsat login`."""
    ls = Langsat()
    me = ls.me()
    key = me.get("api_key") or {}
    print(f"signed in as {me['email']} · tier {me['tier']} · key '{key.get('name')}' with {len(key.get('scopes') or [])} scopes")
    return ls


def get_or_create_project(ls: Langsat, slug: str, *, kind: str = "data_analysis"):
    """`amazon-reviews-<slug>` with the three CSVs uploaded, the schema detected and the data cleaned.
    Reuses an existing project at whatever step it reached."""
    name = f"{PROJECT_PREFIX}-{slug}"
    p = next((x for x in ls.projects.list() if x.name == name), None)
    if p is None:
        p = ls.projects.create(name, kind=kind)
        print(f"created project {p.id} ({name}, {kind})")
    else:
        print(f"reusing project {p.id} ({name}, status={p.status})")
    p.refresh()
    if not p.data.get("file_names"):
        p.sources.upload(*(DATA_DIR / f"{t}.csv" for t in TABLES))
        p.refresh()
        print("uploaded:", p.data.get("file_names"))
    if not p.data.get("detected_schema"):
        print("detecting schema …")
        p.schema.detect().wait(timeout=900)
        p.refresh()
    # `POST /clean` is the data_analysis step; a data_science project cleans inside its training
    # pipeline (upload → detect → task → train), so there is nothing to call here for it.
    if p.data.get("project_type") == "data_analysis" and not p.data.get("cleaning_decided"):
        print("cleaning (0 credits under 500K rows) …")
        p.cleaning.clean().wait(timeout=1800)
        p.refresh()
    print(f"project ready: status={p.status} · type={p.data.get('project_type')} · files={p.data.get('file_names')}")
    return p


def print_schema(p) -> dict:
    sr = p.schema.result()
    sd = sr["schema_data"]
    print("primary keys :", sd.get("primary_keys"))
    print("foreign keys :", sd.get("foreign_keys"))
    print("time columns :", sd.get("time_cols") or sd.get("time_columns"))
    for t in p.tables.list():
        print(f"  {t['name']}: {t['rows']} rows · {t['columns']} columns")
    return sr


def _finished_model(p) -> dict | None:
    """The newest finished model (the active one wins a tie)."""
    done = [m for m in p.models.list()
            if (m.get("status") or "").lower() in ("complete", "completed", "ready", "active", "deployed") and m.get("metrics")]
    done.sort(key=lambda m: (bool(m.get("is_active")), str(m.get("created_at") or "")), reverse=True)
    return done[0] if done else None


def wait_for_training(p, *, known_model_ids=(), poll: float = 20.0, timeout: float = 5400.0) -> dict:
    """Poll until the run that `train()` just started ends. The signal is the NEW model row
    (`models.list()`): it appears as `training`, becomes `ready` with metrics, or `failed`. The
    project's `status` is printed for progress but is not the decision — a second version keeps
    the project `ready` while it trains."""
    t0 = time.monotonic()
    last = None
    while True:
        st = p.pipeline_status()
        new = [m for m in p.models.list() if m.get("model_id") not in known_model_ids]
        row = new[0] if new else None
        line = f"{st.get('status')} · {st.get('progress_stage')} · {st.get('progress_pct')}% · model {(row or {}).get('status')}"
        if line != last:
            print(f"[{int(time.monotonic() - t0):4d}s] {line}")
            last = line
        if row and (row.get("status") or "").lower() in ("ready", "deployed", "complete", "completed") and row.get("metrics"):
            return row
        if (row and (row.get("status") or "").lower() == "failed") or str(st.get("status")) in ("error", "failed"):
            raise RuntimeError(f"training failed: {st.get('error_message') or row}")
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("training still running after the timeout")
        time.sleep(poll)


def train_or_reuse(p, query: str, *, task_type: str | None = None, subtask_type: str | None = None,
                   max_training_min: int | None = None, enable_text_embedding: bool | None = None,
                   model_key: str | None = None, retrain: bool = False) -> dict:
    """Define the task from plain English, train, and return the finished model row (with `metrics`).
    `task_type` is "supervised" | "unsupervised" (or None to let Langsat classify the intent) and
    `subtask_type` pins the flavour: regression | binary_classification | multiclass_classification |
    clustering | anomaly_detection. `model_key` picks the architecture (None = the default GraphSAGE
    "Baseline"; also "sage_full", "gatv2_full", "gin"). A finished model is reused unless `retrain=True`."""
    import os
    retrain = retrain or os.environ.get("LANGSAT_EXAMPLES_RETRAIN") == "1"   # force a new version
    if not retrain:
        m = _finished_model(p)
        if m:
            print(f"reusing model {m['model_id']} ({m.get('model_type')}, trained {m.get('created_at')})")
            return m
    print("defining the task …")
    job = p.tasks.define(query, task_type=task_type, subtask_type=subtask_type).wait(timeout=600)
    cfg = (job.result() or {}).get("task_config") or p.refresh().data.get("task_config") or {}
    if isinstance(cfg, str):                       # the project row carries it JSON-encoded
        try:
            cfg = json.loads(cfg)
        except ValueError:
            cfg = {}
    print(json.dumps({k: cfg.get(k) for k in ("task_type", "entity_table", "entity_col", "target_col", "sql_type",
                                              "text_embed_columns", "explanation", "unsupervised_type")}, indent=2, default=str))
    est = p.estimates().get("train") or {}
    print("estimate:", est)
    # The platform floors the run at what the estimate says this dataset needs (text embedding
    # time + a training floor); asking for less is refused, unused minutes are refunded.
    minutes = max(int(max_training_min or 0), int(est.get("suggested_minutes") or 0) + 2) or None
    print(f"training … (max {minutes} min; credits are reserved for that and refunded for unused minutes)")
    known = tuple(x.get("model_id") for x in p.models.list())
    try:
        p.train(max_training_min=minutes, enable_text_embedding=enable_text_embedding, model_key=model_key)
    except Exception as e:                        # the floor the API states wins over the estimate
        import re
        found = re.search(r"required floor (\d+) min", str(e))
        if not found:
            raise
        minutes = int(found.group(1)) + 1
        print(f"the API asks for at least {minutes - 1} min — retrying with {minutes}")
        p.train(max_training_min=minutes, enable_text_embedding=enable_text_embedding, model_key=model_key)
    m = wait_for_training(p, known_model_ids=known)
    if not m.get("is_active"):
        p.models.activate(m["model_id"])          # the version the app and /predict serve by default
    print(f"model {m['model_id']} · {m.get('version_label')} · {m.get('model_type')} · {m.get('training_duration_sec')} s")
    return m


def credits_used(ls: Langsat) -> int:
    """Credits used this month on the account that pays (the org pool for a team member)."""
    d = ls.credits.dashboard()
    pool = d.get("org_pool") or {}
    return int(pool.get("credits_used_this_month") if pool else d.get("credits_used_this_month") or 0)


def save_metrics(notebook_dir: str | Path, payload: dict) -> Path:
    out = Path(notebook_dir) / "results" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {out}")
    return out


def show(d: dict, keys: tuple[str, ...] | None = None, digits: int = 4) -> None:
    for k, v in d.items():
        if keys and k not in keys:
            continue
        if isinstance(v, float):
            v = round(v, digits)
        print(f"  {k:<22} {v}")
