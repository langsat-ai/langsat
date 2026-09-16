"""Generate the 12 Amazon-reviews notebooks (book style; charts as images). Regenerating WIPES outputs — run `scripts/run_all.sh` afterwards. `NB_ONLY=11,12` limits it."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1] / "examples" / "amazon_reviews"

PREAMBLE = '''import sys, json
sys.path.insert(0, "..")
from _common import connect, get_or_create_project, print_schema, train_or_reuse, credits_used, save_metrics, show, fig, metrics_table, project_models
from langsat import viz
import pandas as pd

ls = connect()'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


import os
ONLY = os.environ.get("NB_ONLY", "").split(",") if os.environ.get("NB_ONLY") else None


def write(folder, filename, cells):
    if ONLY and folder[:2] not in ONLY:
        return
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
    nb.cells = cells
    d = ROOT / folder
    d.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, d / filename)
    print("wrote", d / filename)


def intro(title, usecase, learn, cost, lane=None):
    items = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(learn))
    lane_md = f"\n**Model / lane:** {lane}\n" if lane else ""
    return md(f"""
# {title}

**Use case.** {usecase}
{lane_md}
**What you will learn**
{items}

**What this costs.** {cost}

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.
""")


def next_steps(*lines):
    return md("## Where to go next\n\n" + "\n".join(f"- {l}" for l in lines))


COST_TRAIN = ("one GPU training run (a few minutes on `g4dn.2xlarge`; the API reserves the estimate — about 8,700 credits on the Team plan — "
              "and refunds the unused minutes, so a 90-second run ends up costing a few hundred), plus 50 credits per on-demand prediction. "
              "A re-run that finds the finished model pays only for its predictions.")

# ══════════════════════════════════════════════════════════════════════════════════════════
# 01
# ══════════════════════════════════════════════════════════════════════════════════════════
write("01_setup_and_explore", "01_setup_and_explore.ipynb", [
    intro("01 · Set up a project and explore the data",
          "A data team gets three CSV exports from a bookstore's review system — `customer`, `product`, `review` — and wants them queryable, "
          "joined and charted in minutes, with no ETL and no notebook full of `pd.merge`. This is the notebook every other one builds on.",
          ["Sign in with an API key and see who the key acts as",
           "Create a project and upload the three CSVs — the API counts rows and enforces your plan's row cap",
           "Let Langsat detect primary keys, foreign keys and the time column",
           "See what a clean would cost, then clean (free under 500K rows)",
           "Read rows into pandas — filters, sorting, search, paging",
           "Build charts on a dashboard, draw them **as images** in the notebook, publish a public link",
           "Let the AI build a dashboard (`generate_cards`), and ask questions in plain English — with a chart you can plot from code"],
          "no training. Cleaning is 0 credits under 500K rows; `generate_cards` and each `ask` count against your AI quota (not credits)."),
    md("## 1 · Sign in\n\n`Langsat()` reads the key that `langsat login` saved (or `LANGSAT_API_KEY`). A key *is* you: your plan's limits, your credits, your team role — the API refuses anything the key's scopes don't cover, with a typed error."),
    code(PREAMBLE),
    code('''me = ls.me()
print(f"{me['email']} · plan {me['tier']} · teams: {[t['name'] for t in me.get('teams', [])]}")
print("this key may:", ", ".join(me["api_key"]["scopes"]))'''),
    md("## 2–4 · Project, upload, schema, clean\n\n`get_or_create_project` (in `_common.py`) is thirty lines you could inline: find the project by name or create it, `sources.upload(*paths)` (presign → S3 → confirm), `schema.detect().wait()`, `cleaning.clean().wait()` — skipping any step the project already did. Watch the log: it says which steps ran."),
    code('''p = get_or_create_project(ls, "explore", kind="data_analysis")
sr = print_schema(p)'''),
    md("""Langsat found the relationships itself: `review.customer_id → customer`, `review.product_id → product`, and `review_time` as the time column — nothing was declared. `review` got a synthetic primary key (`__row_id__`) because a review has no natural id.

`estimates()` says what each paid action would cost *before* you run it. For this project cleaning is free (under 500K rows), and an AI question costs one unit of your daily quota."""),
    code('''est = p.estimates()
print("clean :", est["clean"])
print("ask   :", est["ask"])
print("predict:", est["predict"])'''),
    md("## 5 · Rows into pandas\n\n`table(name)` reads the *cleaned* data (types normalised — `verified` became 0/1). `rows()` is one page (≤200), `iter_rows()` walks pages, `to_pandas()` collects up to 100K rows. Filters are `(column, op, value)` triples the server applies — the same operators the app's filter bar uses."),
    code('''review = p.table("review").to_pandas()
product = p.table("product").to_pandas()
customer = p.table("customer").to_pandas()
print(review.shape, product.shape, customer.shape)
review.head(3)'''),
    code('''# server-side filter + sort + search: 5-star reviews that mention "kids", newest first
page = p.table("review").rows(filters=[("rating", "eq", 5)], search="kids", sort=[("review_time", "desc")], page_size=5)
print(page.total, "matching reviews; first page:")
page.to_pandas()[["review_time", "rating", "summary"]]'''),
    code('''print("rating distribution:"); print(review["rating"].value_counts().sort_index().to_dict())
print(f"verified purchases: {review['verified'].astype(int).mean():.1%}")
print(f"reviews per customer: max {review.customer_id.value_counts().max()} · products with ≥5 reviews: {(review.product_id.value_counts() >= 5).sum()}")'''),
    md("A first look at time. `review_time` spans 2008 → September 2018; the count per year and the mean rating side by side."),
    code('''import matplotlib.pyplot as plt
review["review_time"] = pd.to_datetime(review["review_time"])
by_year = review.groupby(review["review_time"].dt.year)["rating"].agg(["count", "mean"])
f, ax = plt.subplots(figsize=(8, 3))
by_year["count"].plot(kind="bar", ax=ax, color="#c4a35a", title="Reviews per year (bars) and mean rating (line)")
by_year["mean"].plot(ax=ax.twinx(), color="#c0392b", marker="o"); fig(f)'''),
    md("""## 6 · A dashboard, drawn as images

A **recipe chart** is deterministic — tables, a grouping, a measure, optional joins and a Top N — no AI, no credits, and the server re-aggregates it on every data refresh. When a grouping has thousands of values (6,000+ brands here) the API refuses an unreadable axis and tells you to add a Top N.

`langsat.viz` turns any chart the platform produces into a matplotlib figure — the same Plotly JSON the app renders, drawn as a PNG so it shows in GitHub, in a PDF, in an email."""),
    code('''tab = next((t for t in p.dashboards.list() if t.name == "Reviews"), None) or p.dashboards.create("Reviews")
tab.refresh()
have = {c.title for c in tab.charts}
if "Reviews by rating" not in have:
    tab.add_chart({"tables": ["review"], "chart_type": "bar", "group_by": [{"column": "rating"}],
                   "measure": {"agg": "count", "column": "rating"}}, title="Reviews by rating")
if "Most-reviewed brands (joined, top 10)" not in have:
    tab.add_chart({"tables": ["review", "product"], "chart_type": "bar", "group_by": [{"column": "brand"}],
                   "measure": {"agg": "count", "column": "rating"},
                   "joins": [{"table": "product", "left_column": "product_id", "right_column": "product_id"}],
                   "order_by": {"dir": "desc", "limit": 10}},                         # 6,000+ brands → Top 10
                  title="Most-reviewed brands (joined, top 10)")
if "Mean rating by year" not in have:
    tab.add_chart({"tables": ["review"], "chart_type": "scatter", "mode": "lines",
                   "group_by": [{"column": "review_time", "transform": "to_year"}],
                   "measure": {"agg": "avg", "column": "rating"}}, title="Mean rating by year")
tab.refresh()
for c in tab.charts:
    print(c.id, "·", c.title, "·", (c.recipe or {}).get("chart_type"))'''),
    code('''fig(viz.draw_tab(tab, cols=3))                     # every widget of the tab, re-aggregated by the server'''),
    md("Behind each chart is the data the dashboard shows — `chart.data(op=\"page\")` gives the rows, `op=\"aggregate\"` the Plotly render. Useful when you want the numbers, not the picture."),
    code('''chart = tab.chart("Most-reviewed brands (joined, top 10)")
pd.DataFrame(chart.data(op="page", page_size=10)["rows"])[["x", "y"]].rename(columns={"x": "brand", "y": "reviews"})'''),
    md("### Publish a link\n\nA **published** share is a frozen copy of the tab whose numbers still follow data refreshes. The token is returned **once**, at creation; the page it opens is also what an `<iframe>` embeds. Below: the tab's existing published share is revoked and a fresh one minted (so the notebook can show the link), then the same widgets are re-read from the public JSON — no key needed — to prove the share is self-contained."),
    code('''for s in tab.shares():
    if s.get("mode") == "published":
        tab.revoke_share(s["share_id"])                # the old link stops working; the new one below replaces it
link = tab.share(mode="published")
token = link["token"]
print("public page :", tab.embed_url(token))
import httpx
public = httpx.get(f"{ls.base_url}/public/share/{token}", timeout=30).json()
print("public JSON :", public.get("tab_name"), "·", len(public.get("widgets", [])), "widgets · published", str((public.get("published") or {}).get("published_at"))[:19])
print("widgets     :", [w.get("title") for w in public.get("widgets", [])])'''),
    md("""### Let the AI build a dashboard

`generate_cards()` is the app's **Add Dashcard** button: it fills a tab with a data preview and five charts about the next table nobody has explored yet (customer, then product, then review), renames the tab after that table, and — once every table has a tab — switches to cross-table charts. One dashboard-generation unit of your quota per call. The cards are recipe charts too: you can read their recipes, they refresh with the data, and `viz.draw_tab` draws them like any other tab."""),
    code('''explored = [t for t in p.dashboards.list() if any(w.get("type") == "data_table" for w in t.refresh().widgets)]   # list() is a summary; refresh() loads the widgets
if not explored:
    ai = p.dashboards.create("AI cards").generate_cards().refresh()      # becomes the "customer" tab
    explored = [ai]
for t in explored:
    print(f"tab {t.name!r}: {len(t.charts)} cards →", [c.title for c in t.charts][:3], "…")
fig(viz.draw_tab(explored[-1], cols=3))'''),
    md("A table-preview card is drawn as a table; the recipe behind any card is on `chart.recipe` — the same JSON `add_chart` accepts, so an AI card can be copied to your own tab or edited."),
    code('''card = next(c for c in explored[-1].charts if c.recipe)
json.dumps(card.recipe, indent=1)[:600]'''),
    md("""## 7 · Ask the data — and plot the answer from code

`ask()` is the app's chat. The answer carries the text, how it was produced (`execution_kind`: SQL, code, or a model) and any **chart specs** — Plotly figures you can draw with `viz` or pin to a dashboard with `tab.add_chart_spec`. `force_chart=True` tells the model a chart is required."""),
    code('''a = p.ask("Which year had the lowest average rating, and how many reviews were written that year?", timeout=300)
print(a.text)
print("\\nexecution:", a.execution_kind, "· charts:", len(a.charts))'''),
    code('''b = p.ask("Show the number of reviews per rating as a bar chart", force_chart=True, timeout=300)
print(b.text[:300] or "(the answer is the chart itself)")
print("execution:", b.execution_kind, "· chart specs:", [c.get("title") for c in b.charts], "· plots:", len(b.plots))
for f in b.figures():
    fig(f)'''),
    md("Pin it. The chart becomes a widget on the tab, exactly as the app's *Add to dashboard* does — and a follow-up question can reference the conversation."),
    code('''if b.charts and "Number of reviews per rating (from chat)" not in {c.title for c in tab.charts}:
    tab.add_chart_spec(b.charts[0], title="Number of reviews per rating (from chat)")
print([c.title for c in tab.charts])
c2 = p.ask("And what share of those are verified purchases, per rating?", conversation_id=b.conversation_id, timeout=300)
print(c2.text[:400])'''),
    code('''save_metrics(".", {"notebook": "01_setup_and_explore", "task": "setup + explore (no model)", "model": "—",
                   "project_id": p.id, "tables": {t["name"]: t["rows"] for t in p.tables.list()},
                   "foreign_keys": sr["schema_data"]["foreign_keys"], "charts": [c.title for c in tab.charts], "ai_tabs": [t.name for t in explored],
                   "headline": {"published_link": bool(token), "chat_chart": bool(b.charts), "answer_chars": len(a.text)}})'''),
    next_steps("[02 · Rating regression](../02_rating_regression/) — the first model on this data",
               "[07 · Forecast](../07_forecast_review_volume/) — the same project, no training",
               "The SDK reference: `help(ls.projects)`, `help(p.table('review'))`, `help(langsat.viz)`"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# shared modeling cells
# ══════════════════════════════════════════════════════════════════════════════════════════
SETUP_DS = '''p = get_or_create_project(ls, "{slug}", kind="data_science")
before = credits_used(ls)'''

MODEL_TAB_MD = """## The Model tab, as images

After training the app builds a **Model** tab: overview, feature importance, training curves and the task-specific
cards. It is a normal dashboard tab whose widgets carry Plotly JSON, so `viz.model_dashboard(p)` draws exactly what the
app shows — and `viz.model_cards(p)` lets you pick one card. The Model Overview table carries numbers the API's
`metrics` dict does not (they live only in the trained model's metadata)."""


def train_md(extra=""):
    return md("## Define the task, train\n\nThe task is one sentence. `train_or_reuse` prints the config Langsat derived (entity table, target, which text columns get embedded), the estimate, then polls the run and reuses a finished model on re-runs. " + extra)


# ══════════════════════════════════════════════════════════════════════════════════════════
# 02 regression
# ══════════════════════════════════════════════════════════════════════════════════════════
write("02_rating_regression", "02_rating_regression.ipynb", [
    intro("02 · Predict a review's rating (regression)",
          "The support team wants a predicted star rating for every review as it arrives — from its text, its summary and the product it is about — as a number they can rank on, not a class.",
          ["Create a `data_science` project on the three tables (no clean step — the training pipeline cleans)",
           "Define the task in plain English and read the config Langsat derived",
           "Train a relational GNN with text embeddings, watch the run",
           "Read the test metrics — MAE, RMSE, R², Spearman — against an always-the-mean baseline",
           "See the Model tab as images: overview, feature importance, training curves, MAE/RMSE per epoch",
           "Rank every review by predicted rating, score one on demand, look at residuals"],
          COST_TRAIN,
          lane="relational GNN — the default GraphSAGE architecture (Langsat labels it *Baseline*), text embeddings on `review_text`, `summary`, `product.title` / `description`"),
    code(PREAMBLE),
    code(SETUP_DS.format(slug="rating-regression")),
    train_md("`task_type=\"supervised\"` + `subtask_type=\"regression\"` pins the flavour; leave both out and Langsat classifies the intent itself."),
    code('''model = train_or_reuse(p, "Predict the rating a review gives, from the review text, its summary and the product it is about",
                       task_type="supervised", subtask_type="regression", enable_text_embedding=True)
metrics = model["metrics"]
show(metrics)'''),
    md("## Baseline\n\nAlways predicting the mean rating: what a model must beat. On 10,000 heavily 5★-skewed reviews the *ranking* metric is the one to read — a Spearman of ~0.5 means the model orders reviews from worst to best far better than chance even where the absolute star value is off by half a star."),
    code('''review = p.table("review").to_pandas()
mean = review["rating"].mean()
baseline_mae = (review["rating"] - mean).abs().mean()
baseline_rmse = ((review["rating"] - mean) ** 2).mean() ** 0.5
print(f"always-mean baseline · MAE {baseline_mae:.3f} · RMSE {baseline_rmse:.3f} · R² 0.000")
print(f"model (test split)    · MAE {metrics.get('mae'):.3f} · RMSE {metrics.get('rmse'):.3f} · R² {metrics.get('r2'):.3f} · Spearman {metrics.get('spearman', float('nan')):.3f}")'''),
    md(MODEL_TAB_MD),
    code('''fig(viz.model_dashboard(p, cols=2))'''),
    code('''cards = viz.model_cards(p)
overview = viz.card_table(cards["Model Overview"])
{k: overview[k] for k in list(overview)[:12]}'''),
    md("The training run's epoch history is also on `GET /projects/{id}/status` — the numbers behind the curves, as a table."),
    code('''hist = pd.DataFrame(p.pipeline_status().get("epoch_history") or [])
hist[["epoch", "train_loss", "val_loss", "epoch_time_sec"]].tail(5)'''),
    md("## Rank, score, explain\n\n`predict.top` ranks every review by its stored predicted rating; `order=\"asc\"` gives the other end. `predict.predict` scores one entity on demand (50 credits). Joining the two ends back to the review table shows what the model thinks a bad and a good review look like."),
    code('''best = ls.predict.top(model["model_id"], n=50)
worst = ls.predict.top(model["model_id"], n=50, order="asc")
print("ranked by:", best.get("ranked_by"), "· total ranked:", best.get("total_ranked"))
ends = pd.DataFrame([{"entity_id": r["entity_id"], "predicted": r["score"], "end": "top 50"} for r in best["ranking"]] +
                    [{"entity_id": r["entity_id"], "predicted": r["score"], "end": "bottom 50"} for r in worst["ranking"]])
joined = ends.merge(review.reset_index().rename(columns={"index": "entity_id"}), on="entity_id", how="left")
print(joined.groupby("end")[["predicted", "rating"]].mean().round(2))
joined[["end", "predicted", "rating", "summary"]].head(6)'''),
    code('''import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(6, 3.5))
for end, grp in joined.groupby("end"):
    ax.scatter(grp["rating"], grp["predicted"], alpha=0.6, label=end)
ax.set_xlabel("actual rating"); ax.set_ylabel("predicted rating"); ax.legend(frameon=False)
ax.set_title("Predicted vs actual at both ends of the ranking", loc="left", fontweight="bold"); fig(f)'''),
    code('''one = ls.predict.predict(model_id=model["model_id"], entity_id=best["ranking"][0]["entity_id"])
print("predicted rating:", round(one["result"]["prediction"], 3), "· task:", one["task_type"], "· model:", one["version_label"])'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "02_rating_regression", "task": "regression · review.rating", "model": model.get("model_type"),
                   "project_id": p.id, "model_id": model["model_id"], "training_duration_sec": model.get("training_duration_sec"),
                   "credits_charged_this_run": charged,
                   "headline": {"mae": metrics.get("mae"), "rmse": metrics.get("rmse"), "r2": metrics.get("r2"), "spearman": metrics.get("spearman")},
                   "baseline": {"mae": round(baseline_mae, 4), "rmse": round(baseline_rmse, 4), "r2": 0.0}})'''),
    next_steps("[03 · Rating classification](../03_rating_classification/) — the same target as five classes, another architecture",
               "[12 · Training options](../12_training_options/) — text embeddings on/off and the architecture bake-off, on this project",
               "[11 · Fine-tune](../11_finetune/) — new reviews arrive; continue this model instead of retraining"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 03 multiclass
# ══════════════════════════════════════════════════════════════════════════════════════════
write("03_rating_classification", "03_rating_classification.ipynb", [
    intro("03 · Predict the star rating as a class (1–5)",
          "The same question as 02, but the product team wants class probabilities — *how sure* is the model this is a 1★ review? — to route angry customers to a person before the review goes live.",
          ["Define the task as **multiclass** classification and pick an architecture (`model_key=\"gatv2_full\"`)",
           "Accuracy and macro-F1 on the test split vs the always-5★ baseline (63% of reviews are 5★)",
           "The Model tab: confusion matrix, accuracy/F1 per epoch — as images",
           "Per-class recall from the confusion matrix",
           "The 1★ queue: rank every review by P(1★) and read the top of it"],
          COST_TRAIN,
          lane="relational GNN — GATv2 (`model_key=\"gatv2_full\"`), 5 classes, text embeddings"),
    code(PREAMBLE),
    code(SETUP_DS.format(slug="rating-classes")),
    train_md("`model_key` picks the architecture; the default is GraphSAGE. GATv2 weighs neighbours (the product's other reviews, the customer's other reviews) with attention."),
    code('''model = train_or_reuse(p, "Classify each review into its star rating from 1 to 5 using the review text, the summary and the product",
                       task_type="supervised", subtask_type="multiclass_classification", enable_text_embedding=True, model_key="gatv2_full")
metrics = model["metrics"]
show(metrics, keys=("acc", "f1_macro", "precision_macro", "recall_macro"))'''),
    code('''review = p.table("review").to_pandas()
majority = review["rating"].value_counts(normalize=True).max()
print(f"always-5★ baseline · accuracy {majority:.3f} · macro-F1 {(2*majority)/(majority+1)/5:.3f} (only one class ever right)")
print(f"model (test split)  · accuracy {metrics.get('acc'):.3f} · macro-F1 {metrics.get('f1_macro'):.3f}")'''),
    md(MODEL_TAB_MD),
    code('''fig(viz.model_dashboard(p, cols=2))'''),
    md("The confusion matrix card carries the counts (the API's `metrics` dict is scalars only). Rows are actual classes, columns predicted — per-class recall is the diagonal over the row sum. Expect the 5★ class to dominate and the 2★/3★ classes to be hardest: they are rare and sit between neighbours."),
    code('''cm_card = viz.model_cards(p)["Confusion Matrix"]
tr = cm_card["plotly"]["data"][0]
cm = pd.DataFrame(tr["z"], index=[f"actual {y}" for y in tr["y"]], columns=[f"pred {x}" for x in tr["x"]])
display(cm)
recall = pd.Series({str(tr["y"][i]): row[list(tr["x"]).index(tr["y"][i])] / max(sum(row), 1) for i, row in enumerate(tr["z"])}).sort_index()
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(5, 3)); recall.plot(kind="bar", ax=ax, color="#c4a35a"); ax.set_ylim(0, 1); ax.set_ylabel("recall")
ax.set_title("Recall per star", loc="left", fontweight="bold"); fig(f)'''),
    md("## The 1★ queue\n\nA multiclass model carries five probabilities per row, so a ranking needs to say which class: `class_index=0` is the first label (1★). That list is the routing queue for the support team — with the text, so a person can act on it."),
    code('''labels = metrics.get("class_labels") or [1, 2, 3, 4, 5]
top = ls.predict.top(model["model_id"], n=8, class_index=0)             # rank by P(rating == labels[0])
print("ranked by:", top.get("ranked_by"), "· class", labels[0], "★ ·", top.get("total_ranked"), "ranked")
queue = pd.DataFrame([{"review": r["entity_id"], "P(1★)": round(r["score"], 3)} for r in top["ranking"]])
queue = queue.merge(review.reset_index().rename(columns={"index": "review"}), on="review", how="left")
queue[["review", "P(1★)", "rating", "summary"]]'''),
    code('''one = ls.predict.predict(model_id=model["model_id"], entity_id=top["ranking"][0]["entity_id"])
probs = one["result"].get("probabilities") or {}
print("prediction:", one["result"]["prediction"], "· probabilities:", {k: round(v, 3) for k, v in probs.items()})
f, ax = plt.subplots(figsize=(4, 2.6)); ax.bar(list(probs), list(probs.values()), color="#c0392b"); ax.set_ylim(0, 1)
ax.set_title("Class probabilities for the top-ranked review", loc="left", fontweight="bold"); fig(f)'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "03_rating_classification", "task": "multiclass · review.rating (5 classes)", "model": model.get("model_type"),
                   "project_id": p.id, "model_id": model["model_id"], "training_duration_sec": model.get("training_duration_sec"),
                   "credits_charged_this_run": charged,
                   "headline": {"accuracy": metrics.get("acc"), "f1_macro": metrics.get("f1_macro"), "recall_1star": round(float(recall.iloc[0]), 4)},
                   "baseline": {"accuracy": round(float(majority), 4), "f1_macro_note": "always-5★"}})'''),
    next_steps("[04 · Verified purchase](../04_verified_purchase_binary/) — a binary target, ROC and PR-AUC",
               "[08 · Predict API & monitoring](../08_predict_api_and_monitoring/) — serving a model from an application"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 04 binary
# ══════════════════════════════════════════════════════════════════════════════════════════
write("04_verified_purchase_binary", "04_verified_purchase_binary.ipynb", [
    intro("04 · Is this a verified purchase? (binary classification)",
          "Trust & safety wants to flag reviews that read like unverified ones — a probability per review that can rank the whole table and set a threshold.",
          ["Define a **binary** task on `review.verified`",
           "AUROC, PR-AUC (vs the class prior), precision / recall / F1 — what each floor means",
           "The Model tab: the ROC curve and the confusion matrix as images",
           "Rank every review both ways; score a batch; look at the probability distribution",
           "Choose a threshold: what the confusion matrix says about the default 0.5"],
          COST_TRAIN,
          lane="relational GNN — default GraphSAGE (*Baseline*), text embeddings"),
    code(PREAMBLE),
    code(SETUP_DS.format(slug="verified")),
    train_md(),
    code('''model = train_or_reuse(p, "Predict whether a review is a verified purchase, from the review text, the summary, the rating and the product",
                       task_type="supervised", subtask_type="binary_classification", enable_text_embedding=True)
metrics = model["metrics"]
show(metrics, keys=("auroc", "pr_auc", "pr_auc_baseline", "acc", "precision", "recall", "f1"))'''),
    md("Read the two AUCs against their floors: AUROC's floor is 0.5 whatever the class balance; PR-AUC's floor is the **positive rate** (71% of reviews are verified), so `pr_auc / pr_auc_baseline` is the lift."),
    code('''print(f"AUROC {metrics['auroc']:.3f} (random 0.500) · PR-AUC {metrics['pr_auc']:.3f} vs prior {metrics['pr_auc_baseline']:.3f} → lift ×{metrics['pr_auc']/metrics['pr_auc_baseline']:.2f} · F1 {metrics['f1']:.3f}")'''),
    md(MODEL_TAB_MD),
    code('''fig(viz.model_dashboard(p, cols=2))'''),
    md("The confusion matrix at the default 0.5 cut: with 71% positives the model leans toward *verified*. If the team wants fewer false alarms they raise the threshold and rank with `order=\"asc\"` instead — the ROC curve is the menu of trade-offs."),
    code('''cm = viz.model_cards(p)["Confusion Matrix"]["plotly"]["data"][0]
pd.DataFrame(cm["z"], index=[f"actual {y}" for y in cm["y"]], columns=[f"pred {x}" for x in cm["x"]])'''),
    md("## Rank every review, score a batch\n\n`predict.top` orders by the positive-class probability; `predict_batch` scores a list of ids in one call (50 credits each). The histogram of scored probabilities shows how decisive the model is."),
    code('''top = ls.predict.top(model["model_id"], n=10)
least = ls.predict.top(model["model_id"], n=10, order="asc")
print("most likely verified :", [(r["entity_id"], round(r["score"], 3)) for r in top["ranking"][:5]])
print("least likely verified:", [(r["entity_id"], round(r["score"], 3)) for r in least["ranking"][:5]])'''),
    code('''ids = [r["entity_id"] for r in top["ranking"][:10]] + [r["entity_id"] for r in least["ranking"][:10]]
batch = ls.predict.predict_batch(model_id=model["model_id"], entity_ids=ids)
probs = [pr.get("probability") for pr in batch["predictions"] if pr.get("probability") is not None]
print(batch["succeeded"], "of", batch["total"], "scored")
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(5, 2.8)); ax.hist(probs, bins=10, color="#c4a35a"); ax.axvline(0.5, color="#c0392b", ls="--")
ax.set_xlabel("P(verified)"); ax.set_title("Scored probabilities (10 top + 10 bottom)", loc="left", fontweight="bold"); fig(f)'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "04_verified_purchase_binary", "task": "binary · review.verified", "model": model.get("model_type"),
                   "project_id": p.id, "model_id": model["model_id"], "training_duration_sec": model.get("training_duration_sec"),
                   "credits_charged_this_run": charged,
                   "headline": {"auroc": metrics.get("auroc"), "pr_auc": metrics.get("pr_auc"), "f1": metrics.get("f1"), "accuracy": metrics.get("acc")},
                   "baseline": {"auroc": 0.5, "pr_auc": metrics.get("pr_auc_baseline")}})'''),
    next_steps("[08 · Predict API & monitoring](../08_predict_api_and_monitoring/) — this model behind an application",
               "[05 · Product clustering](../05_product_clustering/) — no labels at all"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 05 clustering
# ══════════════════════════════════════════════════════════════════════════════════════════
write("05_product_clustering", "05_product_clustering.ipynb", [
    intro("05 · Segment products (clustering, unsupervised)",
          "Merchandising has 8,600 books and no taxonomy beyond 'Books' — they want natural segments from titles, descriptions, prices and *who reviews them*, without labelling anything.",
          ["Define an unsupervised **clustering** task on `product`",
           "Train GraphMAE (self-supervised on the relational graph) and cluster the embeddings",
           "Read cluster quality from the Model tab: silhouette, Davies-Bouldin, sizes, a 2-D projection — as images",
           "Assign a sample of products and read the segments by their members"],
          COST_TRAIN,
          lane="GraphMAE (self-supervised) + clustering on the node embeddings"),
    code(PREAMBLE),
    code(SETUP_DS.format(slug="product-clusters")),
    train_md("For an unsupervised task the flavour is `subtask_type=\"clustering\"`."),
    code('''model = train_or_reuse(p, "Group products into natural segments based on their title, description, price and how they are reviewed",
                       task_type="unsupervised", subtask_type="clustering", enable_text_embedding=True)
metrics = model["metrics"]
show(metrics)'''),
    md("GraphMAE is self-supervised: the run reports its masked-reconstruction **validation loss** (lower = the embedding explains the graph better). The cluster *quality* numbers live on the Model tab — silhouette (−1…1, higher = tighter, better separated), Davies-Bouldin (lower is better), the size balance, and a PCA projection of the embeddings."),
    md(MODEL_TAB_MD),
    code('''fig(viz.model_dashboard(p, cols=2))'''),
    code('''quality = viz.card_table(viz.model_cards(p)["Cluster Quality"])
quality'''),
    md("## Read the segments\n\nScore a sample of products (50 credits each — 40 here) and describe each cluster by its members: titles, median price, how many reviews they attract."),
    code('''product = p.table("product").to_pandas()
review = p.table("review").to_pandas()
sample = product.sample(40, random_state=7)
ids = sample["product_id"].tolist()
batch = ls.predict.predict_batch(model_id=model["model_id"], entity_ids=ids)
sample = sample.assign(cluster=[pr.get("cluster", pr.get("prediction")) for pr in batch["predictions"]])
sample = sample.merge(review.groupby("product_id")["rating"].agg(reviews="count", mean_rating="mean"), left_on="product_id", right_index=True, how="left")
print(sample.groupby("cluster").agg(products=("product_id", "count"), median_price=("price", "median"), reviews=("reviews", "mean"), mean_rating=("mean_rating", "mean")).round(2))
for cl, grp in sample.groupby("cluster"):
    print(f"\\ncluster {cl}:")
    for t in grp["title"].head(4): print("   ", t[:80])'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "05_product_clustering", "task": "clustering · product (unsupervised)", "model": model.get("model_type"),
                   "project_id": p.id, "model_id": model["model_id"], "training_duration_sec": model.get("training_duration_sec"),
                   "credits_charged_this_run": charged,
                   "headline": {"val_loss": metrics.get("val_loss"), "silhouette": quality.get("Silhouette"), "davies_bouldin": quality.get("Davies-Bouldin"), "clusters": quality.get("Clusters"),
                                "note": "quality numbers come from the Model tab cards"}})'''),
    next_steps("[06 · Review anomaly detection](../06_review_anomaly/) — the other unsupervised task"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 06 anomaly
# ══════════════════════════════════════════════════════════════════════════════════════════
write("06_review_anomaly", "06_review_anomaly.ipynb", [
    intro("06 · Find unusual reviews (anomaly detection, unsupervised)",
          "The marketplace team wants a first pass at suspicious reviews — text that does not fit the product, ratings out of line with the words — without labelled examples.",
          ["Define an unsupervised **anomaly detection** task on `review`",
           "Train GraphMAE and score reviews by reconstruction error",
           "Read the anomaly score distribution and the threshold from the Model tab — as an image",
           "Score a sample: which are flagged, with what score, and what they say"],
          COST_TRAIN,
          lane="GraphMAE (self-supervised), anomaly scoring on the node embeddings"),
    code(PREAMBLE),
    code(SETUP_DS.format(slug="review-anomalies")),
    train_md(),
    code('''model = train_or_reuse(p, "Detect unusual or suspicious reviews based on their text, rating and the product and customer they belong to",
                       task_type="unsupervised", subtask_type="anomaly_detection", enable_text_embedding=True)
metrics = model["metrics"]
show(metrics)'''),
    md(MODEL_TAB_MD),
    code('''fig(viz.model_dashboard(p, cols=2))'''),
    md("The **Anomaly Score Distribution** card is the one to read: the dashed line is the cut the run chose; everything to its right is flagged. The card's description quotes the score separation (how far the flagged mean sits from the normal mean)."),
    code('''card = viz.model_cards(p)["Anomaly Score Distribution"]
print(card.get("description"))
threshold = card["plotly"]["layout"]["shapes"][0]["x0"]
print("threshold:", round(threshold, 4))
fig(viz.draw(card, figsize=(8, 3.5)))'''),
    md("## Score a sample\n\nAnomaly detection is self-supervised too: a review whose embedding the model reconstructs badly is unusual for its neighbourhood (its product, its customer, its words). Score 30 reviews (50 credits each) and look at the flagged ones."),
    code('''review = p.table("review").to_pandas()
sample = review.sample(30, random_state=7)
ids = sample.index.tolist()                      # review's key is the synthetic row id
batch = ls.predict.predict_batch(model_id=model["model_id"], entity_ids=ids)
sample = sample.assign(is_anomaly=[pr.get("is_anomaly") for pr in batch["predictions"]],
                       score=[pr.get("anomaly_score") for pr in batch["predictions"]])
print("flagged:", int(sample["is_anomaly"].fillna(False).astype(bool).sum()), "of", len(sample), "· threshold:", round((batch["predictions"][0] or {}).get("threshold", 0), 4))
for _, r in sample.sort_values("score", ascending=False).head(6).iterrows():
    print(f"  {'⚑' if r['is_anomaly'] else ' '} score {r['score']:.3f} · {r['rating']}★ · {str(r['summary'])[:40]!r} · {str(r['review_text'])[:90]!r}")'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "06_review_anomaly", "task": "anomaly detection · review (unsupervised)", "model": model.get("model_type"),
                   "project_id": p.id, "model_id": model["model_id"], "training_duration_sec": model.get("training_duration_sec"),
                   "credits_charged_this_run": charged,
                   "headline": {"val_loss": metrics.get("val_loss"), "threshold": round(float(threshold), 4), "flagged_in_sample": int(sample["is_anomaly"].fillna(False).astype(bool).sum()), "sample_size": len(sample)}})'''),
    next_steps("[07 · Forecast](../07_forecast_review_volume/) — zero-shot, no training at all"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 07 forecast
# ══════════════════════════════════════════════════════════════════════════════════════════
write("07_forecast_review_volume", "07_forecast_review_volume.ipynb", [
    intro("07 · Forecast monthly review volume and rating (zero-shot, no training)",
          "Community managers staff the moderation queue a few months ahead — how many reviews will arrive per month, and is the average rating drifting? Nothing is trained: Chronos-2 forecasts the project's own series on the spot.",
          ["Forecast the monthly review **count** with 10 / 50 / 90 % quantiles and draw the band after the history",
           "Forecast the **mean rating** the same way",
           "Split a forecast by a category (`category_col`) — verified vs unverified reviews",
           "Read the response's calibration flag and warnings, and what a forecast costs"],
          "no training; each forecast call costs 40 + 6 × horizon credits (6 months → 76 credits); a category split costs that × `max_categories`.",
          lane="Chronos-2 zero-shot over `review_time`"),
    code(PREAMBLE),
    code('''p = get_or_create_project(ls, "explore", kind="data_analysis")
before = credits_used(ls)
review = p.table("review").to_pandas(); review["review_time"] = pd.to_datetime(review["review_time"])
monthly = review.set_index("review_time").resample("ME")["rating"].agg(["count", "mean"])
monthly.tail(6)'''),
    md("The sample runs 2008 → September 2018 and thins out in its last months (80 → 36 → 4 → 3 → 1 → 1 reviews). A forecaster should follow that tail — watch the median hug zero while the upper quantile keeps the earlier level in play."),
    code('''fc = ls.predict.forecast(project_id=p.id, target="rating", agg="count", window="month", length=6, quantiles=[0.1, 0.5, 0.9])
print("target:", fc["target"], "(count) · window:", fc["window_type"], "× len", fc["window_length"], "· calibrated:", fc.get("calibrated"))
for pt in fc["forecast"]:
    q = pt["quantiles"]
    print(f"  {str(pt['timestamp'])[:10]}  median {pt['median']:.1f}  band [{float(q['0.1']):.1f}, {float(q['0.9']):.1f}]")
print("warnings:", [w["code"] for w in fc.get("warnings", [])])
fig(viz.forecast(fc, history=monthly["count"].tail(36), title="Reviews per month — history and 6-month forecast"))'''),
    code('''fr = ls.predict.forecast(project_id=p.id, target="rating", agg="mean", window="month", length=6, quantiles=[0.1, 0.5, 0.9])
print("mean rating per month, next 6:", [round(pt["median"], 2) for pt in fr["forecast"]])
fig(viz.forecast(fr, history=monthly["mean"].tail(36), title="Mean rating per month — history and forecast"))'''),
    md("`category_col` splits the series and forecasts each category — here verified vs unverified reviews. Each entry in `series` is a full forecast of its own. **Billing:** a split call costs the single-series fee × `max_categories` (default 8, up to 12), whatever the actual number of values — pass `max_categories` equal to the number of values you expect to pay for exactly those."),
    code('''fs = ls.predict.forecast(project_id=p.id, target="rating", agg="count", window="month", length=6, quantiles=[0.1, 0.5, 0.9], category_col="verified", max_categories=2)
print(fs.get("n_categories"), "categories:", [(s.get("label") or s.get("category")) for s in fs.get("series", [])])
import matplotlib.pyplot as plt
f, axes = plt.subplots(1, max(1, len(fs.get("series", []))), figsize=(10, 3.2), sharey=True)
for ax, s in zip(axes if hasattr(axes, "__len__") else [axes], fs.get("series", [])):
    pts = s.get("forecast") or []
    ax.plot([str(x["timestamp"])[:7] for x in pts], [x["median"] for x in pts], marker="o", color="#c0392b")
    ax.set_title(f"verified = {s.get('label') or s.get('category')}", loc="left", fontweight="bold"); ax.tick_params(axis="x", rotation=45)
fig(f)'''),
    code('''charged = credits_used(ls) - before
print("credits for the three calls:", charged, "· expected (40 + 6×6) × (1 + 1 + 2 categories) =", 76 * 4)
save_metrics(".", {"notebook": "07_forecast_review_volume", "task": "forecast · monthly review count + mean rating (zero-shot)", "model": "Chronos-2 (zero-shot)",
                   "project_id": p.id, "credits_charged_this_run": charged,
                   "headline": {"horizon_months": fc["window_length"], "calibrated": fc.get("calibrated"),
                                "next_month_reviews_median": round(fc["forecast"][0]["median"], 2), "next_month_mean_rating_median": round(fr["forecast"][0]["median"], 2),
                                "note": "zero-shot: no held-out metric by design"}})'''),
    next_steps("[10 · Credits & estimates](../10_credits_estimates_and_errors/) — every price before you pay it"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 08 predict API + monitoring
# ══════════════════════════════════════════════════════════════════════════════════════════
write("08_predict_api_and_monitoring", "08_predict_api_and_monitoring.ipynb", [
    intro("08 · The prediction API and model monitoring",
          "The verified-purchase model (04) goes into an application: score a review the app just received (inductive — from its columns and its neighbours), score in batches, and watch the model's inputs drift.",
          ["Read the model's feature schema — what a request must carry",
           "Inductive prediction: a row the platform has never stored, with `related_entities`",
           "Batch prediction, and the `active:<project>` selector",
           "Monitoring: feature health as a table and a chart, thresholds, alert configuration (email / webhook)",
           "Recent-prediction monitoring for the model"],
          "no training; 50 credits per prediction (batch items included)."),
    code(PREAMBLE),
    code('''p = get_or_create_project(ls, "verified", kind="data_science")
model = next(m for m in project_models(p) if m.get("metrics") and m.get("is_active"))
print("model:", model["label"], "· AUROC", round(model["metrics"]["auroc"], 3))'''),
    code('''schema = ls.predict.model(model["model_id"])
print("task:", schema.get("task_type"), "· entity:", schema.get("entity_table"), schema.get("entity_col"), "· target:", schema.get("target_col"))
print("required features:", [f.get("name") if isinstance(f, dict) else f for f in (schema.get("required_features") or [])])
print("optional features:", [f.get("name") if isinstance(f, dict) else f for f in (schema.get("optional_features") or [])])
print("active selector  :", schema.get("active_selector"))'''),
    md("## Inductive: a review the platform has never stored\n\nA relational model predicts from a neighbourhood, so a brand-new row needs its neighbours: pass the review's own columns plus `related_entities` — the ids of the product and customer it links to (they must exist). Without them the API refuses with a message that says exactly that."),
    code('''review = p.table("review").to_pandas()
sample = review.iloc[0]
features = {"review_text": "Arrived on time, exactly as described. The kids loved it and we read it twice the first night.", "summary": "Great buy", "rating": 5}
out = ls.predict.predict(model_id=model["model_id"], mode="inductive", features=features,
                         related_entities={"product": int(sample["product_id"]), "customer": int(sample["customer_id"])})
print("verified?", out["result"]["prediction"], "· probability", round(out["result"].get("probability") or 0, 3), "· mode", out["mode"])
if out.get("warnings"): print("warnings:", out["warnings"])'''),
    code('''from langsat import errors
try:
    ls.predict.predict(model_id=model["model_id"], mode="inductive", features=features)     # no neighbours → refused, clearly
except errors.LangsatError as e:
    print(type(e).__name__, "·", str(e.message)[:220])'''),
    md("Batch scoring with the **active** selector — `model_id=\"active:<project_id>\"` always resolves to the version the project serves, so a deploy of v2 needs no client change."),
    code('''top = ls.predict.top(model["model_id"], n=5)
ids = [r["entity_id"] for r in top["ranking"]]
batch = ls.predict.predict_batch(model_id=f"active:{p.id}", entity_ids=ids)
print(batch["succeeded"], "scored via active selector;", [round(pr.get("probability") or 0, 3) for pr in batch["predictions"]])'''),
    md("## Monitoring\n\nEvery prediction is logged with its inputs. `monitoring.summary()` compares recent inputs to the training baseline per feature (null rate, drift, new categories …); thresholds turn that into warning / critical, and the alert configuration sends an email or signs a webhook."),
    code('''summary = p.monitoring.summary()
feat = pd.DataFrame(summary.get("features") or [])
cols = [c for c in ("feature_name", "dtype", "health", "request_count", "missing_rate", "drift_score") if c in feat.columns]
print("state:", summary.get("state"), "· predictions in window:", summary.get("total_predictions"), "· alerts:", summary.get("total_alerts"))
feat[cols] if len(feat) else summary'''),
    code('''import matplotlib.pyplot as plt
counts = feat["health"].value_counts() if len(feat) else pd.Series(dtype=int)
f, ax = plt.subplots(figsize=(5, 2.6)); counts.plot(kind="barh", ax=ax, color="#c4a35a"); ax.set_title("Feature health (last 7 days)", loc="left", fontweight="bold"); fig(f)'''),
    code('''th = p.monitoring.thresholds()
print("effective thresholds:", {k: v for k, v in (th.get("effective") or {}).items() if k in ("null_rate_warning", "drift_warning", "new_category_warning")})
print("alerts:", {k: v for k, v in p.monitoring.alerts_config().items() if k in ("email_alerts_enabled", "webhook_enabled", "min_severity", "min_alert_interval_hours")})
recent = ls.predict.monitoring(model["model_id"], days=7)
print("recent:", {k: recent.get(k) for k in ("state", "total_predictions", "features_warning", "features_critical", "anomaly_score")})'''),
    code('''save_metrics(".", {"notebook": "08_predict_api_and_monitoring", "task": "serve + monitor the binary model from 04", "model": model.get("model_type"),
                   "project_id": p.id, "model_id": model["model_id"],
                   "headline": {"inductive_probability": round(out["result"].get("probability") or 0, 4), "batch_scored": batch["succeeded"],
                                "monitored_features": int(len(feat)), "predictions_in_window": summary.get("total_predictions")}})'''),
    next_steps("[09 · Webhooks](../09_webhooks/) — be told when a job finishes (and wire the alert webhook the same way)",
               "The SDK docs page for the API contract: https://langsat.ai/sdk/langsat-v1.json"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 09 webhooks
# ══════════════════════════════════════════════════════════════════════════════════════════
write("09_webhooks", "09_webhooks.ipynb", [
    intro("09 · Webhooks — be told when a job finishes",
          "A nightly pipeline re-uploads data and cleans; instead of polling, the orchestrator is told when the clean finishes (`job.succeeded`) and kicks off the next step. The delivery is signed, retried, and logged.",
          ["Register an https endpoint (a throw-away https://webhook.site URL here) for `clean` jobs",
           "Send a signed test ping and verify the signature the way a receiver would",
           "Run a clean and watch `job.succeeded` arrive exactly once",
           "Read the delivery log; rotate the secret; delete the webhook",
           "Receiver code you can paste into FastAPI or Express"],
          "no credits; the clean is free. Needs the `webhooks:read` / `webhooks:write` scopes on the key."),
    md("""```mermaid
sequenceDiagram
    participant You
    participant Langsat
    participant Endpoint
    You->>Langsat: POST /webhooks {url, events, job_kinds}
    Langsat-->>You: {id, secret}  (secret shown once)
    You->>Langsat: POST /projects/{id}/clean
    Langsat->>Endpoint: POST job.succeeded  X-Langsat-Signature: t=…,v1=HMAC
    Endpoint-->>Langsat: 2xx within 10 s
    Note over Langsat,Endpoint: on failure: retry +1m, +5m, +30m, +2h · then exhausted
```

Set `WEBHOOK_URL` (and `WEBHOOK_SITE_TOKEN` — the uuid in the URL — so the notebook can read what arrived) before running; without them the live cells explain and skip."""),
    code(PREAMBLE),
    code('''import os, time, urllib.request
from langsat.webhooks import verify_signature
from langsat import errors

URL = os.environ.get("WEBHOOK_URL")
TOKEN = os.environ.get("WEBHOOK_SITE_TOKEN")
if not URL:
    print("WEBHOOK_URL not set — open https://webhook.site, copy 'Your unique URL', and set WEBHOOK_URL / WEBHOOK_SITE_TOKEN")
p = get_or_create_project(ls, "explore", kind="data_analysis")
try:
    print("webhooks enabled:", ls.webhooks.enabled()["enabled"], "· key scopes:", [s for s in ls.me()["api_key"]["scopes"] if s.startswith("webhooks")])
    ls.webhooks.list()
except errors.MissingScope as e:
    print(f"this key cannot manage webhooks ({e.scope} missing) — mint one with the Full SDK preset; skipping the live steps")
    URL = None'''),
    md("## Register and test\n\nThe secret is returned **once**. `test()` sends a `ping` immediately; reading it back from webhook.site and verifying with `verify_signature` is exactly what your receiver will do."),
    code('''def received():
    with urllib.request.urlopen(f"https://webhook.site/token/{TOKEN}/requests?sorting=newest", timeout=20) as r:
        return json.load(r)["data"]

if URL:
    for w in ls.webhooks.list():
        if (w.get("description") or "") == "amazon-reviews example":
            ls.webhooks.delete(w["id"])
    w = ls.webhooks.create(URL, job_kinds=["clean"], description="amazon-reviews example")
    secret = w["secret"]                       # shown once
    print("webhook", w["id"], "events", w["events"], "kinds", w["job_kinds"])
    t = ls.webhooks.test(w["id"])
    print("ping delivered:", t["ok"], "· HTTP", t["delivery"]["last_status_code"])
    if TOKEN:
        time.sleep(2)
        hit = next(x for x in received() if x["headers"].get("x-langsat-event") == ["ping"])
        ev = verify_signature(hit["content"], hit["headers"]["x-langsat-signature"][0], secret=secret)
        print("signature verified · event", ev["id"], ev["type"])
        print("headers the receiver saw:", {k: v[0] for k, v in hit["headers"].items() if k.startswith("x-langsat")})'''),
    md("## A real job\n\nA clean on the explore project. The event names the job and where to fetch its result (`result_route`) — never the data itself."),
    code('''if URL:
    p.cleaning.clean().wait(timeout=1800)
    print("clean finished; waiting for the webhook …")
    got = None
    if TOKEN:
        for _ in range(24):
            time.sleep(5)
            got = next((x for x in received() if x["headers"].get("x-langsat-event") == ["job.succeeded"]), None)
            if got: break
        if got:
            ev = verify_signature(got["content"], got["headers"]["x-langsat-signature"][0], secret=secret)
            job = ev["data"]["job"]
            print("job.succeeded ·", job["kind"], job["status"], "· project", ev["data"]["project_id"] == p.id, "· fetch the result at", job["result_route"])
            mine = {d["delivery_id"] for d in ls.webhooks.deliveries(w["id"])}       # this webhook's deliveries only
            n = sum(1 for x in received() if x["headers"].get("x-langsat-event") == ["job.succeeded"]
                    and x["headers"].get("x-langsat-delivery", [""])[0] in mine)
            print("job.succeeded deliveries from this webhook that reached the endpoint:", n, "(exactly once)")
    print(pd.DataFrame(ls.webhooks.deliveries(w["id"]))[["event", "status", "attempts", "last_status_code", "created_at"]])'''),
    md("## Receiver code\n\nUse the **raw** request bytes — a re-serialised body will not match the signature."),
    md("""```python
# FastAPI
from fastapi import FastAPI, Request, HTTPException
from langsat.webhooks import verify_signature, InvalidSignature
app = FastAPI()

@app.post("/langsat")
async def hook(request: Request):
    try:
        event = verify_signature(await request.body(), request.headers.get("X-Langsat-Signature"), secret=SECRET)
    except InvalidSignature:
        raise HTTPException(400)
    if event["type"] == "job.succeeded" and event["data"]["job"]["kind"] == "clean":
        start_next_step(event["data"]["project_id"])
    return {"ok": True}
```

```ts
// Express
import express from "express";
import { verifySignature } from "@langsat/sdk";
const app = express();
app.post("/langsat", express.raw({ type: "*/*" }), async (req, res) => {
  try {
    const event = await verifySignature(req.body, req.header("x-langsat-signature"), { secret: process.env.LANGSAT_WEBHOOK_SECRET! });
    if (event.type === "job.succeeded") await startNextStep(event.data.project_id);
    res.sendStatus(200);
  } catch { res.sendStatus(400); }
});
```"""),
    code('''if URL:
    rotated = ls.webhooks.rotate_secret(w["id"])
    print("rotated:", rotated["secret"] != secret)
    ls.webhooks.delete(w["id"]); print("deleted")
save_metrics(".", {"notebook": "09_webhooks", "task": "webhooks · job.succeeded on clean", "model": "—", "project_id": p.id,
                   "headline": {"ran_live": bool(URL), "ping_ok": (t["ok"] if URL else None), "job_succeeded_seen": (bool(got) if URL and TOKEN else None)}})'''),
    next_steps("Docs: https://langsat.ai/resources/learn/getting-started/sdk#webhooks",
               "[10 · Credits & errors](../10_credits_estimates_and_errors/) — the error codes a receiver-side script can branch on"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 10 credits / estimates / errors
# ══════════════════════════════════════════════════════════════════════════════════════════
write("10_credits_estimates_and_errors", "10_credits_estimates_and_errors.ipynb", [
    intro("10 · Credits, estimates, quotas and typed errors",
          "Before wiring Langsat into automation you want to know what things cost, how much is left, and that a script can branch on *what* went wrong — without parsing English.",
          ["Balance, usage over time (as a chart) and quota",
           "`estimates()` across every project of this series — one table",
           "The training estimate with its knobs (`max_mode`, `fan_out_workers`)",
           "The error catalogue in action: `NotFound`, `Invalid`, `NeedsUserSession` — and what the others mean",
           "Least privilege: what the key may do"],
          "nothing — every call here is a read."),
    code(PREAMBLE),
    code('''d = ls.credits.dashboard()
pool = d.get("org_pool") or {}
print("plan:", d.get("tier"), "· personal balance:", d.get("balance_credits"), "· team pool:", pool.get("balance_credits"), "· used this month:", pool.get("credits_used_this_month"))
q = ls.credits.quota_status()
print("AI quota today:", q["usage"], "· compute cap today (USD):", q["compute"].get("cap_today_usd"), "· used:", q["compute"].get("used_today_usd"))'''),
    md("The ledger. `dashboard()` carries the latest transactions of the account that pays (the team pool here); `usage()` is the public-API lens (predict / forecast calls by day and by model). Below: what this series of notebooks spent, grouped by what it was for."),
    code('''tx = pd.DataFrame(d["recent_transactions"])
tx["created_at"] = pd.to_datetime(tx["created_at"]); tx["kind"] = tx["description"].str.replace(r" \\(.*\\)| ·.*", "", regex=True).str[:32]
display(tx[["created_at", "tx_type", "amount", "description"]].head(8))
spent = (-tx[tx["tx_type"] == "debit"].groupby("kind")["amount"].sum()).sort_values()
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(7, 2.8)); spent.plot(kind="barh", ax=ax, color="#c4a35a"); ax.set_xlabel("credits (last 20 transactions)")
ax.set_title("What the recent calls cost", loc="left", fontweight="bold"); fig(f)
api = ls.credits.usage(days=30)
print("public-API lens, 30 days:", {k: api.get(k) for k in ("total_calls", "total_credits", "training_jobs")})'''),
    md("## What would it cost?\n\n`estimates()` is one read per project: clean, refresh, train (instance and floor), ask, predict, serving. Nothing is reserved or charged."),
    code('''names = ["explore", "rating-regression", "rating-classes", "verified", "product-clusters", "review-anomalies"]
rows = []
for slug in names:
    pr = next((x for x in ls.projects.list() if x.name == f"amazon-reviews-{slug}"), None)
    if pr is None: continue
    e = pr.estimates()
    rows.append({"project": slug, "type": pr.data.get("project_type"), "clean_cr": e["clean"]["credits"], "clean_lane": e["clean"]["lane"],
                 "train_instance": (e.get("train") or {}).get("instance_type"), "train_min_floor": (e.get("train") or {}).get("suggested_minutes"),
                 "predict_cr": e["predict"]["credits_per_call"], "serving_cr_per_h": (e.get("serving") or {}).get("spot", {}).get("credits_per_hour")})
pd.DataFrame(rows)'''),
    md("The training estimate accepts the run's knobs. `max_mode` (the architecture bake-off) and `fan_out_workers` (embedding helpers) change the reservation — compare the three rows."),
    code('''p = next(x for x in ls.projects.list() if x.name == "amazon-reviews-rating-regression")
est = []
for kw in ({}, {"fan_out_workers": 2}, {"max_mode": True}):
    r = ls.credits.estimate_training(estimated_minutes=71, max_training_min=71, project_id=p.id, **kw)
    est.append({"options": kw or "default", "instance": r["instance_type"], "cr_per_min": r["cr_per_min"], "reserved_credits": r["max_cost"],
                "fan_out_helpers": r["fan_out_helpers"], "benchmark_workers": r["benchmark_workers"]})
pd.DataFrame(est)'''),
    md("## Errors you can branch on\n\nEvery 4xx carries `X-Error-Code`; the SDK raises a typed exception with `.status`, `.code`, `.message`, `.detail`."),
    code('''from langsat import errors
explore = next(x for x in ls.projects.list() if x.name == "amazon-reviews-explore")
try:
    ls.projects.get("00000000-0000-0000-0000-000000000000")
except errors.NotFound as e:
    print("NotFound         ·", e.status, e.code, "·", e.message)
try:
    ls.predict.forecast(project_id=explore.id, target="rating", window="fortnight", length=3)
except errors.Invalid as e:                               # 422: the request shape — detail names the field
    bad = e.detail[0] if isinstance(e.detail, list) and e.detail else e.detail
    print("Invalid          ·", e.status, "·", (bad.get("loc"), bad.get("msg")) if isinstance(bad, dict) else bad)
try:
    ls.request("GET", "/settings")                        # billing, settings, key management: a signed-in person only
except errors.NeedsUserSession as e:
    print("NeedsUserSession ·", e.status, e.code, "·", str(e.message)[:90])
print("MissingScope     · 403 missing_scope — a key without the scope a route needs (e.scope says which); this key carries all 20, so nothing here can trigger it")
print("ProjectBusy      · 409 project_busy — a second clean / train while one runs; Retry-After tells a script how long to wait (`wait_if_busy=` on the SDK call does it for you)")
print("InsufficientCredits · 402 · QuotaExceeded · 429 quota_exhausted · RateLimited · 429 rate_limited (retried automatically) · RowCapExceeded · 422 row_cap_exceeded")'''),
    code('''me = ls.me(); key = me["api_key"]
print("key:", key["name"], "·", len(key["scopes"]), "scopes:", key["scopes"])
print("projects allow-list:", key["project_ids"] or "all", "· expires:", key["expires_at"])
print("\\nA key minted with only projects:read + data:read would get MissingScope (403, scope='projects:write') on projects.create(...) —")
print("mint keys with the smallest set a script needs: Settings → API keys → Permissions.")'''),
    code('''save_metrics(".", {"notebook": "10_credits_estimates_and_errors", "task": "credits, estimates, typed errors", "model": "—",
                   "headline": {"plan": d.get("tier"), "scopes_on_key": len(key["scopes"]), "projects_estimated": len(rows),
                                "errors_demonstrated": ["NotFound", "Invalid", "NeedsUserSession"]}})'''),
    next_steps("[11 · Fine-tune](../11_finetune/) and [12 · Training options](../12_training_options/) — the two notebooks that train again"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 11 fine-tune
# ══════════════════════════════════════════════════════════════════════════════════════════
write("11_finetune", "11_finetune.ipynb", [
    intro("11 · Fine-tune a model on new data",
          "The rating model has been in production for a while; a thousand new reviews have arrived. Retraining from scratch costs a full run and forgets nothing — but fine-tuning continues the existing version on the new rows, keeps the lineage, and takes minutes.",
          ["Train a first version on the reviews up to a cut-off date (9,000 rows)",
           "Add the 1,000 newer reviews with `sources.upload_finetune` — additive, the model stays",
           "`finetune_verify_schema` (structure must match) and `finetune_compute_delta` (how many rows are new, what the API recommends)",
           "`train(mode=\"fine_tune\", from_model_id=…, training_mode=\"warm\")` and the lineage v1 → v2",
           "Compare parent and child metrics; the child's Model tab"],
          "two GPU runs — the first is a normal training, the fine-tune reserves the same floor but runs faster (only the new rows are embedded).",
          lane="relational GNN (GraphSAGE), text embeddings; warm fine-tune continues from the parent checkpoint"),
    code(PREAMBLE),
    md("## 1 · A model on the first 9,000 reviews\n\nChronological split: the oldest 9,000 reviews are today's data, the newest 1,000 are 'next month'. The first slice is written as `review.csv` (the table name comes from the file name). The second batch is what a real export looks like: the new reviews **plus the customers and products they reference** — a fine-tune upload carries every table the model uses (rows that already exist are recognised by the delta and not counted as new) — written as **parquet** with real dtypes (`review_time` as a timestamp), which is what the schema check and the delta read."),
    code('''import tempfile, pathlib
from _common import DATA_DIR
review = pd.read_csv(DATA_DIR / "review.csv").sort_values("review_time")
customer = pd.read_csv(DATA_DIR / "customer.csv"); product = pd.read_csv(DATA_DIR / "product.csv")
tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "part1").mkdir(); (tmp / "part2").mkdir()
review.iloc[:9000].to_csv(tmp / "part1" / "review.csv", index=False)
new = review.iloc[9000:].assign(review_time=lambda d: pd.to_datetime(d["review_time"]))
new.to_parquet(tmp / "part2" / "review.parquet", index=False)
customer[customer.customer_id.isin(new.customer_id)].to_parquet(tmp / "part2" / "customer.parquet", index=False)
product[product.product_id.isin(new.product_id)].to_parquet(tmp / "part2" / "product.parquet", index=False)
print("part 1:", 9000, "reviews up to", review.iloc[8999]["review_time"])
print("part 2:", len(new), "reviews after that, by", new.customer_id.nunique(), "customers on", new.product_id.nunique(), "products")
p = get_or_create_project(ls, "finetune", kind="data_science", files=[DATA_DIR / "customer.csv", DATA_DIR / "product.csv", tmp / "part1" / "review.csv"])
before = credits_used(ls)'''),
    code('''train_or_reuse(p, "Predict the rating a review gives, from the review text, its summary and the product it is about",
               task_type="supervised", subtask_type="regression", enable_text_embedding=True)
parent = min((m for m in p.models.list() if m.get("metrics")), key=lambda m: m.get("version", 0))   # v1 — the model that was in production
print("parent:", parent["version_label"], parent["model_type"], "· MAE", round(parent["metrics"]["mae"], 4))'''),
    md("## 2 · New data arrives\n\nOn a trained project the normal upload is refused (it would discard the model) — `upload_finetune` appends. Files are routed to a table by their exact column set, so the file names do not matter."),
    code('''versions = project_models(p)
child = next((m for m in versions if m.get("parent_model_id") == parent["model_id"] and m.get("metrics")), None)
if child:
    print("fine-tuned child already exists:", child["label"], "— reusing")
else:
    for f in p.models.finetune_files()["files"]:                    # a re-run starts from a clean staging area
        if not f.get("is_used"):
            ls.request("DELETE", f"/projects/{p.id}/finetune/files/{f['name']}")
    up = p.sources.upload_finetune(tmp / "part2" / "review.parquet", tmp / "part2" / "customer.parquet", tmp / "part2" / "product.parquet")
    print("appended:", [(f.get("table"), f.get("rows")) for f in up.get("files", [])])
    print("fine-tune files on the project:", [(f["name"], f["rows"], f.get("is_used")) for f in p.models.finetune_files()["files"]])'''),
    md("## 3 · Verify the schema, compute the delta\n\n`verify_schema` is a strict structural check (a missing or retyped column fails with the diff). `compute_delta` anti-joins the new rows against the parent's manifest and recommends a mode: `fine_tune` when at most 10 % of the rows are new, `fresh_recommended` above that, `no_op` when nothing is new. Here 1,000 of 10,000 rows are new — 11 %, just over the line — so the API suggests a fresh train; the notebook fine-tunes anyway to show the flow. The recommendation is advice, the choice is yours."),
    code('''if not child:
    ok = p.models.finetune_verify_schema()
    print("schema ok:", ok.get("ok"), ok.get("errors") or "")
    delta = p.models.finetune_compute_delta(from_model_id=parent["model_id"])
    print({k: delta.get(k) for k in ("new_rows", "existing_rows", "recommended_mode", "boundary_crossed", "will_fallback")})'''),
    md("## 4 · Fine-tune (warm)\n\n`training_mode=\"warm\"` continues from the parent checkpoint (the encoder stays frozen unless `freeze_encoder=False`); `\"cold\"` would retrain fresh weights on the cumulative data as a sibling version (`v2.1`). The child is minted as `v2` with `parent_model_id` set, and activated on completion."),
    code('''if not child:
    from _common import wait_for_training, known_models
    known = known_models(p)
    minutes = int(p.estimates()["train"]["suggested_minutes"]) + 2      # the floor: embedding time for the NEW rows + the training floor
    print("reserving", minutes, "min")
    p.train(mode="fine_tune", from_model_id=parent["model_id"], training_mode="warm", max_training_min=minutes)
    child = wait_for_training(p, known_model_ids=known)
    print("child:", child.get("version_label"), "· parent:", child.get("parent_model_id") == parent["model_id"])'''),
    md("## 5 · Parent vs child\n\nThe child was tested on the cumulative data (its test split now includes the newest reviews); the parent's numbers are from its own run on the older data. Read the direction, not the third decimal — each run took under a minute, and the point of a fine-tune is keeping a model current on the rows that just arrived without paying for a full retrain."),
    code('''parent = next(m for m in p.models.list() if m["model_id"] == parent["model_id"])
child = next(m for m in p.models.list() if m["model_id"] == child["model_id"])
lineage = pd.DataFrame([{"version": m.get("version_label"), "family": m.get("family"), "parent": (m.get("parent_model_id") or "")[:8], "active": m.get("is_active"),
                         "trained": str(m.get("created_at"))[:19], "seconds": m.get("training_duration_sec")} for m in sorted(p.models.list(), key=lambda m: m.get("version", 0))])
display(lineage)
metrics_table(parent, child)'''),
    code('''fig(viz.model_dashboard(p, cols=2))'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "11_finetune", "task": "fine-tune · regression on +1,000 newer reviews", "model": child.get("model_type"),
                   "project_id": p.id, "model_id": child["model_id"], "parent_model_id": parent["model_id"], "training_duration_sec": child.get("training_duration_sec"),
                   "credits_charged_this_run": charged,
                   "headline": {"parent_mae": round(parent["metrics"]["mae"], 4), "child_mae": round(child["metrics"]["mae"], 4), "child_version": child.get("version_label")},
                   "baseline": {"parent_r2": round(parent["metrics"].get("r2", 0), 4), "child_r2": round(child["metrics"].get("r2", 0), 4)}})'''),
    next_steps("Cold fine-tune (`training_mode=\"cold\"`) when the data distribution moved; `freeze_encoder=False` when you have many new rows",
               "[12 · Training options](../12_training_options/) — architectures and embeddings on the same task"),
])

# ══════════════════════════════════════════════════════════════════════════════════════════
# 12 training options
# ══════════════════════════════════════════════════════════════════════════════════════════
write("12_training_options", "12_training_options.ipynb", [
    intro("12 · Training options — text embeddings, architectures, the bake-off, parallel embedding",
          "Same task as 02 (predict the rating), three ways to train it. Which options matter on this data, what they cost, and how to keep the best version serving.",
          ["v1 (from 02): the default — GraphSAGE with text embeddings",
           "v2: `enable_text_embedding=False` — how much the review text is worth",
           "v3: `max_mode=True` — the architecture bake-off (every architecture is trained, the best one ships)",
           "`fan_out_workers`: what it parallelises and why it did not fire at this size",
           "Compare the versions in one table, re-activate the best"],
          "two more GPU runs on the regression project (the bake-off may launch several boxes; credits are estimated first and unused minutes refunded).",
          lane="GraphSAGE / GATv2 / GIN — the platform picks in max mode"),
    code(PREAMBLE),
    code('''p = get_or_create_project(ls, "rating-regression", kind="data_science")
before = credits_used(ls)
versions = project_models(p)
v1 = next(m for m in sorted(versions, key=lambda m: m.get("version", 0)) if m.get("metrics"))
print("v1:", v1["label"], "· MAE", round(v1["metrics"]["mae"], 4))
print("text columns the task embeds:", json.loads(p.data["task_config"]).get("text_embed_columns") if isinstance(p.data.get("task_config"), str) else (p.data.get("task_config") or {}).get("text_embed_columns"))'''),
    md("## v2 · without text embeddings\n\n`enable_text_embedding=False` keeps the text columns but encodes them categorically — the model sees the graph and the numbers, not the words. The estimate drops the embedding minutes; the metrics tell you what the words were worth."),
    code('''from _common import wait_for_training, known_models
by_label = {m.get("display_name") or "": m for m in versions}
v2 = next((m for m in versions if m.get("metrics") and m.get("version") == 2), None)
if v2 is None:
    minutes = int(p.estimates()["train"]["suggested_minutes"]) + 2
    est = ls.credits.estimate_training(estimated_minutes=minutes, max_training_min=minutes, project_id=p.id)
    print("reservation for a normal run:", est["max_cost"], "credits for", minutes, "min")
    known = known_models(p)
    p.train(enable_text_embedding=False, max_training_min=minutes)
    v2 = wait_for_training(p, known_model_ids=known)
print("v2:", v2.get("version_label"), v2.get("model_type"), "· MAE", round(v2["metrics"]["mae"], 4), "·", v2.get("training_duration_sec"), "s")'''),
    md("## v3 · the bake-off (`max_mode=True`)\n\nMax mode trains every architecture the plan allows — GraphSAGE small and full, GATv2, GIN — and keeps the single best on the validation metric. One model row comes back; `model_type` says who won. On Max/Team plans the platform can put each architecture on its own GPU box (`benchmark_workers` in the estimate says how many it will launch); when that is 0 the four run one after another on one box, which is what happened here."),
    code('''v3 = next((m for m in project_models(p) if m.get("metrics") and m.get("version") == 3), None)
if v3 is None:
    minutes = int(p.estimates()["train"]["suggested_minutes"]) + 2
    est = ls.credits.estimate_training(estimated_minutes=minutes, max_training_min=minutes, project_id=p.id, max_mode=True)
    print("reservation in max mode:", est["max_cost"], "credits · benchmark workers:", est["benchmark_workers"])
    known = known_models(p)
    p.train(max_mode=True, enable_text_embedding=True, max_training_min=minutes)
    v3 = wait_for_training(p, known_model_ids=known)
print("v3:", v3.get("version_label"), "· winner:", v3.get("model_type"), "· MAE", round(v3["metrics"]["mae"], 4), "·", v3.get("training_duration_sec"), "s")'''),
    md("## `fan_out_workers` — parallel text embedding\n\nOn Max/Team plans, `fan_out_workers=2` or `3` splits the *text embedding* step across helper boxes. It only fires when the embedding estimate is at least 5 minutes — 10,000 reviews × 4 text columns embed in about 3 minutes on one box, so here the platform runs single-box and says so. The estimate still shows the helper cost, which is how you decide."),
    code('''rows = []
floor = int(p.estimates()["train"]["suggested_minutes"])
for kw in ({}, {"fan_out_workers": 2}, {"fan_out_workers": 3}, {"max_mode": True}):
    r = ls.credits.estimate_training(estimated_minutes=floor, max_training_min=floor, project_id=p.id, **kw)
    rows.append({"options": kw or "default", "reserved_credits": r["max_cost"], "fan_out_helpers": r["fan_out_helpers"], "benchmark_workers": r["benchmark_workers"]})
print("train floor for this project:", p.estimates()["train"])
pd.DataFrame(rows)'''),
    md("## Compare and choose\n\nThree versions, one table. Activate the best by MAE — `predict` with `active:<project>` then serves it without any client change."),
    code('''versions = sorted([m for m in p.models.list() if m.get("metrics")], key=lambda m: m.get("version", 0))
display(pd.DataFrame([{"version": m.get("version_label"), "architecture": m.get("model_type"), "seconds": m.get("training_duration_sec"), "active": m.get("is_active")} for m in versions]))
display(metrics_table(*versions))
best = min(versions, key=lambda m: m["metrics"]["mae"])
if not best.get("is_active"):
    p.models.activate(best["model_id"])
print("serving:", best.get("version_label"), best.get("model_type"), "· MAE", round(best["metrics"]["mae"], 4))
fig(viz.model_dashboard(p, only=["Overview", "Importance", "MAE"], cols=3))'''),
    md("`Tasks.set_text_embed_columns` chooses *which* columns get embedded — e.g. only the review text, not the product description — before a (re)train. It is a task setting, so it applies to the next run."),
    code('''print("current:", json.loads(p.refresh().data["task_config"]).get("text_embed_columns") if isinstance(p.data.get("task_config"), str) else (p.data.get("task_config") or {}).get("text_embed_columns"))
print("example (not applied here): p.tasks.set_text_embed_columns({'review': ['review_text', 'summary']})")'''),
    code('''charged = credits_used(ls) - before
save_metrics(".", {"notebook": "12_training_options", "task": "regression · text on/off · max-mode bake-off", "model": best.get("model_type"),
                   "project_id": p.id, "model_id": best["model_id"], "credits_charged_this_run": charged,
                   "headline": {f"mae_{m.get('version_label')}_{m.get('model_type')}": round(m["metrics"]["mae"], 4) for m in versions} | {"serving": best.get("version_label")}})'''),
    next_steps("[11 · Fine-tune](../11_finetune/) — keep the winner current as data arrives",
               "[08 · Predict API](../08_predict_api_and_monitoring/) — `active:<project>` follows whatever you activate"),
])
