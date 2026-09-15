# Langsat examples

Everything the [Langsat](https://langsat.ai) app does — upload relational data, detect how the tables
join, clean, chart, ask questions in plain English, train a model, predict, forecast, monitor — through
the same API, from code. **The same user, the same rights, the same prices, with no screen.**

This repo is the worked example: one small relational dataset (Amazon book reviews — 10,000 reviews
by 9,821 customers of 8,607 books, three tables with real foreign keys) taken through every kind of task the
platform runs, as executed Jupyter notebooks.
Every number in the table below was produced by running the notebook you see — outputs are committed.

## Install and sign in

```bash
pip install langsat            # Python 3.9+   ·   npm install @langsat/sdk  for JavaScript / TypeScript
langsat login                  # paste an API key from the app: Settings → API keys → Generate new key
langsat whoami
```

A key acts **as you** — your plan's limits, your credits, your team role. Give it only the scopes a
script needs (presets: *Predict only*, *Read only*, *Full SDK*). Docs: <https://langsat.ai/resources/learn/getting-started/sdk>.

## The use cases

Dataset: [`data/amazon_reviews_10k/`](data/amazon_reviews_10k/README.md) — `customer`, `product`, `review`
(review → customer, review → product, time column `review_time`).

<!-- results:start -->
| notebook | task | model / lane | result (test split) | baseline | training time | credits charged |
|---|---|---|---|---|---|---|
| [01_setup_and_explore](examples/amazon_reviews/01_setup_and_explore/01_setup_and_explore.ipynb) | setup + explore (no model) | — | published_link yes, answer_chars 104 | — | — | — |
| [02_rating_regression](examples/amazon_reviews/02_rating_regression/02_rating_regression.ipynb) | regression · review.rating | GraphSAGE (default, labelled *Baseline*) | mae 0.497, rmse 0.689, r2 0.438, spearman 0.532 | mae 0.770, rmse 0.980, r2 0.000 | 1 min 25 s | 100 |
| [03_rating_classification](examples/amazon_reviews/03_rating_classification/03_rating_classification.ipynb) | multiclass · review.rating (5 classes) | GATv2 | accuracy 0.665, f1_macro 0.466 | accuracy 0.631, f1_macro_note always-5★ | 2 min 41 s | 100 |
| [04_verified_purchase_binary](examples/amazon_reviews/04_verified_purchase_binary/04_verified_purchase_binary.ipynb) | binary · review.verified | GraphSAGE (default, labelled *Baseline*) | auroc 0.628, pr_auc 0.783, f1 0.778, accuracy 0.667 | auroc 0.500, pr_auc 0.692 | 1 min 9 s | 350 |
| [05_product_clustering](examples/amazon_reviews/05_product_clustering/05_product_clustering.ipynb) | clustering · product (unsupervised) | GraphMAE (self-supervised) | val_loss 0.776, epochs 100, clusters_in_sample 2, note self-supervised: reconstruction loss, no labels | — | 1 min 1 s | 1,000 |
| [06_review_anomaly](examples/amazon_reviews/06_review_anomaly/06_review_anomaly.ipynb) | anomaly detection · review (unsupervised) | GraphMAE (self-supervised) | val_loss 0.621, epochs 100, flagged_in_sample 1, sample_size 20, note self-supervised: reconstruction loss, no labels | — | 1 min 6 s | 1,000 |
| [07_forecast_review_volume](examples/amazon_reviews/07_forecast_review_volume/07_forecast_review_volume.ipynb) | forecast · monthly review count + mean rating (zero-shot) | Chronos-2 (zero-shot) | horizon_months 6, calibrated yes, next_month_reviews_median 0.000, next_month_mean_rating_median 4.090, note zero-shot: no held-out metric by design | — | — | 152 |
| [08_predict_api_and_monitoring](examples/amazon_reviews/08_predict_api_and_monitoring/08_predict_api_and_monitoring.ipynb) | serve + monitor the binary model from 04 | GraphSAGE (default, labelled *Baseline*) | batch_scored 3, monitoring_keys state, model_id, project_name, task_type, period_days, version, family, version_label | — | — | — |
| [10_credits_estimates_and_errors](examples/amazon_reviews/10_credits_estimates_and_errors/10_credits_estimates_and_errors.ipynb) | credits, estimates, typed errors | — | tier team, scopes_on_key 17, errors_demonstrated NotFound, Invalid/LangsatError, Conflict/LangsatError | — | — | — |
<!-- results:end -->

Metrics are what the platform reports on its held-out test split for the trained model; baselines are
computed in the notebook from the same table (always-the-mean / majority class / class prior). *Credits* are what
the recorded run charged — a run that reuses a finished model pays only for its predictions (50 credits each);
the first run of a training notebook reserves ~65 min of GPU (≈8,700 credits on the Team plan) and refunds the
unused minutes, so a few-minute training costs a few hundred.

| # | notebook | what you learn |
|---|---|---|
| 01 | [Set up and explore](examples/amazon_reviews/01_setup_and_explore/) | upload → schema detection → clean (free) → pandas → dashboard + public link → ask |
| 02 | [Rating regression](examples/amazon_reviews/02_rating_regression/) | a task in one sentence → relational GNN + text embeddings → MAE / RMSE / R² → predict, rank, importance |
| 03 | [Rating classification](examples/amazon_reviews/03_rating_classification/) | the same target as 5 classes → accuracy / macro-F1 → probabilities, confusion matrix |
| 04 | [Verified purchase (binary)](examples/amazon_reviews/04_verified_purchase_binary/) | AUROC / PR-AUC / F1 → rank every review → batch scoring |
| 05 | [Product clustering](examples/amazon_reviews/05_product_clustering/) | unsupervised GraphMAE → silhouette / Davies-Bouldin → segment members |
| 06 | [Review anomaly detection](examples/amazon_reviews/06_review_anomaly/) | unsupervised → anomaly rate, threshold → the flagged reviews |
| 07 | [Forecast review volume](examples/amazon_reviews/07_forecast_review_volume/) | zero-shot Chronos-2 → quantile bands, no training |
| 08 | [Predict API and monitoring](examples/amazon_reviews/08_predict_api_and_monitoring/) | feature schema → inductive prediction → batch → drift monitoring, thresholds, alerts |
| 09 | [Webhooks](examples/amazon_reviews/09_webhooks/) | `job.succeeded` to your endpoint, signature verification, delivery log |
| 10 | [Credits, estimates, errors](examples/amazon_reviews/10_credits_estimates_and_errors/) | what things cost before you run them; typed errors a script can branch on |

Each folder has the notebook, a short README and `results/metrics.json` (what the run ended on).

## Run them yourself

```bash
git clone https://github.com/langsat-ai/langsat.git && cd langsat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
langsat login
jupyter lab examples/amazon_reviews/01_setup_and_explore/01_setup_and_explore.ipynb
```

or all of them headlessly, in order: `scripts/run_all.sh`. Notebooks are idempotent — projects are
found by name and reused, a finished model is not retrained. The training notebooks (02–06) each run a
GPU job of a few minutes; the estimate is printed before each one and credits are reserved, then
refunded for unused minutes. `07`–`10` do not train.

## What is not here (yet)

Link prediction / recommendation needs a direct foreign key between the two tables you want to link
(here customer → product goes through `review`, so the platform refuses it — that is the right answer);
temporal churn tasks need more repeat customers than this sample has (≤4 reviews per customer, 155 repeat
customers). Both work on the larger samples of the same dataset.

## License

MIT for the code in this repository. The dataset sample is redistributed under its
[own terms](data/amazon_reviews_10k/README.md#source-and-attribution).
