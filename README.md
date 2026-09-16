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
| [01_setup_and_explore](examples/amazon_reviews/01_setup_and_explore/01_setup_and_explore.ipynb) | setup + explore (no model) | — | published_link yes, chat_chart yes, answer_chars 489 | — | — | — |
| [02_rating_regression](examples/amazon_reviews/02_rating_regression/02_rating_regression.ipynb) | regression · review.rating | GraphSAGE (default, labelled *Baseline*) | mae 0.476, rmse 0.662, r2 0.482, spearman 0.536 | mae 0.770, rmse 0.980, r2 0.000 | 2 min 32 s | 150 |
| [03_rating_classification](examples/amazon_reviews/03_rating_classification/03_rating_classification.ipynb) | multiclass · review.rating (5 classes) | GATv2 | accuracy 0.665, f1_macro 0.466, recall_1star 0.375 | accuracy 0.631, f1_macro_note always-5★ | 2 min 41 s | 100 |
| [04_verified_purchase_binary](examples/amazon_reviews/04_verified_purchase_binary/04_verified_purchase_binary.ipynb) | binary · review.verified | GraphSAGE (default, labelled *Baseline*) | auroc 0.628, pr_auc 0.783, f1 0.778, accuracy 0.667 | auroc 0.500, pr_auc 0.692 | 1 min 9 s | 1,100 |
| [05_product_clustering](examples/amazon_reviews/05_product_clustering/05_product_clustering.ipynb) | clustering · product (unsupervised) | GraphMAE (self-supervised) | val_loss 0.776, silhouette 0.4934, davies_bouldin 0.8753, clusters 2, note quality numbers come from the Model tab cards | — | 1 min 1 s | 2,000 |
| [06_review_anomaly](examples/amazon_reviews/06_review_anomaly/06_review_anomaly.ipynb) | anomaly detection · review (unsupervised) | GraphMAE (self-supervised) | val_loss 0.621, threshold 0.435, flagged_in_sample 2, sample_size 30 | — | 1 min 6 s | 1,500 |
| [07_forecast_review_volume](examples/amazon_reviews/07_forecast_review_volume/07_forecast_review_volume.ipynb) | forecast · monthly review count + mean rating (zero-shot) | Chronos-2 (zero-shot) | horizon_months 6, calibrated yes, next_month_reviews_median 0.000, next_month_mean_rating_median 4.090, note zero-shot: no held-out metric by design | — | — | 304 |
| [08_predict_api_and_monitoring](examples/amazon_reviews/08_predict_api_and_monitoring/08_predict_api_and_monitoring.ipynb) | serve + monitor the binary model from 04 | GraphSAGE (default, labelled *Baseline*) | inductive_probability 0.909, batch_scored 5, monitored_features 9, predictions_in_window 70 | — | — | — |
| [09_webhooks](examples/amazon_reviews/09_webhooks/09_webhooks.ipynb) | webhooks · job.succeeded on clean | — | ran_live yes, ping_ok yes, job_succeeded_seen yes | — | — | — |
| [10_credits_estimates_and_errors](examples/amazon_reviews/10_credits_estimates_and_errors/10_credits_estimates_and_errors.ipynb) | credits, estimates, typed errors | — | plan team, scopes_on_key 20, projects_estimated 6, errors_demonstrated NotFound, Invalid, NeedsUserSession | — | — | — |
| [11_finetune](examples/amazon_reviews/11_finetune/11_finetune.ipynb) | fine-tune · regression on +1,000 newer reviews | GraphSAGE (default, labelled *Baseline*) | parent_mae 0.495, child_mae 0.507, child_version v2 | parent_r2 0.444, child_r2 0.452 | 0 min 44 s | 0 |
| [12_training_options](examples/amazon_reviews/12_training_options/12_training_options.ipynb) | regression · text on/off · max-mode bake-off | GraphSAGE (default, labelled *Baseline*) | mae_v1_Baseline 0.497, mae_v2_Baseline 0.730, mae_v3_Baseline 0.476, serving v3 | — | — | 1,000 |
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
| 11 | [Fine-tune on new data](examples/amazon_reviews/11_finetune/) | train on 9,000 reviews → 1,000 newer ones arrive → `upload_finetune` → delta → warm fine-tune → v1 vs v2 |
| 12 | [Training options](examples/amazon_reviews/12_training_options/) | text embeddings on/off, the architecture bake-off (`max_mode`), `fan_out_workers`, `model_key` → pick and activate the best version |

Each folder has the notebook, a short README and `results/metrics.json` (what the run ended on).

**Charts are images.** Every chart in these notebooks — the Model tab the app builds after training (overview,
feature importance, training curves, confusion matrix, ROC, cluster quality, the anomaly threshold), dashboard
tabs, charts the chat wrote, forecasts — is drawn by `langsat.viz` from the same Plotly JSON the app renders,
as a PNG that GitHub shows inline. `pip install "langsat[viz]"`; see [`SDK docs/viz.md`](https://langsat.ai/resources/learn/getting-started/sdk#charts).

## Run them yourself

```bash
git clone https://github.com/langsat-ai/langsat.git && cd langsat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
langsat login
jupyter lab examples/amazon_reviews/01_setup_and_explore/01_setup_and_explore.ipynb
```

or all of them headlessly, in order: `scripts/run_all.sh`. The notebooks are generated from
`scripts/build_notebooks.py` (edit there, regenerate, re-run — regenerating clears the outputs) and the per-notebook
READMEs from `scripts/build_readmes.py`. Notebooks are idempotent — projects are
found by name and reused, a finished model is not retrained. The training notebooks (02–06, 11, 12) each run
GPU jobs of a few minutes; the estimate is printed before each one and credits are reserved, then
refunded for unused minutes. `07`–`10` do not train. `09` needs `WEBHOOK_URL` (a https://webhook.site URL) and
`WEBHOOK_SITE_TOKEN` in the environment to run live.

## What is not here (yet)

Link prediction / recommendation needs a direct foreign key between the two tables you want to link
(here customer → product goes through `review`, so the platform refuses it — that is the right answer);
temporal churn tasks need more repeat customers than this sample has (≤4 reviews per customer, 155 repeat
customers). Both work on the larger samples of the same dataset.

## License

MIT for the code in this repository. The dataset sample is redistributed under its
[own terms](data/amazon_reviews_10k/README.md#source-and-attribution).
