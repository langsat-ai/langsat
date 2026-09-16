# 02 · Predict a review's rating (regression)

**Use case.** The support team wants a predicted star rating for every review as it arrives — from its text, its summary and the product it is about — as a number they can rank on, not a class.

**Model / lane:** relational GNN — the default GraphSAGE architecture (Langsat labels it *Baseline*), text embeddings on `review_text`, `summary`, `product.title` / `description`

**What you will learn**
1. Create a `data_science` project on the three tables (no clean step — the training pipeline cleans)
2. Define the task in plain English and read the config Langsat derived
3. Train a relational GNN with text embeddings, watch the run
4. Read the test metrics — MAE, RMSE, R², Spearman — against an always-the-mean baseline
5. See the Model tab as images: overview, feature importance, training curves, MAE/RMSE per epoch
6. Rank every review by predicted rating, score one on demand, look at residuals

**What this costs.** one GPU training run (a few minutes on `g4dn.2xlarge`; the API reserves the estimate — about 8,700 credits on the Team plan — and refunds the unused minutes, so a 90-second run ends up costing a few hundred), plus 50 credits per on-demand prediction. A re-run that finds the finished model pays only for its predictions.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`02_rating_regression.ipynb`](02_rating_regression.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | regression · review.rating |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| mae | 0.4760 |
| rmse | 0.6621 |
| r2 | 0.4816 |
| spearman | 0.5356 |
| baseline mae | 0.7701 |
| baseline rmse | 0.9803 |
| baseline r2 | 0.0000 |
| training time | 2 min 32 s |
| credits charged (this run) | 150 |
| project | `8847cd90-8444-4934-aad9-8faa372216bf` |
| model | `927511dc-161d-4a92-ac2b-6e3b97a0c9fc` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 02_rating_regression.ipynb
```
