# 02 · Predict a review's rating (regression)

**Use case:** the support team wants a predicted star rating for every review as it arrives, from its text, its summary and the product it is about — a number, not a class.

**Model / lane:** relational GNN — the default GraphSAGE architecture (Langsat labels it *Baseline*), text embeddings on `review_text`, `summary`, `product.title` / `description`

**Sub-tasks**
1. Create a `data_science` project on the same three tables
2. Define the task in plain English — Langsat writes the task config (entity, target, text columns)
3. Train a relational GNN with text embeddings (the estimate is shown first)
4. Read the test metrics: MAE, RMSE, R², Spearman — against an always-the-mean baseline
5. Score one review, then look at feature importance

Notebook: [`02_rating_regression.ipynb`](02_rating_regression.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | regression · review.rating |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| mae | 0.4967 |
| rmse | 0.6894 |
| r2 | 0.4379 |
| spearman | 0.5317 |
| baseline mae | 0.7701 |
| baseline rmse | 0.9803 |
| baseline r2 | 0.0000 |
| training time | 1 min 25 s |
| credits charged (this run) | 100 |
| project | `8847cd90-8444-4934-aad9-8faa372216bf` |
| model | `bac91893-e681-4f20-9e90-70478ccd1874` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 02_rating_regression.ipynb
```
