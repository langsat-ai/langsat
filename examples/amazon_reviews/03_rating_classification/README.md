# 03 · Predict the star rating as a class (1–5)

**Use case.** The same question as 02, but the product team wants class probabilities — *how sure* is the model this is a 1★ review? — to route angry customers to a person before the review goes live.

**Model / lane:** relational GNN — GATv2 (`model_key="gatv2_full"`), 5 classes, text embeddings

**What you will learn**
1. Define the task as **multiclass** classification and pick an architecture (`model_key="gatv2_full"`)
2. Accuracy and macro-F1 on the test split vs the always-5★ baseline (63% of reviews are 5★)
3. The Model tab: confusion matrix, accuracy/F1 per epoch — as images
4. Per-class recall from the confusion matrix
5. The 1★ queue: rank every review by P(1★) and read the top of it

**What this costs.** one GPU training run (a few minutes on `g4dn.2xlarge`; the API reserves the estimate — about 8,700 credits on the Team plan — and refunds the unused minutes, so a 90-second run ends up costing a few hundred), plus 50 credits per on-demand prediction. A re-run that finds the finished model pays only for its predictions.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`03_rating_classification.ipynb`](03_rating_classification.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | multiclass · review.rating (5 classes) |
| model / lane | GATv2 |
| accuracy | 0.6650 |
| f1_macro | 0.4655 |
| recall_1star | 0.3750 |
| baseline accuracy | 0.6311 |
| baseline f1_macro_note | always-5★ |
| training time | 2 min 41 s |
| credits charged (this run) | 100 |
| project | `c40a4efc-7d71-4534-85f1-f80135e71ab5` |
| model | `cb1c7b94-613b-413a-9bf7-50342f63a953` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 03_rating_classification.ipynb
```
