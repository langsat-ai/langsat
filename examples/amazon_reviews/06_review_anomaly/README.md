# 06 · Find unusual reviews (anomaly detection, unsupervised)

**Use case.** The marketplace team wants a first pass at suspicious reviews — text that does not fit the product, ratings out of line with the words — without labelled examples.

**Model / lane:** GraphMAE (self-supervised), anomaly scoring on the node embeddings

**What you will learn**
1. Define an unsupervised **anomaly detection** task on `review`
2. Train GraphMAE and score reviews by reconstruction error
3. Read the anomaly score distribution and the threshold from the Model tab — as an image
4. Score a sample: which are flagged, with what score, and what they say

**What this costs.** one GPU training run (a few minutes on `g4dn.2xlarge`; the API reserves the estimate — about 8,700 credits on the Team plan — and refunds the unused minutes, so a 90-second run ends up costing a few hundred), plus 50 credits per on-demand prediction. A re-run that finds the finished model pays only for its predictions.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`06_review_anomaly.ipynb`](06_review_anomaly.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | anomaly detection · review (unsupervised) |
| model / lane | GraphMAE (self-supervised) |
| val_loss | 0.6214 |
| threshold | 0.4352 |
| flagged_in_sample | 2 |
| sample_size | 30 |
| training time | 1 min 6 s |
| credits charged (this run) | 1,500 |
| project | `828e4626-323e-4bee-9de4-7c825f7835d3` |
| model | `1489addf-1a1b-46c5-874d-afcbd9297d66` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 06_review_anomaly.ipynb
```
