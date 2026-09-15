# 06 · Find unusual reviews (anomaly detection, unsupervised)

**Use case:** the marketplace team wants a first pass at suspicious reviews — text that does not fit the product, ratings out of line with the words — without labelled examples.

**Model / lane:** GraphMAE (unsupervised lane), anomaly scoring on the node embeddings

**Sub-tasks**
1. Define an unsupervised **anomaly detection** task on `review`
2. Train GraphMAE and score reviews by reconstruction error
3. Read what the run reports (`val_loss`, epochs)
4. Score a sample of reviews: which are flagged, with what score

Notebook: [`06_review_anomaly.ipynb`](06_review_anomaly.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | anomaly detection · review (unsupervised) |
| model / lane | GraphMAE (self-supervised) |
| val_loss | 0.6214 |
| epochs | 100 |
| flagged_in_sample | 1 |
| sample_size | 20 |
| note | self-supervised: reconstruction loss, no labels |
| training time | 1 min 6 s |
| credits charged (this run) | 1,000 |
| project | `828e4626-323e-4bee-9de4-7c825f7835d3` |
| model | `1489addf-1a1b-46c5-874d-afcbd9297d66` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 06_review_anomaly.ipynb
```
