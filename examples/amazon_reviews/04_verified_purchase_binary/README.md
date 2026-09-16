# 04 · Is this a verified purchase? (binary classification)

**Use case.** Trust & safety wants to flag reviews that read like unverified ones — a probability per review that can rank the whole table and set a threshold.

**Model / lane:** relational GNN — default GraphSAGE (*Baseline*), text embeddings

**What you will learn**
1. Define a **binary** task on `review.verified`
2. AUROC, PR-AUC (vs the class prior), precision / recall / F1 — what each floor means
3. The Model tab: the ROC curve and the confusion matrix as images
4. Rank every review both ways; score a batch; look at the probability distribution
5. Choose a threshold: what the confusion matrix says about the default 0.5

**What this costs.** one GPU training run (a few minutes on `g4dn.2xlarge`; the API reserves the estimate — about 8,700 credits on the Team plan — and refunds the unused minutes, so a 90-second run ends up costing a few hundred), plus 50 credits per on-demand prediction. A re-run that finds the finished model pays only for its predictions.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`04_verified_purchase_binary.ipynb`](04_verified_purchase_binary.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | binary · review.verified |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| auroc | 0.6281 |
| pr_auc | 0.7831 |
| f1 | 0.7783 |
| accuracy | 0.6670 |
| baseline auroc | 0.5000 |
| baseline pr_auc | 0.6915 |
| training time | 1 min 9 s |
| credits charged (this run) | 1,100 |
| project | `28106708-9ec0-48e4-b292-d600b9bc2ed8` |
| model | `61b6e350-ea4c-4219-a694-00835a452fd5` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 04_verified_purchase_binary.ipynb
```
