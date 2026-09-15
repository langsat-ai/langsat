# 04 · Is this a verified purchase? (binary classification)

**Use case:** trust & safety wants to flag reviews that read like unverified ones — a binary score per review that can rank the whole table.

**Model / lane:** relational GNN — default GraphSAGE (*Baseline*), text embeddings

**Sub-tasks**
1. Define a **binary** task on `review.verified`
2. Train the relational GNN
3. AUROC, PR-AUC (vs the class-prior baseline), precision / recall / F1
4. Rank all reviews with `predict.top` and score a batch of entity ids

Notebook: [`04_verified_purchase_binary.ipynb`](04_verified_purchase_binary.ipynb) · outputs are from a real run on 2026-09-15.

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
| credits charged (this run) | 350 |
| project | `28106708-9ec0-48e4-b292-d600b9bc2ed8` |
| model | `61b6e350-ea4c-4219-a694-00835a452fd5` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 04_verified_purchase_binary.ipynb
```
