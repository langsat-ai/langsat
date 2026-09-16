# 05 · Segment products (clustering, unsupervised)

**Use case.** Merchandising has 8,600 books and no taxonomy beyond 'Books' — they want natural segments from titles, descriptions, prices and *who reviews them*, without labelling anything.

**Model / lane:** GraphMAE (self-supervised) + clustering on the node embeddings

**What you will learn**
1. Define an unsupervised **clustering** task on `product`
2. Train GraphMAE (self-supervised on the relational graph) and cluster the embeddings
3. Read cluster quality from the Model tab: silhouette, Davies-Bouldin, sizes, a 2-D projection — as images
4. Assign a sample of products and read the segments by their members

**What this costs.** one GPU training run (a few minutes on `g4dn.2xlarge`; the API reserves the estimate — about 8,700 credits on the Team plan — and refunds the unused minutes, so a 90-second run ends up costing a few hundred), plus 50 credits per on-demand prediction. A re-run that finds the finished model pays only for its predictions.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`05_product_clustering.ipynb`](05_product_clustering.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | clustering · product (unsupervised) |
| model / lane | GraphMAE (self-supervised) |
| val_loss | 0.7757 |
| silhouette | 0.4934 |
| davies_bouldin | 0.8753 |
| clusters | 2 |
| note | quality numbers come from the Model tab cards |
| training time | 1 min 1 s |
| credits charged (this run) | 2,000 |
| project | `44eb67a5-aea9-427e-be04-5efeafe0b708` |
| model | `fef7e68f-e85f-49ce-ac60-a59a8447d457` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 05_product_clustering.ipynb
```
