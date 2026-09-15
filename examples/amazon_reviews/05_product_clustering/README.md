# 05 · Segment products (clustering, unsupervised)

**Use case:** merchandising has 8,600 books and no taxonomy beyond 'Books' — they want natural segments from titles, descriptions, prices and who reviews them.

**Model / lane:** GraphMAE (unsupervised lane) + k-means on the node embeddings

**Sub-tasks**
1. Define an unsupervised **clustering** task on `product`
2. Train GraphMAE (self-supervised on the relational graph) and cluster the embeddings
3. Read what the run reports (reconstruction `val_loss`, epochs)
4. Assign a sample of products to their clusters and look at the segments

Notebook: [`05_product_clustering.ipynb`](05_product_clustering.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | clustering · product (unsupervised) |
| model / lane | GraphMAE (self-supervised) |
| val_loss | 0.7757 |
| epochs | 100 |
| clusters_in_sample | 2 |
| note | self-supervised: reconstruction loss, no labels |
| training time | 1 min 1 s |
| credits charged (this run) | 1,000 |
| project | `44eb67a5-aea9-427e-be04-5efeafe0b708` |
| model | `fef7e68f-e85f-49ce-ac60-a59a8447d457` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 05_product_clustering.ipynb
```
