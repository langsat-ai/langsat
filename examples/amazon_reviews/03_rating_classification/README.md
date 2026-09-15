# 03 · Predict the star rating as a class (1–5)

**Use case:** the same question as 02, but the product team wants class probabilities — how sure is the model this is a 1-star review? — to route angry customers to a person.

**Model / lane:** relational GNN — this time the GATv2 architecture (`model_key="gatv2_full"`), 5 classes, text embeddings

**Sub-tasks**
1. Define the task as **multiclass** classification on `rating`
2. Train the relational GNN
3. Accuracy and macro-F1 on the test split vs the always-5★ baseline (63% of reviews are 5★)
4. Per-class precision / recall from the confusion matrix
5. Probabilities for one review

Notebook: [`03_rating_classification.ipynb`](03_rating_classification.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | multiclass · review.rating (5 classes) |
| model / lane | GATv2 |
| accuracy | 0.6650 |
| f1_macro | 0.4655 |
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
