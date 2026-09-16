# 13 · Choose your model — tabular vs relational

**Use case.** The same question as chapter 02 — predict a review's rating — trained two ways. A **tabular** project holds one table and trains a tabular network (MLP, FT-Transformer, ResNet, TabNet); a **relational** project holds several tables joined by keys and trains a graph network (GraphSAGE, GAT, GIN) that also reads a review's neighbours: the product's other reviews, the customer's other reviews. This chapter shows how the mode is decided, what each menu offers, trains two tabular models on a flattened copy of the data, and compares them with the relational model from chapter 12.

**Model / lane:** tabular: FT-Transformer and TabNet · relational (from 12): GraphSAGE bake-off winner

**What you will learn**
1. How Langsat picks the mode: one file → tabular, several files → relational
2. Read the model menu for each mode from the API (`/tiers/{plan}/models?mode=…`) — what your plan lets you pick
3. Build a single flattened table (review + product + customer columns) and create a tabular project
4. Train `ft_transformer`, then `tabnet`, with `model_key` — and read their Model tabs as images
5. Compare tabular vs relational on the same target: what the neighbours are worth
6. Predict from a tabular model without `related_entities` — the trade-off in one call

**What this costs.** two GPU training runs on the tabular project (a few minutes each; the estimate is reserved and unused minutes refunded) plus a few predictions at 50 credits.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`13_choose_your_model.ipynb`](13_choose_your_model.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | regression · tabular (FT-Transformer, TabNet) vs relational |
| model / lane | ft_transformer |
| mae_tabular_ft_transformer | 0.4706 |
| mae_tabular_tabnet | 0.5474 |
| mae_relational_Baseline | 0.4760 |
| credits charged (this run) | 50 |
| project | `8af313bf-c0a1-4514-bb97-0c47f1ec05c0` |
| model | `93eb4467-ca17-4986-b441-7a619eb6db03` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 13_choose_your_model.ipynb
```
