# 11 · Fine-tune a model on new data

**Use case.** The rating model has been in production for a while; a thousand new reviews have arrived. Retraining from scratch costs a full run and forgets nothing — but fine-tuning continues the existing version on the new rows, keeps the lineage, and takes minutes.

**Model / lane:** relational GNN (GraphSAGE), text embeddings; warm fine-tune continues from the parent checkpoint

**What you will learn**
1. Train a first version on the reviews up to a cut-off date (9,000 rows)
2. Add the 1,000 newer reviews with `sources.upload_finetune` — additive, the model stays
3. `finetune_verify_schema` (structure must match) and `finetune_compute_delta` (how many rows are new, what the API recommends)
4. `train(mode="fine_tune", from_model_id=…, training_mode="warm")` and the lineage v1 → v2
5. Compare parent and child metrics; the child's Model tab

**What this costs.** two GPU runs — the first is a normal training, the fine-tune reserves the same floor but runs faster (only the new rows are embedded).

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`11_finetune.ipynb`](11_finetune.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | fine-tune · regression on +1,000 newer reviews |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| parent_mae | 0.4951 |
| child_mae | 0.5069 |
| child_version | v2 |
| baseline parent_r2 | 0.4436 |
| baseline child_r2 | 0.4523 |
| training time | 0 min 44 s |
| credits charged (this run) | 0 |
| project | `da326da0-ae64-4c17-bbb0-d8b8b1919ad7` |
| model | `23382ce4-8945-483b-96da-af900fbd39e4` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 11_finetune.ipynb
```
