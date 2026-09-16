# 12 · Training options — text embeddings, architectures, the bake-off, parallel embedding

**Use case.** Same task as 02 (predict the rating), three ways to train it. Which options matter on this data, what they cost, and how to keep the best version serving.

**Model / lane:** GraphSAGE / GATv2 / GIN — the platform picks in max mode

**What you will learn**
1. v1 (from 02): the default — GraphSAGE with text embeddings
2. v2: `enable_text_embedding=False` — how much the review text is worth
3. v3: `max_mode=True` — the architecture bake-off (every architecture is trained, the best one ships)
4. `fan_out_workers`: what it parallelises and why it did not fire at this size
5. Compare the versions in one table, re-activate the best

**What this costs.** two more GPU runs on the regression project (the bake-off may launch several boxes; credits are estimated first and unused minutes refunded).

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`12_training_options.ipynb`](12_training_options.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | regression · text on/off · max-mode bake-off |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| mae_v1_Baseline | 0.4967 |
| mae_v2_Baseline | 0.7299 |
| mae_v3_Baseline | 0.4760 |
| serving | v3 |
| credits charged (this run) | 0 |
| project | `8847cd90-8444-4934-aad9-8faa372216bf` |
| model | `927511dc-161d-4a92-ac2b-6e3b97a0c9fc` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 12_training_options.ipynb
```
