# 15 · Gradient boosting with a feature-engineering plan

**Use case.** The graph models learn features from the neighbourhood; a gradient-boosted tree wants them written down as columns. Langsat's GBM lane does that with a **feature-engineering SQL plan** — SQL over the cleaned tables that produces one row per entity with engineered columns, written by the model for your task, or by you in the app — and trains LightGBM on it. This chapter reviews the cleaning plan on a training project, has the model write the feature plan, reads the SQL and its safety checks, trains LightGBM, reads importance, and scores a new review with its related rows.

**Model / lane:** LightGBM on a feature-plan table

**What you will learn**
1. The model menu with a `gbm` family next to `neural`, and how to pick it (`model_key="lightgbm"`)
2. The cleaning plan on a `data_science` project (the training pipeline runs it — review it first)
3. `feature-plan/generate`: the model writes the feature SQL for your task; read the SQL, the columns, the roles, the warnings
4. What the validator refuses — including a leak the correlation check cannot see
5. Editing the SQL yourself: the app's Feature Engineering panel; why a key is refused there
6. Train LightGBM; the engine fields on the model row; feature importance
7. Predict from a new row: a GBM model scores values, and a plan that joins related tables needs them inline (`related_data`)

**What this costs.** one GBM training run (a few minutes; the estimate is reserved and unused minutes refunded), one feature-plan generation (1,000 credits — a model call, refunded on any failure) and a prediction at 50 credits.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`15_gbm_feature_plan.ipynb`](15_gbm_feature_plan.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | regression · LightGBM on a feature-engineering SQL plan |
| model / lane | lightgbm |
| mae | 0.5348 |
| rmse | 0.7939 |
| r2 | 0.3634 |
| n_features | 58 |
| plan_columns | 51 |
| feature_plan_origin | llm |
| predicted_rating | 4.7400 |
| training time | 1 min 11 s |
| credits charged (this run) | 1,543 |
| project | `0613fe4c-8ba8-4082-a131-e25a3164f2b4` |
| model | `d01d4059-6e25-418f-82c3-dfe078f67106` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 15_gbm_feature_plan.ipynb
```
