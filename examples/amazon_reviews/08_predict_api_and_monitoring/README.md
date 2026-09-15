# 08 · The prediction API and model monitoring

**Use case:** the verified-purchase model (04) goes into an application: score a review the app just received
(inductive — from features, not a stored row), score in batches, and watch the model's inputs and outputs drift.

**Sub-tasks**
1. Read the model's feature schema (`predict.model`)
2. Inductive prediction from raw features
3. Batch prediction
4. Monitoring summary, thresholds and alert configuration
5. Recent prediction monitoring for the model

Notebook: [`08_predict_api_and_monitoring.ipynb`](08_predict_api_and_monitoring.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | serve + monitor the binary model from 04 |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| batch_scored | 3 |
| monitoring_keys | ['state', 'model_id', 'project_name', 'task_type', 'period_days', 'version', 'family', 'version_label'] |
| project | `28106708-9ec0-48e4-b292-d600b9bc2ed8` |
| model | `61b6e350-ea4c-4219-a694-00835a452fd5` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 08_predict_api_and_monitoring.ipynb
```
