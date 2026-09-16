# 08 · The prediction API and model monitoring

**Use case.** The verified-purchase model (04) goes into an application: score a review the app just received (inductive — from its columns and its neighbours), score in batches, and watch the model's inputs drift.

**What you will learn**
1. Read the model's feature schema — what a request must carry
2. Inductive prediction: a row the platform has never stored, with `related_entities`
3. Batch prediction, and the `active:<project>` selector
4. Monitoring: feature health as a table and a chart, thresholds, alert configuration (email / webhook)
5. Recent-prediction monitoring for the model

**What this costs.** no training; 50 credits per prediction (batch items included).

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`08_predict_api_and_monitoring.ipynb`](08_predict_api_and_monitoring.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | serve + monitor the binary model from 04 |
| model / lane | GraphSAGE (default, labelled *Baseline*) |
| inductive_probability | 0.9087 |
| batch_scored | 5 |
| monitored_features | 9 |
| predictions_in_window | 70 |
| project | `28106708-9ec0-48e4-b292-d600b9bc2ed8` |
| model | `61b6e350-ea4c-4219-a694-00835a452fd5` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 08_predict_api_and_monitoring.ipynb
```
