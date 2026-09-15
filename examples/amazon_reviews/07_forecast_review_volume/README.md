# 07 · Forecast monthly review volume and rating (zero-shot, no training)

**Use case:** community managers staff the moderation queue a few months ahead — how many reviews will arrive per
month, and is the average rating drifting?

**Model / lane:** Chronos-2 zero-shot over the project's own `review_time` series — nothing is trained; the cost is
40 + 6 × horizon credits per call.

**Sub-tasks**
1. Reuse the explore project (01)
2. A 6-month forecast of the review **count** with 10 / 50 / 90 % quantiles
3. A 6-month forecast of the **mean rating**
4. Plot history and the forecast band; read the response's calibration flag and warnings

The sample runs 2008 → September 2018 and thins out in its last weeks; the forecast follows that tail — which is what a
forecaster should do.

Notebook: [`07_forecast_review_volume.ipynb`](07_forecast_review_volume.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | forecast · monthly review count + mean rating (zero-shot) |
| model / lane | Chronos-2 (zero-shot) |
| horizon_months | 6 |
| calibrated | yes |
| next_month_reviews_median | 0.0000 |
| next_month_mean_rating_median | 4.0900 |
| note | zero-shot: no held-out metric by design |
| credits charged (this run) | 152 |
| project | `0b6ffd93-08da-4220-9dfa-5b62b5f5951f` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 07_forecast_review_volume.ipynb
```
