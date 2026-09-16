# 07 · Forecast monthly review volume and rating (zero-shot, no training)

**Use case.** Community managers staff the moderation queue a few months ahead — how many reviews will arrive per month, and is the average rating drifting? Nothing is trained: Chronos-2 forecasts the project's own series on the spot.

**Model / lane:** Chronos-2 zero-shot over `review_time`

**What you will learn**
1. Forecast the monthly review **count** with 10 / 50 / 90 % quantiles and draw the band after the history
2. Forecast the **mean rating** the same way
3. Split a forecast by a category (`category_col`) — verified vs unverified reviews
4. Read the response's calibration flag and warnings, and what a forecast costs

**What this costs.** no training; each forecast call costs 40 + 6 × horizon credits (6 months → 76 credits); a category split costs that × `max_categories`.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`07_forecast_review_volume.ipynb`](07_forecast_review_volume.ipynb) · outputs are from a real run on 2026-09-16.

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
| credits charged (this run) | 304 |
| project | `0b6ffd93-08da-4220-9dfa-5b62b5f5951f` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 07_forecast_review_volume.ipynb
```
