# 18 · Trend — time on the axis, the five grains, and the filter that freezes

**Use case.** A date column is the one field the SDK cannot recognise on its own: it has no schema, so `charts.column(axis="review.review_time")` is a chart of 2,772 categories, capped at 50, in descending order of count. `charts.date(column, grain)` is how you say what it is — and the four line and area shapes are what you draw once you have.

**What you will learn**
1. The four time shapes: `line`, `line_markers`, `area`, `stacked_area`
2. All five grains — year, quarter, month, week, date — and what each one costs in points
3. Why `top_n` does not exist on a lines trace, and what happens if you rank a line by measure
4. `date_between` **freezes**; `last_n_days` is resolved at query time — and on 2018 data returns nothing
5. A date in the LEGEND, which is the same trap one well over
6. Reading the series back as a pandas frame with `chart.data`

**What this costs.** nothing.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`18_trend.ipynb`](18_trend.ipynb) · outputs are from a real run on 2026-09-30.

## Result

| | |
|---|---|
| task | time on the axis · 4 shapes, 5 grains, the filter that freezes |
| model / lane | — |
| charts_built | 7 |
| months | 129 |
| days_in_data | 2772 |
| peak_month | 2014-12 |
| last_n_days_90_points | 0 |
| credits charged (this run) | 0 |
| project | `5c40e337-9753-4618-a7b3-574419f93c1e` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 18_trend.ipynb
```
