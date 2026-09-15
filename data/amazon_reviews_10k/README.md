# Amazon book reviews — a 10,000-review connected sample

Three related tables, the way they arrive from a review system, sampled so that **every review's customer and
product are present** — the relational graph is complete (a review always has both neighbours). Small enough to
upload in seconds and to clean for free (under Langsat's 500K-row line), big enough for every kind of task.

| file | rows | columns | keys |
|---|---|---|---|
| `customer.csv` | 9,821 | `customer_id`, `customer_name` | PK `customer_id` |
| `product.csv` | 8,607 | `product_id`, `category`, `brand`, `title`, `description`, `price` | PK `product_id` |
| `review.csv` | 10,000 | `review_time`, `customer_id`, `product_id`, `rating`, `verified`, `review_text`, `summary` | FK → customer, FK → product, time `review_time` |

```
customer ──< review >── product
              │
         review_time (2008-01 … 2018-09)
```

Facts that matter for the tasks: ratings are skewed (1★ 282 · 2★ 358 · 3★ 850 · 4★ 2,199 · 5★ 6,311 — 63 % five-star);
71 % of reviews are verified purchases; almost every product is a book (`category` starts with `Books|`); a
product has up to 14 reviews and a customer up to 4 (155 customers wrote more than one); `review_text` is free text
with quotes and line breaks (a real CSV parser is needed — Langsat's is).

## Source and attribution

A connected-component sample of the Amazon Reviews (2018) dataset — Jianmo Ni, Jiacheng Li, Julian McAuley,
*Justifying recommendations using distantly-labeled reviews and fine-grained aspects*, EMNLP 2019 — in the
relational layout used by [RelBench](https://relbench.stanford.edu) (`rel-amazon`). Reviewer display names and
review text are as published by Amazon. Use for research and evaluation; see the original dataset's terms for
anything else.
