#!/usr/bin/env bash
# Execute every notebook in order, in place (outputs are saved into the .ipynb), then rebuild the
# README results table. Needs: `pip install -r requirements.txt` and `langsat login` (or LANGSAT_API_KEY).
#   scripts/run_all.sh            # all
#   scripts/run_all.sh 01 07 10   # a subset, by number
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
want=("$@")
for dir in "$ROOT"/examples/amazon_reviews/[0-9][0-9]_*/; do
  n="$(basename "$dir" | cut -c1-2)"
  if [ ${#want[@]} -gt 0 ] && ! printf '%s\n' "${want[@]}" | grep -qx "$n"; then continue; fi
  nb="$(ls "$dir"/*.ipynb | head -1)"
  echo "━━━ $(basename "$nb")"
  ( cd "$dir" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=5400 "$(basename "$nb")" )
done
"$PY" "$ROOT/scripts/build_results_table.py"
