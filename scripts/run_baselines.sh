#!/usr/bin/env bash
# Fast non-graph baselines on all 12 WildfireSpreadTS folds; extra arguments go to every run.
# Summarise with: uv run python -m graphml_wildfire.benchmark summary
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs

BASELINES=(
  "--model persistence"
  "--model logreg --window 1"
  "--model logreg --window 5"
  "--model mlp --window 5"
  "--model cnn --window 5"
)
for baseline in "${BASELINES[@]}"; do
  for fold in $(seq 0 11); do
    name="$(echo "$baseline" | tr -d '-' | tr ' ' '_')_f${fold}"
    echo "== $baseline --fold $fold"
    uv run python -m graphml_wildfire.train $baseline --fold "$fold" "$@" > "data/logs/${name}.log" 2>&1
    tail -n 1 "data/logs/${name}.log"
  done
done
