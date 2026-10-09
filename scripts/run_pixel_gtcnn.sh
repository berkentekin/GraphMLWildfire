#!/usr/bin/env bash
# The pixel-graph-time CNN (attention shift operator, 0.75 km nodes) on all 12 WildfireSpreadTS folds.
# Pass --model gtcnn for the node-only ablation; extra arguments go to every run.
# Needs data/processed/wildfirespreadts_benchmark_b2 and data/cache/pixels (see README), ~31 min per fold.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
for fold in $(seq 0 11); do
  log="data/logs/pixel_gtcnn_f${fold}.log"
  grep -qs "test AP" "$log" && continue
  echo "== fold $fold $(date '+%F %T')"
  uv run python -m graphml_wildfire.train --model pixel_gtcnn --graph attention --patch 2 --eval-batch-size 8 \
    --fold "$fold" "$@" > "$log" 2>&1 || echo "FAILED fold $fold (see $log)"
  tail -n 1 "$log"
done
