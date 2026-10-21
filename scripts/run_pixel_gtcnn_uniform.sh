#!/usr/bin/env bash
# The pixel-graph-time CNN on all 12 WildfireSpreadTS folds; extra arguments go to every run.
# Needs the training cache data/cache/b2 (see README); ~25 min per fold on an RTX 5080.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
for fold in $(seq 0 11); do
  log="data/logs/pixel_gtcnn_uniform_f${fold}.log"
  grep -qs "test AP" "$log" && continue
  echo "== fold $fold $(date '+%F %T')"
  uv run python -m graphml_wildfire.train --model pixel_gtcnn_uniform --fold "$fold" "$@" > "$log" 2>&1 || echo "FAILED fold $fold (see $log)"
  tail -n 1 "$log"
done
