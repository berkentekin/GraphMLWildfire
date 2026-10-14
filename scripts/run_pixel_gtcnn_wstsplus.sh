#!/usr/bin/env bash
# The pixel-graph-time CNN on the 4 WSTS+ folds (2016-2023); extra arguments go to every run.
# Needs the training cache data/cache/b2 with all eight years (see README). One fold at a time: the test window
# of the WSTS+ years does not fit next to a second training run on a 16 GB GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
for fold in 0 1 2 3; do
  log="data/logs/pixel_gtcnn_wstsplus_f${fold}.log"
  grep -qs "test AP" "$log" && continue
  echo "== fold $fold $(date '+%F %T')"
  uv run python -m graphml_wildfire.train --model pixel_gtcnn --protocol wstsplus --fold "$fold" "$@" > "$log" 2>&1 || echo "FAILED fold $fold (see $log)"
  tail -n 1 "$log"
done
