#!/usr/bin/env bash
# Fetch/rebuild the tabular benchmark fixtures for the KAN examples (T1/T2).
#   - Iris (UCI, 150×4, 3-class) -> tests/fixtures/iris.csv           (committed)
#   - California housing (Géron mirror) -> cleaned 8-feature label-last CSV:
#       full  -> $DATA_DIR/california.csv                             (repo-external)
#       2000  -> tests/fixtures/california.csv                        (committed subset)
#
# The California raw has a categorical column (ocean_proximity) and NA rows
# (empty total_bedrooms); we drop both and put the target (median_house_value) last.
#
# Usage: scripts/dev/fetch_tabular_datasets.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/hakiko_ai_ws/03_implementation/nagare_data}"
mkdir -p "$DATA_DIR" "$REPO/tests/fixtures"

echo "== Iris =="
curl -fsSL --connect-timeout 20 \
  -o "$REPO/tests/fixtures/iris.csv" \
  https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
# UCI iris.data ends with a blank line; the loader skips empties.
echo "  iris rows: $(grep -c . "$REPO/tests/fixtures/iris.csv")"

echo "== California housing =="
curl -fsSL --connect-timeout 25 \
  -o "$DATA_DIR/housing_raw.csv" \
  https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
# cols 1-8 = features; col 9 = median_house_value (target); col 10 = ocean_proximity (drop);
# drop rows with empty total_bedrooms (col 5).
awk -F, 'NR>1 && $5!="" {print $1","$2","$3","$4","$5","$6","$7","$8","$9}' \
  "$DATA_DIR/housing_raw.csv" > "$DATA_DIR/california.csv"
head -2000 "$DATA_DIR/california.csv" > "$REPO/tests/fixtures/california.csv"
echo "  california full rows: $(wc -l < "$DATA_DIR/california.csv")  subset (committed): 2000"

echo "done. Fixtures in $REPO/tests/fixtures ; full California in $DATA_DIR"
