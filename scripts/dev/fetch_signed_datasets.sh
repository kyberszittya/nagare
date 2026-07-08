#!/usr/bin/env bash
# Fetch the SNAP signed-network datasets used by the `signed_link` / `auroc_eval`
# examples. These lived on kato15 at /tmp/hajdu/signed/; this reproduces them from
# source on any machine (kato15-independent).
#
# Usage:  scripts/dev/fetch_signed_datasets.sh [TARGET_DIR]
#   default TARGET_DIR = ~/hakiko_ai_ws/03_implementation/nagare_data/signed  (repo-external)
#
# Then, e.g.:
#   cargo run --release --example signed_link -- \
#     --data "$TARGET_DIR/soc-sign-bitcoinalpha.csv" --scale 10 --seed 0
#
# Reproduced pure-Nagare closed-form AUROC (seed 0, --scale 10), 2026-07-09 on M5 Pro:
#   Bitcoin-Alpha 0.9044 · Bitcoin-OTC 0.9275 · Slashdot 0.9097 · Epinions 0.9506
# (matches the documented ceilings 0.904 / 0.928 / 0.910 / 0.951).
set -euo pipefail

TARGET_DIR="${1:-$HOME/hakiko_ai_ws/03_implementation/nagare_data/signed}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

URLS=(
  https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz
  https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
  https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz
  https://snap.stanford.edu/data/soc-sign-epinions.txt.gz
)

for url in "${URLS[@]}"; do
  gz="$(basename "$url")"
  plain="${gz%.gz}"
  if [ -f "$plain" ]; then
    echo "skip $plain (exists)"
    continue
  fi
  echo "fetch $gz"
  curl -fsSL --connect-timeout 20 -o "$gz" "$url"
  gunzip -f "$gz"
  echo "  -> $plain ($(grep -vc '^#' "$plain") edges)"
done

echo "signed datasets ready in: $TARGET_DIR"
