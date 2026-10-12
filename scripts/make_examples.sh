#!/usr/bin/env bash
# Regenerate the example run shown in the README (docs/example-n17/).
# Seed 2 is the run that escapes the 4 + sqrt(2)/2 trap for N = 17.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/sqpack run 17 --seed 2 --out runs/example_n17 --frames 100 --no-html --quiet

out=docs/example-n17
mkdir -p "$out"
rm -f "$out"/*.png "$out"/*.gif
cp runs/example_n17/figures/*.png runs/example_n17/figures/*.gif "$out"/
echo "updated $out"
