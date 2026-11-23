#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/results
python -m whisper_steer.probe --language sr_rs --model large-v2 \
  --vector artifacts/vectors/sr_cyrl_latn_large-v2_10.npz \
  --examples-per-script 50 --output-csv artifacts/results/probe_serbian.csv
python -m whisper_steer.probe --language cmn_hans_cn --model large-v2 \
  --vector artifacts/vectors/zh_sim_trad_large-v2_10.npz \
  --examples-per-script 50 --output-csv artifacts/results/probe_mandarin.csv
