#!/usr/bin/env bash
# Reproduce Tables 3-4: encoder-only steering on Whisper and OWSM-CTC v4.
set -euo pipefail

SIGMAS=(0.1 0.2 0.3 0.4 0.5)
mkdir -p artifacts/vectors artifacts/results artifacts/sigma

best_sigma() {
  local log="$1" category="$2" column
  column=$(awk -v key="${category}:" '$1 == key {print $2}' "$log")
  case "$column" in
    *_1_acc) echo 0.1 ;; *_2_acc) echo 0.2 ;; *_3_acc) echo 0.3 ;;
    *_4_acc) echo 0.4 ;; *_5_acc) echo 0.5 ;;
    *) echo "Could not determine sigma for $category" >&2; return 1 ;;
  esac
}

run_backend() {
  local backend="$1" model="$2" lang="$3" stem="$4"
  local vector="artifacts/vectors/${backend}_${stem}.npz"
  whisper-steer train --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "" --max-files 10 --steer encoder \
    -v "$vector" -s bias-sigma-0.1 -p mean -b "$backend"

  for direction_vector in "$vector" "artifacts/vectors/reverse_${backend}_${stem}.npz"; do
    local label category sigma_log sigma
    label=$(basename "${direction_vector%.npz}")
    if [[ "$lang" == sr_rs && "$direction_vector" == *"/reverse_"* ]]; then
      category=sr_to_cyl
    elif [[ "$lang" == sr_rs ]]; then
      category=sr_to_lat
    elif [[ "$direction_vector" == *"/reverse_"* ]]; then
      category=zh_to_sim
    else
      category=zh_to_trad
    fi
    sigma_log="artifacts/sigma/${label}.log"
    whisper-steer search-sigma --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "" --steer encoder --sigmas "${SIGMAS[@]}" \
      -v "$direction_vector" -s bias-sigma-0.0 -p mean -b "$backend" \
      -o "artifacts/sigma/${label}.csv" | tee "$sigma_log"
    sigma=$(best_sigma "$sigma_log" "$category")
    for test_sigma in 0.0 "$sigma"; do
      whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
        --src-prompt "" --trg-prompt "" --steer encoder -v "$direction_vector" \
        -s "bias-sigma-${test_sigma}" -p mean -b "$backend" \
        -o "artifacts/results/${label}_${test_sigma}.csv"
    done
    if [[ "$backend" == whisper ]]; then
      whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
        --src-prompt "" --trg-prompt "" --steer encoder -v "$direction_vector" \
        -s bias-sigma-1.0 -p mean -b whisper \
        -o "artifacts/results/${label}_fixed_1.0.csv"
    fi
  done
}

run_backend whisper large-v2 sr_rs sr_cyrl_latn_large-v2_10
run_backend whisper large-v2 cmn_hans_cn zh_sim_trad_large-v2_10
run_backend owsm-ctc espnet/owsm_ctc_v4_1B sr_rs sr_cyrl_latn_10
run_backend owsm-ctc espnet/owsm_ctc_v4_1B cmn_hans_cn zh_sim_trad_10
