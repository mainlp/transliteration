#!/usr/bin/env bash
set -euo pipefail

MODELS=(tiny base small medium large large-v2 large-v3)
LAYERS_BY_MODEL() {
  case "$1" in
    tiny) seq -s ' ' 0 3 ;;
    base) seq -s ' ' 0 5 ;;
    small) seq -s ' ' 0 11 ;;
    medium) seq -s ' ' 0 23 ;;
    large|large-v2|large-v3) seq -s ' ' 0 31 ;;
  esac
}

mkdir -p artifacts/vectors artifacts/results artifacts/sigma

run_language() {
  local lang="$1" stem="$2" source_prompt="$3" target_prompt="$4"
  for model in "${MODELS[@]}"; do
    local layers vector reverse_vector
    layers=$(LAYERS_BY_MODEL "$model")
    vector="artifacts/vectors/${stem}_${model}_10.npz"
    reverse_vector="artifacts/vectors/reverse_${stem}_${model}_10.npz"

    whisper-steer train --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "$source_prompt" --trg-prompt "$target_prompt" --max-files 10 \
      -v "$vector" -s bias-sigma-0.1 -p mean -b whisper

    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "" --layers-to-steer $layers -v "$vector" \
      -s bias-sigma-0.0 -p mean -b whisper \
      -o "artifacts/results/${lang}_${model}_no_prompt.csv"
    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "$target_prompt" --layers-to-steer $layers -v "$vector" \
      -s bias-sigma-0.0 -p mean -b whisper \
      -o "artifacts/results/${lang}_${model}_forward_prompt.csv"
    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "$source_prompt" --layers-to-steer $layers -v "$reverse_vector" \
      -s bias-sigma-0.0 -p mean -b whisper \
      -o "artifacts/results/${lang}_${model}_reverse_prompt.csv"

    local forward_metric reverse_metric
    case "$lang" in
      sr_rs) forward_metric=sr_to_lat; reverse_metric=sr_to_cyl ;;
      cmn_hans_cn) forward_metric=zh_to_trad; reverse_metric=zh_to_sim ;;
    esac

    # Section 3.4: choose sigma on validation by highest mean accuracy, then
    # evaluate that selected value once on the held-out test split.
    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "" --layers-to-steer $layers \
      -v "$vector" -s bias-sigma-0.0 -p mean -b whisper \
      --tune-sigma --selection mean --metric "$forward_metric" \
      --validation-output-csv "artifacts/sigma/${lang}_${model}_forward.csv" \
      -o "artifacts/results/${lang}_${model}_forward_steer.csv"
    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "" --layers-to-steer $layers \
      -v "$reverse_vector" -s bias-sigma-0.0 -p mean -b whisper \
      --tune-sigma --selection mean --metric "$reverse_metric" \
      --validation-output-csv "artifacts/sigma/${lang}_${model}_reverse.csv" \
      -o "artifacts/results/${lang}_${model}_reverse_steer.csv"

    # One-shot uses the first pair passing the same theta=0.4 filter.
    vector="artifacts/vectors/${stem}_${model}_1.npz"
    reverse_vector="artifacts/vectors/reverse_${stem}_${model}_1.npz"
    whisper-steer train --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "$source_prompt" --trg-prompt "$target_prompt" --max-files 1 \
      -v "$vector" -s bias-sigma-0.1 -p mean -b whisper
    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "" --layers-to-steer $layers \
      -v "$vector" -s bias-sigma-0.0 -p mean -b whisper \
      --tune-sigma --selection mean --metric "$forward_metric" \
      --validation-output-csv "artifacts/sigma/${lang}_${model}_forward_one_shot.csv" \
      -o "artifacts/results/${lang}_${model}_forward_one_shot.csv"
    whisper-steer infer --model "$model" --src-lang "$lang" --tgt-lang "$lang" \
      --src-prompt "" --trg-prompt "" --layers-to-steer $layers \
      -v "$reverse_vector" -s bias-sigma-0.0 -p mean -b whisper \
      --tune-sigma --selection mean --metric "$reverse_metric" \
      --validation-output-csv "artifacts/sigma/${lang}_${model}_reverse_one_shot.csv" \
      -o "artifacts/results/${lang}_${model}_reverse_one_shot.csv"
  done
}

run_language sr_rs sr_cyrl_latn "Ово је српска реченица" "Ovo je srpska rečenica"
run_language cmn_hans_cn zh_sim_trad "这是一句普通话" "這是一句普通話"
