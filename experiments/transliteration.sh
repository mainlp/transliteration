#!/usr/bin/env bash
set -euo pipefail

MODEL=large-v2
LAYERS=$(seq -s ' ' 0 31)
SIGMAS=(0.1 0.2 0.3 0.4 0.5)
mkdir -p artifacts/vectors artifacts/results artifacts/sigma

best_sigma() {
  local log="$1" category="$2"
  local column
  column=$(awk -v key="${category}:" '$1 == key {print $2}' "$log")
  case "$column" in
    *_1_acc) echo 0.1 ;; *_2_acc) echo 0.2 ;; *_3_acc) echo 0.3 ;;
    *_4_acc) echo 0.4 ;; *_5_acc) echo 0.5 ;;
    *) echo "Could not determine sigma for $category" >&2; return 1 ;;
  esac
}

run_direction() {
  local lang="$1" direction="$2" prompt="$3" base_vector="$4" category="$5"
  local sigma_csv="artifacts/sigma/${lang}_${direction}.csv"
  local sigma_log="${sigma_csv%.csv}.log"

  whisper-steer search-sigma --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "$prompt" --layers-to-steer $LAYERS \
    --sigmas "${SIGMAS[@]}" --selection max -v "$base_vector" \
    -s bias-sigma-0.0 -p mean -b whisper -o "$sigma_csv" | tee "$sigma_log"
  local sigma
  sigma=$(best_sigma "$sigma_log" "$category")

  # no-prompt, prompt, and zero-shot rows in Figures 4/5
  whisper-steer infer --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "" --layers-to-steer $LAYERS -v "$base_vector" \
    -s bias-sigma-0.0 -p mean -b whisper -o "artifacts/results/${lang}_${direction}_no_prompt.csv"
  whisper-steer infer --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "$prompt" --layers-to-steer $LAYERS -v "$base_vector" \
    -s bias-sigma-0.0 -p mean -b whisper -o "artifacts/results/${lang}_${direction}_prompt.csv"
  whisper-steer infer --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "" --layers-to-steer $LAYERS -v "$base_vector" \
    -s "bias-sigma-${sigma}" -p mean -b whisper -o "artifacts/results/${lang}_${direction}_zero_shot.csv"

  # Learn the language-specific vector from zero-shot pseudo-labels.
  whisper-steer second-pass-train --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "$prompt" --max-files 10 -v "$base_vector" \
    -s "bias-sigma-${sigma}" -p mean -b whisper
  local pseudo="artifacts/vectors/${lang}_ft_$(basename "${base_vector%.npz}").npz"
  local pseudo_sigma_csv="artifacts/sigma/${lang}_${direction}_pseudo.csv"
  local pseudo_sigma_log="${pseudo_sigma_csv%.csv}.log"
  whisper-steer search-sigma --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "$prompt" --layers-to-steer $LAYERS \
    --sigmas "${SIGMAS[@]}" --selection max -v "$pseudo" \
    -s bias-sigma-0.0 -p mean -b whisper -o "$pseudo_sigma_csv" | tee "$pseudo_sigma_log"
  local pseudo_sigma
  pseudo_sigma=$(best_sigma "$pseudo_sigma_log" "$category")
  whisper-steer infer --model "$MODEL" --src-lang "$lang" --tgt-lang "$lang" \
    --src-prompt "" --trg-prompt "" --layers-to-steer $LAYERS -v "$pseudo" \
    -s "bias-sigma-${pseudo_sigma}" -p mean -b whisper -o "artifacts/results/${lang}_${direction}_pseudo_label.csv"
}

ROMAN_VECTOR=artifacts/vectors/sr_cyrl_latn_large-v2_10.npz
CYRILLIC_VECTOR=artifacts/vectors/reverse_sr_cyrl_latn_large-v2_10.npz

run_direction ru_ru romanization "eto russkoye predlozheniye" "$ROMAN_VECTOR" roman
run_direction hi_in romanization "yah ek hindi vaakya hai" "$ROMAN_VECTOR" roman
run_direction el_gr romanization "auti einai mia elliniki protasi" "$ROMAN_VECTOR" roman
run_direction ja_jp romanization "kore wa nihongo no bun desu" "$ROMAN_VECTOR" roman
run_direction ko_kr romanization "igeoseun hangugeo munjangibnida" "$ROMAN_VECTOR" roman
run_direction it_it cyrillization "куэста э уна фразе итальяна" "$CYRILLIC_VECTOR" cyrl
run_direction hi_in cyrillization "йе эк хинди вакья хэ" "$CYRILLIC_VECTOR" cyrl
run_direction el_gr cyrillization "афти инэ мия эллиники протаси" "$CYRILLIC_VECTOR" cyrl
run_direction ja_jp cyrillization "корэ ва нихонго но бун десу" "$CYRILLIC_VECTOR" cyrl
run_direction ko_kr cyrillization "игосын хангуго мунджанимнида" "$CYRILLIC_VECTOR" cyrl
