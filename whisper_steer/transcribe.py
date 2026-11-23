from contextlib import nullcontext
import torch
import pandas as pd
from tqdm import tqdm


from .hooks import MLPTimeAverager, LayerwiseSteer
from .metrics import chinese_traditional_accuracy, chinese_simplified_accuracy, serbian_latin_accuracy, serbian_cyrillic_accuracy, romanization_accuracy, japanese_romanization_accuracy, cyrillization_accuracy, cyrillization_accuracy_japanese
from .backend import get_adapter
from .config import DEVICE, STEER_CONFIGS
from .data import fleurs_audio_path, load_fleurs_split
from .language_codes import FLEURS_MAP

def _fleurs_lang_to_whisper_lang(fleurs_lang):
    return FLEURS_MAP.get(fleurs_lang, {}).get("whisper")

def _fleurs_lang_to_isocode(fleurs_lang):
    return FLEURS_MAP.get(fleurs_lang, {}).get("iso")

@torch.no_grad()
def collect_means_and_text(model, audio, fleurs_lang, initial_prompt, pool_config, backend_config, which="decoder"):
    adapter = get_adapter(backend_config)
    with MLPTimeAverager(model, pool_config=pool_config, backend_config=backend_config, which=which) as cap:
        res = adapter.transcribe(
            model=model,
            audio_path=audio,
            language=_fleurs_lang_to_whisper_lang(fleurs_lang),
            task="transcribe",
            initial_prompt=initial_prompt,
        )
    text = res["text"].strip()
    A = cap.means()
    return A, text


@torch.no_grad()
def evaluate(model, steer_weights, src_lang, tgt_lang, tgt_prompt, tgt_script, steer_config, pool_config, backend_config, layers_to_steer, cache_dir, enc_steer_weights=None):
    if tgt_prompt == "":
        tgt_prompt = None
    adapter = get_adapter(backend_config)
    ds = load_fleurs_split(src_lang, "test", cache_dir)
    results = []
    with tqdm(ds) as pbar:
        for item in pbar:
            audio = fleurs_audio_path(item, src_lang, "test", cache_dir)
            dec_ctx = (
                LayerwiseSteer(model, steer_weights, steer_config, pool_config, backend_config, layers_to_steer, target="decoder")
                if steer_weights is not None else nullcontext()
            )
            enc_ctx = (
                LayerwiseSteer(model, enc_steer_weights, steer_config, pool_config, backend_config, range(len(enc_steer_weights)), target="encoder")
                if enc_steer_weights is not None else nullcontext()
            )
            with dec_ctx, enc_ctx:
                trans_pred = adapter.transcribe(
                    model=model,
                    audio_path=audio,
                    language=_fleurs_lang_to_whisper_lang(src_lang),
                    task="transcribe", 
                    initial_prompt=tgt_prompt,
                )["text"].strip()
            # metrics here:
            roman_metric = japanese_romanization_accuracy if src_lang == "ja_jp" else romanization_accuracy
            cyrillic_metric = cyrillization_accuracy_japanese if src_lang == "ja_jp" else cyrillization_accuracy
            roman_acc = roman_metric(trans_pred, item["transcription"])
            cyrl_acc = cyrillic_metric(trans_pred, item["transcription"])
            zh_to_trad_acc = chinese_traditional_accuracy(trans_pred, item["transcription"])
            zh_to_sim_acc = chinese_simplified_accuracy(trans_pred, item["transcription"])
            sr_to_cyl_acc = serbian_cyrillic_accuracy(trans_pred, item["transcription"])
            sr_to_lat_acc = serbian_latin_accuracy(trans_pred, item["transcription"])

            results.append({"trans": trans_pred, "gold": item["transcription"],
            "roman_acc": roman_acc, "cyrl_acc": cyrl_acc, "zh_to_trad_acc": zh_to_trad_acc, "zh_to_sim_acc": zh_to_sim_acc,
            "sr_to_cyl_acc": sr_to_cyl_acc, "sr_to_lat_acc": sr_to_lat_acc})
    return pd.DataFrame(results)

_SEARCH_SIGMAS = [
    "bias-sigma-0.1",
    "bias-sigma-0.2",
    "bias-sigma-0.3",
    "bias-sigma-0.4",
    "bias-sigma-0.5",
]

_METRIC_FNS = [
    ("roman", romanization_accuracy),
    ("cyrl", cyrillization_accuracy),
    ("zh_to_trad", chinese_traditional_accuracy),
    ("zh_to_sim", chinese_simplified_accuracy),
    ("sr_to_cyl", serbian_cyrillic_accuracy),
    ("sr_to_lat", serbian_latin_accuracy),
]

@torch.no_grad()
def find_best_sigma(model, steer_weights, src_lang, tgt_lang, tgt_prompt, tgt_script, pool_config, backend_config, layers_to_steer, cache_dir, enc_steer_weights=None, search_sigmas=None):
    if tgt_prompt == "":
        tgt_prompt = None
    adapter = get_adapter(backend_config)
    ds = load_fleurs_split(src_lang, "validation", cache_dir)
    results = []
    metric_fns = list(_METRIC_FNS)
    if src_lang == "ja_jp":
        metric_fns = [
            (name, japanese_romanization_accuracy if name == "roman" else cyrillization_accuracy_japanese if name == "cyrl" else fn)
            for name, fn in metric_fns
        ]
    with tqdm(ds) as pbar:
        for item in pbar:
            audio = fleurs_audio_path(item, src_lang, "validation", cache_dir)
            row = {"gold": item["transcription"]}
            sigmas_to_search = search_sigmas if search_sigmas is not None else _SEARCH_SIGMAS
            for i, sigma_key in enumerate(sigmas_to_search, 1):
                dec_ctx = (
                    LayerwiseSteer(model, steer_weights, STEER_CONFIGS[sigma_key], pool_config, backend_config, layers_to_steer, target="decoder")
                    if steer_weights is not None else nullcontext()
                )
                enc_ctx = (
                    LayerwiseSteer(model, enc_steer_weights, STEER_CONFIGS[sigma_key], pool_config, backend_config, range(len(enc_steer_weights)), target="encoder")
                    if enc_steer_weights is not None else nullcontext()
                )
                with dec_ctx, enc_ctx:
                    pred = adapter.transcribe(
                        model=model,
                        audio_path=audio,
                        language=_fleurs_lang_to_whisper_lang(src_lang),
                        task="transcribe",
                        initial_prompt=tgt_prompt,
                    )["text"].strip()
                row[f"trans_pred_{i}"] = pred
                for metric_name, metric_fn in metric_fns:
                    row[f"{metric_name}_{i}_acc"] = metric_fn(pred, item["transcription"])
            results.append(row)
    return pd.DataFrame(results)
