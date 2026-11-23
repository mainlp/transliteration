import numpy as np
import torch
import random
from contextlib import nullcontext
from tqdm import tqdm
from .transcribe import collect_means_and_text
import whisper_steer.config as config
from .data import fleurs_audio_path, load_fleurs_split
from .metrics import (
    cyrillization_accuracy,
    cyrillization_accuracy_japanese,
    japanese_romanization_accuracy,
    looks_like_script,
    romanization_accuracy,
)
from .hooks import LayerwiseSteer


def _dedupe_fleurs(dset):
    """Return dataset with only first occurrence of each id."""
    ids = dset["id"]
    seen = set()
    keep = []
    for i, _id in enumerate(ids):
        if _id not in seen:
            seen.add(_id)
            keep.append(i)
    return dset.select(keep)


def load_fleurs(langs, max_files, seed, cache_dir, split="train", parallel=False):
    dsets = {
        lang: load_fleurs_split(lang, split, cache_dir)
        for lang in langs
    }
    dsets = {lang: _dedupe_fleurs(ds) for lang, ds in dsets.items()}

    if parallel:
        id_sets = [set(ds["id"]) for ds in dsets.values()]
        shared_ids = set.intersection(*id_sets)

        fleurs_data = {
            lang: ds.filter(lambda x: x["id"] in shared_ids)
            for lang, ds in dsets.items()
        }
        shared_ids = sorted(list(shared_ids))
        rng = random.Random(seed)
        rng.shuffle(shared_ids)
        sampled_ids = set(shared_ids[:max_files])

        fleurs_data = {
            lang: ds.filter(lambda x: x["id"] in sampled_ids)
            for lang, ds in fleurs_data.items()
        }
        fleurs_data = {
            lang: ds.sort("id")
            for lang, ds in fleurs_data.items()
        }
    else:
        fleurs_data = {lang: ds.shuffle(seed=seed) for lang, ds in dsets.items()}
    return fleurs_data


@torch.no_grad()
def gather_encoder_means(
    model,
    lang,
    prompt,
    pool_config,
    backend_config,
    max_files,
    seed,
    cache_dir,
    script_filter=None,
    filter_threshold=1.0 - config.EDIT_DISTANCE_THRESHOLD["script_confusion"],
):
    """Collect encoder activation means from audio where the decoder output passes script_filter.

    This is intentionally UNPAIRED: the src and tgt sets are collected separately with
    different prompts and filters, so different audio samples end up in each set.
    This is necessary because the encoder is prompt-independent — the same audio always
    produces the same encoder representations regardless of prompt.
    """
    ds_dict = load_fleurs([lang], max_files * 4, seed, cache_dir, split="train", parallel=False)
    ds = ds_dict[lang]

    means = []
    accepted = 0

    with tqdm(total=max_files) as pbar:
        for item in ds:
            if accepted >= max_files:
                break

            audio = fleurs_audio_path(item, lang, "train", cache_dir)
            mean_enc, pred = collect_means_and_text(
                model, audio, fleurs_lang=lang,
                initial_prompt=prompt,
                pool_config=pool_config, backend_config=backend_config,
                which="encoder",
            )

            if script_filter is not None and not looks_like_script(
                pred, item["transcription"], script_filter, threshold=filter_threshold
            ):
                continue

            means.append(mean_enc)
            accepted += 1
            pbar.set_postfix_str(f"pred: {pred[:20]}...")
            pbar.update(1)

    return np.stack(means, 0)


@torch.no_grad()
def gather_means_pairwise(
    model,
    lang,
    src_prompt,
    tgt_prompt,
    pool_config,
    backend_config,
    max_files,
    seed,
    cache_dir,
    script_filter_src=None,
    script_filter_tgt=None,
    filter_threshold=1.0 - config.EDIT_DISTANCE_THRESHOLD["script_confusion"],
    steer_weights=None,
    steer_config=None,
):
    """Collect paired (src, tgt) activation means for the same audio under two prompts.

    The same audio file is run through the model twice — once with src_prompt (to get
    source-script activations) and once with tgt_prompt (to get target-script activations).
    The mean decoder activations from each pass form a matched pair used to learn a
    per-layer steering vector.

    If steer_weights and steer_config are provided, the target pass runs under LayerwiseSteer
    (used for second-pass training with pseudo-labeled data).
    """
    ds_dict = load_fleurs([lang], max_files * 4, seed, cache_dir, split="train", parallel=False)
    ds = ds_dict[lang]

    means_src = []
    means_tgt = []
    accepted = 0

    with tqdm(total=max_files) as pbar:
        for item in ds:
            if accepted >= max_files:
                break

            audio = fleurs_audio_path(item, lang, "train", cache_dir)

            mean_src, pred_src = collect_means_and_text(
                model, audio, fleurs_lang=lang,
                initial_prompt=src_prompt,
                pool_config=pool_config, backend_config=backend_config,
            )
            if script_filter_src is not None and not looks_like_script(
                pred_src, item["transcription"], script_filter_src, threshold=filter_threshold
            ):
                continue

            if steer_weights is not None:
                ctx = LayerwiseSteer(
                    model, steer_weights, steer_config, pool_config, backend_config,
                    layers_to_steer=range(0, 32),
                )
            else:
                ctx = nullcontext()

            with ctx:
                mean_tgt, pred_tgt = collect_means_and_text(
                    model, audio, fleurs_lang=lang,
                    initial_prompt=tgt_prompt,
                    pool_config=pool_config, backend_config=backend_config,
                )

            if script_filter_tgt is not None and not looks_like_script(
                pred_tgt, item["transcription"], script_filter_tgt, threshold=filter_threshold
            ):
                continue

            means_src.append(mean_src)
            means_tgt.append(mean_tgt)
            accepted += 1
            pbar.set_postfix_str(f"S:{pred_src[:12]} | T:{pred_tgt[:12]}")
            pbar.update(1)

    return np.stack(means_src, 0), np.stack(means_tgt, 0)


def train(method, means_src, means_tgt):
    L = means_src.shape[1]
    assert means_tgt.shape[1] == L

    if method == "bias":
        return (means_tgt - means_src).mean(axis=0).astype(np.float32)  # [L, D]

    else:
        raise ValueError(f"Unknown training method: {method}")


@torch.no_grad()
def train_from_fleurs(
    model,
    method,
    pool_config,
    backend_config,
    src_lang,
    tgt_lang,
    src_prompt,
    trg_prompt,
    max_files,
    seed,
    cache_dir,
    steer="decoder",
):
    assert len(src_lang) == 1 and len(tgt_lang) == 1 and src_lang[0] == tgt_lang[0]
    lang = src_lang[0]
    src_filter, tgt_filter = config.SCRIPT_FILTERS[lang]

    dec_weights = dec_reverse = None
    if steer in ("decoder", "both"):
        means_src, means_tgt = gather_means_pairwise(
            model, lang,
            src_prompt=src_prompt[0] or None,
            tgt_prompt=trg_prompt[0] or None,
            pool_config=pool_config,
            backend_config=backend_config,
            max_files=max_files,
            seed=seed,
            cache_dir=cache_dir,
            script_filter_src=src_filter,
            script_filter_tgt=tgt_filter,
        )
        dec_weights = train(method, means_src, means_tgt)
        dec_reverse = train(method, means_tgt, means_src)

    enc_weights = enc_reverse = None
    if steer in ("encoder", "both"):
        enc_means_src = gather_encoder_means(
            model, lang, src_prompt[0] or None,
            pool_config, backend_config, max_files, seed,
            cache_dir,
            script_filter=src_filter,
        )
        enc_means_tgt = gather_encoder_means(
            model, lang, trg_prompt[0] or None,
            pool_config, backend_config, max_files, seed,
            cache_dir,
            script_filter=tgt_filter,
        )
        enc_weights = train(method, enc_means_src, enc_means_tgt)
        enc_reverse = train(method, enc_means_tgt, enc_means_src)

    return dec_weights, dec_reverse, enc_weights, enc_reverse


@torch.no_grad()
def train_from_fleurs_with_steering(
    model,
    vector_name,
    steer_weights,
    method,
    steer_config,
    pool_config,
    backend_config,
    src_lang,
    tgt_lang,
    src_prompt,
    trg_prompt,
    max_files,
    seed,
    cache_dir,
):
    assert len(src_lang) == 1 and len(tgt_lang) == 1 and src_lang[0] == tgt_lang[0]
    lang = src_lang[0]

    if "reverse" in vector_name.stem:
        tgt_filter = cyrillization_accuracy_japanese if lang == "ja_jp" else cyrillization_accuracy
        filter_threshold = 1.0 - config.EDIT_DISTANCE_THRESHOLD["cyrillization"]
    else:
        tgt_filter = japanese_romanization_accuracy if lang == "ja_jp" else romanization_accuracy
        filter_threshold = 1.0 - config.EDIT_DISTANCE_THRESHOLD["romanization"]

    means_src, means_tgt = gather_means_pairwise(
        model, lang,
        src_prompt=src_prompt[0] or None,
        tgt_prompt=trg_prompt[0] or None,
        pool_config=pool_config,
        backend_config=backend_config,
        max_files=max_files,
        seed=seed,
        cache_dir=cache_dir,
        script_filter_src=None,
        script_filter_tgt=tgt_filter,
        filter_threshold=filter_threshold,
        steer_weights=steer_weights,
        steer_config=steer_config,
    )
    return train(method, means_src, means_tgt)
