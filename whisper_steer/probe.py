"""Mass-mean script probe used for Figure 6."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .backend import get_adapter
from .config import DEVICE, EDIT_DISTANCE_THRESHOLD
from .data import fleurs_audio_path, load_fleurs_split
from .hooks import MLPTimeAverager
from .io import load_weights_npz
from .metrics import (
    chinese_simplified_accuracy,
    chinese_traditional_accuracy,
    looks_like_script,
    serbian_cyrillic_accuracy,
    serbian_latin_accuracy,
)
from .transcribe import _fleurs_lang_to_whisper_lang

PROBE_TASKS = {
    "sr_rs": (
        (serbian_cyrillic_accuracy, "Ово је српска реченица"),
        (serbian_latin_accuracy, "Ovo je srpska rečenica"),
    ),
    "cmn_hans_cn": (
        (chinese_simplified_accuracy, "这是一句普通话"),
        (chinese_traditional_accuracy, "這是一句普通話"),
    ),
}


def collect_examples(dataset, adapter, model, language, split, cache_dir, max_per_script=50):
    threshold = 1.0 - EDIT_DISTANCE_THRESHOLD["probe"]
    activations, labels = [], []
    for label, (metric, prompt) in enumerate(PROBE_TASKS[language]):
        accepted = 0
        for item in dataset:
            if accepted >= max_per_script:
                break
            audio = fleurs_audio_path(item, language, split, cache_dir)
            with MLPTimeAverager(model, "mean", "whisper") as capture:
                text = adapter.transcribe(
                    model,
                    audio,
                    language=_fleurs_lang_to_whisper_lang(language),
                    task="transcribe",
                    initial_prompt=prompt,
                )["text"].strip()
            if looks_like_script(text, item["transcription"], metric, threshold=threshold):
                activations.append(capture.means())
                labels.append(label)
                accepted += 1
        if accepted < max_per_script:
            raise RuntimeError(
                f"Could only collect {accepted} examples for label {label} in {language}; "
                f"expected {max_per_script}."
            )
    return np.stack(activations), np.asarray(labels)


def probe_accuracy(activations, labels, vector, train_mean):
    centered = torch.as_tensor(activations - train_mean, dtype=torch.float32)
    direction = torch.as_tensor(np.asarray(vector), dtype=torch.float32)
    predictions = (torch.sigmoid((centered * direction).sum(dim=-1)) >= 0.5).numpy()
    return (predictions == labels[:, None]).mean(axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the paper's layer-wise script probe.")
    parser.add_argument("--language", choices=PROBE_TASKS, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--model", default="large-v2", choices=["large-v2"])
    parser.add_argument("--examples-per-script", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    if args.cache_dir is None:
        from datasets.config import HF_DATASETS_CACHE

        print(f"[data] FLEURS cache: {HF_DATASETS_CACHE} (Hugging Face default)")
    else:
        print(f"[data] FLEURS cache: {args.cache_dir} (explicit)")

    adapter = get_adapter("whisper")
    model = adapter.load_model(args.model, DEVICE)
    vector = load_weights_npz(args.vector)
    train = load_fleurs_split(args.language, "train", args.cache_dir)
    test = load_fleurs_split(args.language, "test", args.cache_dir)
    train_acts, train_labels = collect_examples(
        train, adapter, model, args.language, "train", args.cache_dir, args.examples_per_script
    )
    test_acts, test_labels = collect_examples(
        test, adapter, model, args.language, "test", args.cache_dir, args.examples_per_script
    )
    accuracy = probe_accuracy(test_acts, test_labels, vector, train_acts.mean(axis=0))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"layer": np.arange(len(accuracy)), "accuracy": accuracy}).to_csv(
        args.output_csv, index=False
    )


if __name__ == "__main__":
    main()
