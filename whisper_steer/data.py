"""FLEURS dataset helpers."""

from pathlib import Path


def load_fleurs_split(language, split, cache_dir=None):
    """Load FLEURS while retaining its original downloaded audio bytes."""
    from datasets import Audio, load_dataset

    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    dataset = load_dataset("google/fleurs", language, split=split, **cache_kwargs)
    return dataset.cast_column("audio", Audio(decode=False))


def fleurs_audio_path(item, language, split, cache_dir=None) -> str:
    """Materialize a FLEURS record's downloaded WAV bytes and return its path."""
    from datasets.config import HF_DATASETS_CACHE

    recorded_path = str(item["path"])
    try:
        if Path(recorded_path).is_file():
            return recorded_path
    except OSError:
        # Hosted records can contain an inaccessible path from the machine on
        # which the dataset was prepared. The audio bytes remain authoritative.
        pass

    audio = item["audio"]
    payload = audio.get("bytes") if isinstance(audio, dict) else None
    if payload is None:
        raise FileNotFoundError(
            f"FLEURS record {item.get('id', recorded_path)!r} has neither an "
            "accessible path nor embedded audio bytes. Clear its Hugging Face "
            "dataset cache entry and download it again."
        )

    effective_cache = Path(cache_dir or HF_DATASETS_CACHE).expanduser()
    filename = Path(recorded_path).name
    if not filename.lower().endswith(".wav"):
        filename = f"{item['id']}.wav"
    destination = effective_cache / "fleurs_audio" / language / split / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw = bytes(payload)
    if not destination.is_file() or destination.stat().st_size != len(raw):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
    return str(destination)
