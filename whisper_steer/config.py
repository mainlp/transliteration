"""Configuration for the experiments reported in Shim et al. (2026)."""

import torch

from .metrics import (
    chinese_simplified_accuracy,
    chinese_traditional_accuracy,
    serbian_cyrillic_accuracy,
    serbian_latin_accuracy,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POOL_CONFIGS = ["mean"]
BACKEND_CONFIGS = ["whisper", "owsm-ctc"]

# Section 3.4: ten accepted training pairs. One-shot runs override this with 1.
DEFAULT_TRAIN_EXAMPLES = 10

# Equation 4 / Section 3.4. These are edit-distance thresholds; the code's
# looks_like_script helper consumes the corresponding similarity threshold.
EDIT_DISTANCE_THRESHOLD = {
    "script_confusion": 0.4,
    "romanization": 0.4,
    "cyrillization": 0.8,
    "probe": 0.1,
}

# Section 3.4: validation grid for decoder, encoder, and OWSM-CTC tuning.
SIGMA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
ENCODER_FIXED_SIGMA = 1.0

SCRIPT_FILTERS = {
    "sr_rs": (serbian_cyrillic_accuracy, serbian_latin_accuracy),
    "cmn_hans_cn": (chinese_simplified_accuracy, chinese_traditional_accuracy),
}

STEER_CONFIGS = {
    f"bias-sigma-{sigma:.1f}": {
        "train": "bias",
        "infer": "bias-sigma",
        "sigma": sigma,
    }
    for sigma in (0.0, *SIGMA_GRID, ENCODER_FIXED_SIGMA)
}
