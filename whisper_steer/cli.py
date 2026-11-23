import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from . import config
from .learn import train_from_fleurs, train_from_fleurs_with_steering
from .io import save_weights_npz, load_weights_npz, load_enc_weights_npz
from .transcribe import evaluate, find_best_sigma
from .backend import get_adapter
from .config import DEVICE
#from .baselines import BASELINE_REGISTRY


def set_seed(seed: int):
    # Python built-ins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

    # Ensure deterministic behavior (may reduce performance a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_args(argv):
    # Shared arguments
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model", type=str, required=True)
    parent.add_argument("--src-lang", type=str, nargs="+", required=True)
    parent.add_argument("--tgt-lang", type=str, nargs="+", required=True)
    parent.add_argument("--src-prompt", type=str, nargs="+", default="")
    parent.add_argument("--trg-prompt", type=str, nargs="+", default="")
    parent.add_argument("--tgt-script", type=str, help="In cases like Chinese where multiple scripts exist, specify the target script: Hans (simplified) or Hant (traditional)")
    parent.add_argument("--seed", type=int, default=42)
    parent.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Writable cache directory for FLEURS. By default, Hugging Face's "
            "standard cache and environment variables are used."
        ),
    )
    parent.add_argument("-v", "--vector", type=Path, required=True)
    parent.add_argument(
        "-s", "--steer-config",
        choices=list(config.STEER_CONFIGS.keys()),
        required=True,
        help="Vector config name (from config.STEER_CONFIGS)"
    )
    parent.add_argument(
        "-p", "--pool-config",
        choices=list(config.POOL_CONFIGS),
        required=True,
        help="Pool config name (from config.POOL_CONFIGS)"
    )

    parent.add_argument(
        "-b", "--backend-config",
        choices=list(config.BACKEND_CONFIGS),
        required=True,
        help="Backend config name (from config.BACKEND_CONFIGS)"
    )

    # Main parser
    p = argparse.ArgumentParser(description="Whisper transliteration steering.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_learn = sub.add_parser("train", parents=[parent], help="Learn per-layer steering vectors from a FLEURS split")
    p_learn.add_argument("--max-files", type=int, default=config.DEFAULT_TRAIN_EXAMPLES)
    p_learn.add_argument("--steer", choices=["decoder", "encoder", "both"], default="decoder", help="Which part of the model to train steering vectors for (default: decoder)")

    # second pass train
    p_learn = sub.add_parser("second-pass-train", parents=[parent], help="Learns on top of steered model outputs")
    p_learn.add_argument("--max-files", type=int, default=config.DEFAULT_TRAIN_EXAMPLES)

    # find-sigma
    p_infer = sub.add_parser("search-sigma", parents=[parent], help="Grid search sigmas")
    p_infer.add_argument("--layers-to-steer", nargs="+", type=int, default=list(range(32)), help="Layers to steer (default: all layers)")
    p_infer.add_argument("-o", "--output-csv", type=Path, required=True)
    p_infer.add_argument("--steer", choices=["decoder", "encoder", "both"], default="decoder", help="Which part of the model to steer (default: decoder)")
    p_infer.add_argument("--sigmas", nargs="+", type=float, default=None, help="Sigma values to search over (default: 0.1 0.2 0.3 0.4 0.5)")
    p_infer.add_argument(
        "--selection",
        choices=["mean", "max"],
        default="mean",
        help="Validation criterion: mean for script confusion, max for transliteration.",
    )

    # inference
    p_infer = sub.add_parser("infer", parents=[parent], help="Inference and evaluate with steering vectors")
    p_infer.add_argument("--layers-to-steer", nargs="+", type=int, default=list(range(32)), help="Layers to steer (default: all layers)")
    p_infer.add_argument("-o", "--output-csv", type=Path, required=True)
    p_infer.add_argument("--steer", choices=["decoder", "encoder", "both"], default="decoder", help="Which part of the model to steer (default: decoder)")
    

    """
    # cross-model train
    p_cross_model_learn = sub.add_parser("cross-model-train", parents=[parent], help="Learn cross model mapping")
    p_cross_model_learn.add_argument("--source-model", type=str, required=True)
    p_cross_model_learn.add_argument("--source-model-backend", type=str, required=True)
    p_cross_model_learn.add_argument("--max-files", type=int, default=100)
    """
    # args
    args = p.parse_args(argv)
    args.steer_config = config.STEER_CONFIGS[args.steer_config]

    return args

def main(argv=None):
    args = get_args(argv)

    if args.cache_dir is None:
        from datasets.config import HF_DATASETS_CACHE

        print(f"[data] FLEURS cache: {HF_DATASETS_CACHE} (Hugging Face default)")
    else:
        print(f"[data] FLEURS cache: {args.cache_dir} (explicit)")

    set_seed(args.seed)
    adapter = get_adapter(args.backend_config)
    model = adapter.load_model(model_name=args.model, device=DEVICE)

    if args.cmd == "train":
        dec_weights, dec_reverse, enc_weights, enc_reverse = train_from_fleurs(
            model=model,
            method=args.steer_config["train"],
            pool_config=args.pool_config,
            backend_config=args.backend_config,
            src_lang=list(args.src_lang),
            tgt_lang=list(args.tgt_lang),
            src_prompt=list(args.src_prompt),
            trg_prompt=list(args.trg_prompt),
            max_files=args.max_files,
            seed=args.seed,
            cache_dir=args.cache_dir,
            steer=args.steer,
        )
        save_weights_npz(dec_weights, args.vector, enc_weights=enc_weights)
        reverse_vector_save_path = args.vector.with_stem("reverse_" + args.vector.stem)
        save_weights_npz(dec_reverse, reverse_vector_save_path, enc_weights=enc_reverse)

    if args.cmd == "second-pass-train":
        weights = load_weights_npz(args.vector)
        new_weights = train_from_fleurs_with_steering(
            model=model,
            vector_name=args.vector,
            steer_weights=weights,
            method=args.steer_config["train"],
            steer_config=args.steer_config,
            pool_config=args.pool_config,
            backend_config=args.backend_config,
            src_lang=list(args.src_lang),
            tgt_lang=list(args.tgt_lang),
            src_prompt=list(args.src_prompt),
            trg_prompt=list(args.trg_prompt),
            max_files=args.max_files,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
        vector_save_path = args.vector.with_stem(list(args.src_lang)[0] +"_ft_" + args.vector.stem)
        save_weights_npz(new_weights, vector_save_path)

    elif args.cmd == "search-sigma":
        weights = load_weights_npz(args.vector) if args.steer in ("decoder", "both") else None
        enc_weights = load_enc_weights_npz(args.vector) if args.steer in ("encoder", "both") else None
        search_sigmas = (
            [f"bias-sigma-{s}" for s in args.sigmas]
            if args.sigmas is not None else None
        )
        df = find_best_sigma(
            model=model,
            steer_weights=weights,
            src_lang=list(args.src_lang)[0],
            tgt_lang=list(args.tgt_lang)[0],
            tgt_prompt=list(args.trg_prompt)[0],
            tgt_script=args.tgt_script,
            pool_config=args.pool_config,
            backend_config=args.backend_config,
            layers_to_steer=list(args.layers_to_steer),
            cache_dir=args.cache_dir,
            enc_steer_weights=enc_weights,
            search_sigmas=search_sigmas,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, escapechar='\\')
        # print results
        # best sigma for all categories
        acc_cols = [col for col in df.columns if col.endswith("_acc")]

        # group columns by category prefix (e.g. cyrl, roman, zh_to_trad)
        from collections import defaultdict

        category_cols = defaultdict(list)
        for col in acc_cols:
            # assumes pattern like "<category>_<n>_acc"
            category = col.rsplit("_", 2)[0]
            category_cols[category].append(col)

        print("\nBest column per category:")
        for category, cols in sorted(category_cols.items()):
            scores = df[cols].mean() if args.selection == "mean" else df[cols].max()
            best_col = scores.idxmax()
            print(f"{category}: {best_col}")

    elif args.cmd == "infer":
        weights = load_weights_npz(args.vector) if args.steer in ("decoder", "both") else None
        enc_weights = load_enc_weights_npz(args.vector) if args.steer in ("encoder", "both") else None
        df = evaluate(
            model=model,
            steer_weights=weights,
            src_lang=list(args.src_lang)[0],
            tgt_lang=list(args.tgt_lang)[0],
            tgt_prompt=list(args.trg_prompt)[0],
            tgt_script=args.tgt_script,
            steer_config=args.steer_config,
            pool_config=args.pool_config,
            backend_config=args.backend_config,
            layers_to_steer=list(args.layers_to_steer),
            cache_dir=args.cache_dir,
            enc_steer_weights=enc_weights,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, escapechar='\\')
        # print results
        acc_cols = [col for col in df.columns if col.endswith('_acc')]
        acc_means = df[acc_cols].mean()
        for col, mean_val in acc_means.items():
            print(f"{col}: {mean_val:.4f}")
    
    """
    elif args.cmd == "cross-model-train":
        source_model_adapter = get_adapter(args.source_model_backend)
        model_x = source_model_adapter.load_model(model_name=args.source_model, device=DEVICE)
        weights = train_cross_model(
            model_x=model_x,
            model_y=model,
            pool_config=args.pool_config,
            source_model_backend_config=args.source_model_backend,
            target_model_backend_config=args.backend_config,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            src_prompt=args.src_prompt,
            trg_prompt=args.trg_prompt,
            max_files=args.max_files,
            seed=args.seed,
        )
        save_weights_npz(weights, args.vector)
    """

if __name__ == "__main__":
    main()
