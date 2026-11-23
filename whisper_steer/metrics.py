import re
import regex

import pandas as pd

from functools import lru_cache

import cyrtranslit
import pykakasi

import editdistance
from GlotScript import sp
import uroman as ur
from icu import Transliterator

import opencc

@lru_cache(maxsize=1)
def _get_opencc_t2s() -> opencc.OpenCC:
    """Get an OpenCC traditional to simplified converter (cached)."""
    return opencc.OpenCC('t2s.json')

@lru_cache(maxsize=1)
def _get_opencc_s2t() -> opencc.OpenCC:
    """Get an OpenCC simplified to traditional converter (cached)."""
    return opencc.OpenCC('s2t.json')

def to_simplified_chinese(text: str) -> str:
    """Convert Traditional/mixed Chinese text to Simplified Chinese.
    If some substring is already Simplified Chinese, it is returned unchanged.
    """
    converter = _get_opencc_t2s()
    return converter.convert(text)

def to_traditional_chinese(text: str) -> str:
    """Convert Simplified/mixed Chinese text to Traditional Chinese.
    If some substring is already Traditional Chinese, it is returned unchanged.
    """
    converter = _get_opencc_s2t()
    return converter.convert(text)

uroman = ur.Uroman() 
kks = pykakasi.kakasi()
lat_to_cyr = Transliterator.createInstance("Latin-Cyrillic")

def normalize_roman(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)  # remove punctuation/spaces
    return s

"""
ja.apply(lambda row: japanese_romanization_accuracy(row["trans"], row["gold"]),axis=1)
"""

def japanese_romanization_accuracy(pred: str, gt: str) -> float:
    pred = str(pred)
    gt = str(gt)
    romanized_gt = " ".join([i['hepburn'] for i in kks.convert(gt)])
    pred = normalize_roman(pred)
    romanized_gt = normalize_roman(romanized_gt)
    return 1.0 - (editdistance.eval(pred, romanized_gt) / max(len(pred), len(romanized_gt)))

def romanization_accuracy(pred: str, gt: str) -> str:
    romanized_gt = uroman.romanize_string(gt)
    pred = normalize_roman(pred)
    romanized_gt = normalize_roman(romanized_gt)
    return 1.0 - (editdistance.eval(pred, romanized_gt) / max(len(pred), len(romanized_gt)))

def normalize_cyrillic(s):
    s = s.lower()
    s = regex.sub(r"[^\p{IsCyrillic}0-9]+", "", s)
    return s

def cyrillization_accuracy(pred: str, gt: str) -> str:
    romanized_gt = uroman.romanize_string(gt)
    cyrilicized_gt = lat_to_cyr.transliterate(romanized_gt)
    pred = normalize_cyrillic(pred)
    cyrilicized_gt = normalize_cyrillic(cyrilicized_gt)
    return 1.0 - (editdistance.eval(pred, cyrilicized_gt) / max(len(pred), len(cyrilicized_gt)))

def cyrillization_accuracy_japanese(pred: str, gt: str) -> float:
    pred = str(pred)
    gt = str(gt)
    romanized_gt = " ".join([i['hepburn'] for i in kks.convert(gt)])
    cyrilicized_gt = lat_to_cyr.transliterate(romanized_gt)
    cyrilicized_gt = normalize_cyrillic(cyrilicized_gt)
    pred = normalize_cyrillic(pred)
    return 1.0 - (editdistance.eval(pred, cyrilicized_gt) / max(len(pred), len(cyrilicized_gt)))

def script_converted(pred: str, src_script: str, trg_script: str):
    """
    Returns whether proportion of target script is higher than src script.
    src_script and trg_script should be from supported codes in GlotScript, where e.g. 'Latn' is Latin and 'Cyrl' is Cyrillic.
    """
    script_composition = sp(pred)[2]['details']
    try:
        src_script_proportion = script_composition[src_script]
    except KeyError:
        src_script_proportion = 0.0
    try:
        trg_script_proportion = script_composition[trg_script]
    except KeyError:
        trg_script_proportion = 0.0
    return trg_script_proportion > src_script_proportion

def normalize_chinese(s: str) -> str:
    return regex.sub(r"[\p{P}\p{S}\s]+", "", s)

def chinese_traditional_accuracy(pred: str, ground_truth: str):
    #d = Distance()
    trad_ground_truth = to_traditional_chinese(ground_truth)

    pred = normalize_chinese(pred)
    trad_ground_truth = normalize_chinese(trad_ground_truth)

    return 1.0 - (
        editdistance.eval(pred, trad_ground_truth)
        / max(len(pred), len(trad_ground_truth))
    )


def chinese_simplified_accuracy(pred: str, ground_truth: str):
    #d = Distance()
    sim_ground_truth = to_simplified_chinese(ground_truth)

    pred = normalize_chinese(pred)
    sim_ground_truth = normalize_chinese(sim_ground_truth)

    return 1.0 - (
        editdistance.eval(pred, sim_ground_truth)
        / max(len(pred), len(sim_ground_truth))
    )

def normalize_serbian(text: str) -> str:
    """Normalize Serbian text by lowercasing, stripping, and removing punctuation."""
    text = text.lower().strip()
    text = regex.sub(r"[\p{P}\p{S}]+", "", text)  # removed \s
    return text


def serbian_cyrillic_accuracy(pred: str, ground_truth: str):
    """
    Compute accuracy when the ground truth is converted to Cyrillic.
    """
    #d = Distance()
    pred_norm = normalize_serbian(pred)
    gt_cyr = normalize_serbian(cyrtranslit.to_cyrillic(ground_truth, "sr"))

    return 1.0 - (
        editdistance.eval(pred_norm, gt_cyr)
        / max(len(pred_norm), len(gt_cyr))
    )


def serbian_latin_accuracy(pred: str, ground_truth: str):
    """
    Compute accuracy when the ground truth is converted to Latin.
    """
    #d = Distance()
    pred_norm = normalize_serbian(pred)
    gt_lat = normalize_serbian(cyrtranslit.to_latin(ground_truth, "sr"))

    return 1.0 - (
        editdistance.eval(pred_norm, gt_lat)
        / max(len(pred_norm), len(gt_lat))
    )

def looks_like_script(pred, gt, script_metric, threshold): # 0.6 for most; 0.2 for cyrl
    return script_metric(pred, gt) > threshold

def print_results(results_csv):
    df = pd.read_csv(results_csv)
    acc_cols = [col for col in df.columns if col.endswith('_acc')]
    acc_means = df[acc_cols].mean()
    for col, mean_val in acc_means.items():
        print(f"{col}: {mean_val:.4f}")

