import os
import numpy as np

def save_weights_npz(dec_weights, out_path, enc_weights=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    d = {}
    if dec_weights is not None:
        d.update({f"W{i}": W for i, W in enumerate(dec_weights)})
    if enc_weights is not None:
        d.update({f"EW{i}": W for i, W in enumerate(enc_weights)})
    np.savez_compressed(out_path, **d)
    parts = []
    if dec_weights is not None:
        parts.append(f"{len(dec_weights)} decoder layer maps")
    if enc_weights is not None:
        parts.append(f"{len(enc_weights)} encoder layer maps")
    print(f"[save] wrote {' + '.join(parts)} to: {out_path}")

def load_weights_npz(path):
    data = np.load(path)
    keys = sorted(
        [k for k in data.files if k.startswith("W") and not k.startswith("EW")],
        key=lambda k: int(k[1:]),
    )
    W_by_layer = [data[k] for k in keys]
    print(f"[load] loaded {len(W_by_layer)} decoder layer maps from: {path}")
    return W_by_layer

def load_enc_weights_npz(path):
    data = np.load(path)
    keys = sorted(
        [k for k in data.files if k.startswith("EW")],
        key=lambda k: int(k[2:]),
    )
    if not keys:
        return None
    enc_weights = [data[k] for k in keys]
    print(f"[load] loaded {len(enc_weights)} encoder layer maps from: {path}")
    return enc_weights