import contextlib
import numpy as np
import torch
from .backend import get_adapter
from .config import DEVICE

class MLPTimeAverager(contextlib.AbstractContextManager):
    def __init__(self, model, pool_config, backend_config, which="decoder"):
        """
        model: whisper model
        pool_config: one of config.POOL_CONFIGS
        backend_config: one of config.BACKEND_CONFIGS
        which: "decoder" or "encoder" — which blocks to hook into
        """
        self.model = model
        self.adapter = get_adapter(backend_config)

        self.hooks = []
        if which == "decoder":
            self.blocks = self.adapter.decoder_blocks(self.model)
        elif which == "encoder":
            self.blocks = self.adapter.encoder_blocks(self.model)
        else:
            raise ValueError(f"Unknown which: {which!r}")
        D = self.adapter.hidden_size(self.model)
        self.sums = [torch.zeros(D, device=DEVICE) for _ in self.blocks]
        self.counts = [0 for _ in self.blocks]
        self.pool_config = pool_config

    def _pool(self, feats):
        if self.pool_config == "mean":
            return feats.mean(dim=(0, 1))
        else:
            raise ValueError(f"Unknown pool type: {self.pool_config}")

    def __enter__(self):
        for li, blk in enumerate(self.blocks):
            def _make_hook(layer_idx):
                def hook(module, inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output # in case is owsm, which outputs tuple
                    if torch.is_tensor(tensor):
                        o = tensor.detach()
                        m = self._pool(o)
                        self.sums[layer_idx] += m
                        self.counts[layer_idx] += 1
                    return output
                return hook
            h = blk.register_forward_hook(_make_hook(li))
            self.hooks.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        return False

    @torch.no_grad()
    def means(self):
        if not self.sums:
            return np.zeros((0,), dtype=np.float32)
        outs = []
        for s, c in zip(self.sums, self.counts):
            assert c > 0
            outs.append(s / float(c))
        return torch.stack(outs, dim=0).detach().cpu().numpy()


class LayerwiseSteer(contextlib.AbstractContextManager):
    def __init__(self, model, weights, steer_config, pool_config, backend_config, layers_to_steer, target="decoder"):
        """
        model: whisper model
        weights: list of parameters (W or v) for each layer
        steer_config: one of config.STEER_CONFIGS
        pool_config: one of config.POOL_CONFIGS
        backend_config: one of config.BACKEND_CONFIG
        layers_to_steer: which layers to steer
        target: "decoder" (steer last token only) or "encoder" (steer all positions)
        """
        self.model = model
        self.adapter = get_adapter(backend_config)

        self.steer_config = steer_config
        self.pool_config = pool_config
        self.target = target

        self.layers_to_steer = layers_to_steer

        if target == "decoder":
            self.blocks = self.adapter.decoder_blocks(self.model)
        elif target == "encoder":
            self.blocks = self.adapter.encoder_blocks(self.model)
        else:
            raise ValueError(f"Unknown target: {target!r}")
        self.params = [torch.tensor(p, device=DEVICE) for p in weights]
        self.hooks = []

        if self.pool_config != "mean":
            raise ValueError("Not implemented for pool_config other than 'mean'")
        if len(self.params) != len(self.blocks):
            raise ValueError(f"weights length must match number of {target} layers.")

    def __enter__(self):
        for li, blk in enumerate(self.blocks):
            if li in self.layers_to_steer:
                def _make_hook(param, transform_fn, target):
                    def hook(module, inputs, output):
                        if isinstance(output, tuple):
                            y, *rest = output
                        else:
                            y, rest = output, []
                        h = y.detach().clone()

                        if target == "encoder":
                            # steer all positions
                            y_steered = transform_fn(h, param)
                        else:
                            # decoder: steer last token only
                            last = h[:, -1, :]
                            steered_last = transform_fn(last, param)
                            if h.size(1) == 1:
                                y_steered = steered_last.unsqueeze(1)
                            else:
                                h[:, -1, :] = steered_last
                                y_steered = h

                        if isinstance(output, tuple):
                            return (y_steered, *rest)
                        else:
                            return y_steered
                    return hook

                h = blk.register_forward_hook(_make_hook(self.params[li], self._transform, self.target))
                self.hooks.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        return False

    def _transform(self, last, param):
        """
        last: hidden state of the last time step
        param: the steering vector [D] or matrix [D, D]
        """
        method = self.steer_config["infer"]
        sigma = self.steer_config.get("sigma", 1.0)
        param = param.to(dtype=last.dtype)

        if method == "bias-sigma":
            return last + (sigma * param)
        else:
            raise ValueError(f"Unknown inference method: {method}")
