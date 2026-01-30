from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..canon import get_canon
from ..model.forward import run_with_u_capture
from ..model.hooks import MlpLayer


class RasterExtractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedRasters:
    a: np.ndarray  # uint16 [N, L, T]
    x: np.ndarray  # uint8  [N, L, T]
    achieved_rate_by_layer: np.ndarray  # float64 [L]


def _v_from_z(z: torch.Tensor, spike_def_id: str) -> torch.Tensor:
    canon = get_canon()
    one = str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_ONE_SIDED_POS"])
    two = str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_TWO_SIDED_ABS"])
    if spike_def_id == one:
        return z
    if spike_def_id == two:
        return torch.abs(z)
    raise RasterExtractError(f"Unknown spike_def_id: {spike_def_id}")


def extract_rasters(
    *,
    model: torch.nn.Module,
    layers: list[MlpLayer],
    token_windows: list[list[int]],
    gain: float,
    gain_target: str = "MLP",
    gain_by_layer: dict[int, float] | None = None,
    default_gain: float = 1.0,
    spike_def_id: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    tau_by_layer: np.ndarray,
    device: str,
    seq_len: int,
) -> ExtractedRasters:
    canon = get_canon()
    eps = float(canon["CONST"]["EPS"])

    if tau_by_layer.shape[0] != len(layers):
        raise RasterExtractError("tau_by_layer must have shape [L]")

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    denom_t = torch.tensor(sigma, dtype=torch.float32, device=device) + eps
    tau_t = torch.tensor(tau_by_layer, dtype=torch.float32, device=device)

    L = len(layers)
    N = len(token_windows)
    a = np.zeros((N, L, seq_len), dtype=np.uint16)
    spikes_sum = np.zeros((L,), dtype=np.float64)
    denom_units = np.zeros((L,), dtype=np.float64)

    cur = {"i": 0}

    def on_u(layer_idx: int, u: torch.Tensor) -> None:
        if u.shape[0] != 1:
            raise RasterExtractError("Batch size > 1 is not supported for raster extraction")
        z = (u.to(dtype=torch.float32) - mu_t[layer_idx]) / denom_t[layer_idx]
        v = _v_from_z(z, spike_def_id)
        spikes = v > tau_t[layer_idx]
        spikes_sum[layer_idx] += float(spikes.sum().detach().cpu().item())
        denom_units[layer_idx] += float(v.numel())
        a_tok = spikes.sum(dim=-1).to(dtype=torch.int32).detach().cpu().numpy().reshape(-1)
        a[cur["i"], layer_idx, :] = a_tok.astype(np.uint16)

    for ids in token_windows:
        if len(ids) != seq_len:
            raise RasterExtractError("Token window length mismatch")
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        run_with_u_capture(
            model=model,
            layers=layers,
            input_ids=input_ids,
            attention_mask=attn,
            gain=gain,
            gain_target=gain_target,
            gain_by_layer=gain_by_layer,
            default_gain=default_gain,
            on_u=on_u,
            need_logits=False,
        )
        cur["i"] += 1

    x = (a > 0).astype(np.uint8)
    achieved = spikes_sum / np.maximum(denom_units, 1.0)
    return ExtractedRasters(a=a, x=x, achieved_rate_by_layer=achieved)
