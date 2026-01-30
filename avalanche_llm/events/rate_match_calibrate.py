from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..canon import get_canon
from ..model.forward import run_with_u_capture
from ..model.hooks import MlpLayer


class RateMatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class TauCalibration:
    tau_by_target_rate: dict[str, np.ndarray]  # target_rate_str -> [L] float64
    achieved_rate_by_target_rate: dict[str, np.ndarray]  # target_rate_str -> [L] float64
    v_min: np.ndarray  # [L]
    v_max: np.ndarray  # [L]


@dataclass(frozen=True)
class RasterBundle:
    a_by_target_rate: dict[str, np.ndarray]  # [N, L, T] uint16
    x_by_target_rate: dict[str, np.ndarray]  # [N, L, T] uint8


def _round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def _v_from_z(z: torch.Tensor, spike_def_id: str) -> torch.Tensor:
    canon = get_canon()
    one = str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_ONE_SIDED_POS"])
    two = str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_TWO_SIDED_ABS"])
    if spike_def_id == one:
        return z
    if spike_def_id == two:
        return torch.abs(z)
    raise RateMatchError(f"Unknown spike_def_id: {spike_def_id}")


def calibrate_tau_and_optionally_rasters(
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
    r_targets_by_target_rate: dict[str, np.ndarray],
    device: str,
    seq_len: int,
    collect_rasters: bool,
) -> tuple[TauCalibration, RasterBundle | None]:
    canon = get_canon()
    bins = int(canon["CONST"]["RATE_MATCH_HIST_BINS"])
    edge_step = float(canon["CONST"]["RATE_MATCH_EDGE_ROUND_ABS"])
    tol = float(canon["CONST"]["RATE_MATCH_TOL_ABS"])
    eps = float(canon["CONST"]["EPS"])

    if device not in {"cpu", "cuda"}:
        raise RateMatchError(f"Unsupported device: {device}")

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sig_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    denom_t = sig_t + eps

    L = mu_t.shape[0]
    if mu_t.shape != sig_t.shape:
        raise RateMatchError("mu and sigma shapes mismatch")
    if L != len(layers):
        raise RateMatchError(f"mu/sigma L mismatch: mu has {L} layers, model has {len(layers)}")

    # Pass 1: min/max per layer
    v_min = np.full((L,), np.inf, dtype=np.float64)
    v_max = np.full((L,), -np.inf, dtype=np.float64)

    def on_u_minmax(layer_idx: int, u: torch.Tensor) -> None:
        z = (u.to(dtype=torch.float32) - mu_t[layer_idx]) / denom_t[layer_idx]
        v = _v_from_z(z, spike_def_id)
        mn = float(v.amin().detach().cpu().item())
        mx = float(v.amax().detach().cpu().item())
        if mn < v_min[layer_idx]:
            v_min[layer_idx] = mn
        if mx > v_max[layer_idx]:
            v_max[layer_idx] = mx

    for ids in token_windows:
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
            on_u=on_u_minmax,
            need_logits=False,
        )

    v_min_r = np.array([_round_down(float(x), edge_step) for x in v_min], dtype=np.float64)
    v_max_r = np.array([_round_up(float(x), edge_step) for x in v_max], dtype=np.float64)
    if np.any(~np.isfinite(v_min_r)) or np.any(~np.isfinite(v_max_r)):
        raise RateMatchError("Non-finite v min/max encountered")
    if np.any(v_min_r >= v_max_r):
        raise RateMatchError("Degenerate histogram range in at least one layer")

    # Pass 2: hist counts per layer
    hist_counts = np.zeros((L, bins), dtype=np.int64)
    total_elems = np.zeros((L,), dtype=np.int64)

    def on_u_hist(layer_idx: int, u: torch.Tensor) -> None:
        z = (u.to(dtype=torch.float32) - mu_t[layer_idx]) / denom_t[layer_idx]
        v = _v_from_z(z, spike_def_id).reshape(-1)
        mn = float(v_min_r[layer_idx])
        mx = float(v_max_r[layer_idx])
        # torch.histc is non-deterministic on CUDA; compute bins deterministically with
        # discretization + bincount (integer counts are order-independent).
        width = mx - mn
        if not math.isfinite(width) or width <= 0.0:
            raise RateMatchError("Invalid histogram range width")
        scale = float(bins) / width
        v_clip = torch.clamp(v, min=mn, max=mx)
        idx = torch.floor((v_clip - mn) * scale).to(dtype=torch.int64)
        idx = torch.clamp(idx, 0, bins - 1)
        h = torch.bincount(idx, minlength=bins).to(dtype=torch.int64).detach().cpu().numpy()
        hist_counts[layer_idx] += h
        total_elems[layer_idx] += int(v.numel())

    for ids in token_windows:
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
            on_u=on_u_hist,
            need_logits=False,
        )

    if np.any(total_elems <= 0):
        raise RateMatchError("No samples collected for histogram")

    # Compute tau per target rate from histograms
    bin_w = (v_max_r - v_min_r) / float(bins)

    tau_by_tr: dict[str, np.ndarray] = {}
    for tr, r_targets in r_targets_by_target_rate.items():
        r_targets = np.asarray(r_targets, dtype=np.float64)
        if r_targets.shape != (L,):
            raise RateMatchError(f"Target rates shape mismatch for {tr}: {r_targets.shape}")
        tau = np.zeros((L,), dtype=np.float64)
        for l in range(L):
            cdf = np.cumsum(hist_counts[l], dtype=np.int64) / float(total_elems[l])
            tail = 1.0 - cdf
            target = float(r_targets[l])
            idx = int(np.argmax(tail <= target))
            edge_idx = min(idx + 1, bins)
            tau[l] = float(v_min_r[l] + bin_w[l] * edge_idx)
        tau_by_tr[str(tr)] = tau

    # Pass 3: achieved rates (and optionally rasters) for all target rates in one pass.
    achieved_spikes: dict[str, np.ndarray] = {tr: np.zeros((L,), dtype=np.float64) for tr in tau_by_tr}
    denom_units: dict[str, np.ndarray] = {tr: np.zeros((L,), dtype=np.float64) for tr in tau_by_tr}

    rasters: RasterBundle | None = None
    if collect_rasters:
        a_by_tr: dict[str, np.ndarray] = {
            tr: np.zeros((len(token_windows), L, seq_len), dtype=np.uint16) for tr in tau_by_tr
        }
        x_by_tr: dict[str, np.ndarray] = {
            tr: np.zeros((len(token_windows), L, seq_len), dtype=np.uint8) for tr in tau_by_tr
        }

        rasters = RasterBundle(a_by_target_rate=a_by_tr, x_by_target_rate=x_by_tr)

    cur_seq = {"i": 0}

    def on_u_verify(layer_idx: int, u: torch.Tensor) -> None:
        z = (u.to(dtype=torch.float32) - mu_t[layer_idx]) / denom_t[layer_idx]
        v = _v_from_z(z, spike_def_id)
        for tr, tau_vec in tau_by_tr.items():
            tau = float(tau_vec[layer_idx])
            spikes = v > tau
            achieved_spikes[tr][layer_idx] += float(spikes.sum().detach().cpu().item())
            denom_units[tr][layer_idx] += float(v.numel())
            if rasters is not None:
                a_tok = spikes.sum(dim=-1).to(dtype=torch.int32).detach().cpu().numpy()
                a_tok = a_tok.reshape(-1)  # batch=1
                rasters.a_by_target_rate[tr][cur_seq["i"], layer_idx, :] = a_tok.astype(np.uint16)
                rasters.x_by_target_rate[tr][cur_seq["i"], layer_idx, :] = (a_tok > 0).astype(np.uint8)

    for ids in token_windows:
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
            on_u=on_u_verify,
            need_logits=False,
        )
        cur_seq["i"] += 1

    achieved_by_tr = {tr: achieved_spikes[tr] / np.maximum(denom_units[tr], 1.0) for tr in tau_by_tr}
    for tr, achieved in achieved_by_tr.items():
        targets = np.asarray(r_targets_by_target_rate[tr], dtype=np.float64)
        max_err = float(np.max(np.abs(achieved - targets)))
        if max_err > tol:
            raise RateMatchError(f"Rate match failed for target_rate={tr} max_abs_err={max_err} tol={tol}")

    return (
        TauCalibration(
            tau_by_target_rate=tau_by_tr,
            achieved_rate_by_target_rate=achieved_by_tr,
            v_min=v_min_r,
            v_max=v_max_r,
        ),
        rasters,
    )
