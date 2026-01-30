from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .hooks import AttentionGainIntervention, GainIntervention, MlpLayer, UCapture, get_attn_layers


class ForwardError(RuntimeError):
    pass


@dataclass(frozen=True)
class ForwardOutput:
    logits: torch.Tensor | None


def run_with_u_capture(
    *,
    model: torch.nn.Module,
    layers: list[MlpLayer],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gain: float,
    gain_target: str = "MLP",
    gain_by_layer: dict[int, float] | None = None,
    default_gain: float = 1.0,
    on_u: Callable[[int, torch.Tensor], None],
    need_logits: bool,
) -> ForwardOutput:
    if input_ids.ndim != 2:
        raise ForwardError("input_ids must be [batch, T]")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    gain_target_norm = str(gain_target).upper()
    if gain_target_norm == "ATTN":
        attn_layers = get_attn_layers(model)
        if gain_by_layer is None:
            intervention = AttentionGainIntervention(attn_layers, gain=float(gain))
        else:
            intervention = AttentionGainIntervention(
                attn_layers, gain_by_layer=gain_by_layer, default_gain=float(default_gain)
            )
    elif gain_target_norm == "MLP":
        if gain_by_layer is None:
            intervention = GainIntervention(layers, gain=float(gain))
        else:
            intervention = GainIntervention(layers, gain_by_layer=gain_by_layer, default_gain=float(default_gain))
    else:
        raise ForwardError(f"Unknown gain_target: {gain_target}")

    with torch.no_grad():
        with intervention:
            with UCapture(layers, on_u=on_u):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
    logits = getattr(out, "logits", None) if need_logits else None
    return ForwardOutput(logits=logits)
