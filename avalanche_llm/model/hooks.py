from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F


class HookError(RuntimeError):
    pass


@dataclass(frozen=True)
class MlpLayer:
    layer_index: int
    mlp: torch.nn.Module
    gate_proj: torch.nn.Module
    up_proj: torch.nn.Module
    down_proj: torch.nn.Module


@dataclass(frozen=True)
class AttnLayer:
    layer_index: int
    attn: torch.nn.Module


def _get_layers_container(model: torch.nn.Module) -> list[Any]:
    # Llama/Qwen2-style
    for path in [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    ]:
        cur: Any = model
        ok = True
        for part in path:
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, (list, torch.nn.ModuleList)):
            return list(cur)
    raise HookError("Unsupported architecture: cannot locate transformer layers container")


def get_mlp_layers(model: torch.nn.Module) -> list[MlpLayer]:
    layers = _get_layers_container(model)
    out: list[MlpLayer] = []
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            raise HookError(f"Layer {idx} missing .mlp")
        mlp = getattr(layer, "mlp")
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(mlp, name):
                raise HookError(f"Layer {idx} mlp missing {name}")
        out.append(
            MlpLayer(
                layer_index=idx,
                mlp=mlp,
                gate_proj=getattr(mlp, "gate_proj"),
                up_proj=getattr(mlp, "up_proj"),
                down_proj=getattr(mlp, "down_proj"),
            )
        )
    return out


def get_attn_layers(model: torch.nn.Module) -> list[AttnLayer]:
    layers = _get_layers_container(model)
    out: list[AttnLayer] = []
    for idx, layer in enumerate(layers):
        attn = None
        for name in ("self_attn", "attention", "attn"):
            if hasattr(layer, name):
                attn = getattr(layer, name)
                break
        if attn is None:
            raise HookError(f"Layer {idx} missing attention module (expected self_attn/attention/attn)")
        out.append(AttnLayer(layer_index=idx, attn=attn))
    return out


class UCapture:
    """
    Capture u = silu(gate_proj(h)) * up_proj(h) per layer via module hooks.
    """

    def __init__(self, layers: list[MlpLayer], on_u: Callable[[int, torch.Tensor], None]):
        self._layers = layers
        self._on_u = on_u
        self._handles: list[Any] = []
        self._gate_out: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "UCapture":
        for layer in self._layers:
            li = layer.layer_index

            def _gate_hook(_mod, _inp, out, *, _li=li) -> None:
                self._gate_out[_li] = out

            def _up_hook(_mod, _inp, out, *, _li=li) -> None:
                gate = self._gate_out.pop(_li, None)
                if gate is None:
                    raise HookError("up_proj hook fired before gate_proj output was captured")
                u = F.silu(gate) * out
                self._on_u(_li, u)

            self._handles.append(layer.gate_proj.register_forward_hook(_gate_hook))
            self._handles.append(layer.up_proj.register_forward_hook(_up_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._gate_out.clear()


class GainIntervention:
    """
    Scale MLP output m by a scalar gain g before residual add (h_next = ... + g*mlp_out).
    Implemented via a forward hook on each layer's mlp module.
    """

    def __init__(
        self,
        layers: list[MlpLayer],
        *,
        gain: float | None = None,
        gain_by_layer: dict[int, float] | None = None,
        default_gain: float = 1.0,
    ):
        self._layers = layers
        if gain_by_layer is not None and gain is not None:
            raise HookError("Pass either gain or gain_by_layer, not both")
        if gain_by_layer is None and gain is None:
            raise HookError("Missing gain")
        self._gain = float(gain) if gain is not None else None
        self._gain_by_layer = {int(k): float(v) for k, v in (gain_by_layer or {}).items()}
        self._default_gain = float(default_gain)
        self._handles: list[Any] = []

    def __enter__(self) -> "GainIntervention":
        def _hook(_mod, _inp, out, *, _g: float):
            if isinstance(out, torch.Tensor):
                return out * _g
            if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                scaled0 = out[0] * _g
                if isinstance(out, tuple):
                    return (scaled0,) + tuple(out[1:])
                return [scaled0] + list(out[1:])
            raise HookError("MLP output is not a Tensor; cannot apply gain")

        for layer in self._layers:
            g = self._gain if self._gain is not None else self._gain_by_layer.get(layer.layer_index, self._default_gain)
            self._handles.append(layer.mlp.register_forward_hook(lambda m, i, o, _g=g: _hook(m, i, o, _g=_g)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class AttentionGainIntervention:
    """
    Scale attention output by a scalar gain g before residual add.
    Implemented via a forward hook on each layer's attention module.
    """

    def __init__(
        self,
        layers: list[AttnLayer],
        *,
        gain: float | None = None,
        gain_by_layer: dict[int, float] | None = None,
        default_gain: float = 1.0,
    ):
        self._layers = layers
        if gain_by_layer is not None and gain is not None:
            raise HookError("Pass either gain or gain_by_layer, not both")
        if gain_by_layer is None and gain is None:
            raise HookError("Missing gain")
        self._gain = float(gain) if gain is not None else None
        self._gain_by_layer = {int(k): float(v) for k, v in (gain_by_layer or {}).items()}
        self._default_gain = float(default_gain)
        self._handles: list[Any] = []

    def __enter__(self) -> "AttentionGainIntervention":
        def _hook(_mod, _inp, out, *, _g: float):
            if isinstance(out, torch.Tensor):
                return out * _g
            if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                scaled0 = out[0] * _g
                if isinstance(out, tuple):
                    return (scaled0,) + tuple(out[1:])
                return [scaled0] + list(out[1:])
            raise HookError("Attention output is not a Tensor; cannot apply gain")

        for layer in self._layers:
            g = self._gain if self._gain is not None else self._gain_by_layer.get(layer.layer_index, self._default_gain)
            self._handles.append(layer.attn.register_forward_hook(lambda m, i, o, _g=g: _hook(m, i, o, _g=_g)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class DownprojInputCapture:
    """
    Capture the input to down_proj (which should equal u) for validation.
    """

    def __init__(self, layer: MlpLayer):
        self._layer = layer
        self._handle = None
        self.captured: torch.Tensor | None = None

    def __enter__(self) -> "DownprojInputCapture":
        def _pre(_mod, inp):
            if not inp:
                return
            self.captured = inp[0]

        self._handle = self._layer.down_proj.register_forward_pre_hook(_pre)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
        self._handle = None
