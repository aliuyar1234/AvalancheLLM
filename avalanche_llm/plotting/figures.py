from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .savefig import save_figure


LegendMode = Literal["auto", "inside", "right", "below", "none"]


def _is_asymmetric_yerr(v: Any) -> bool:
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        return False
    lo, hi = v
    return isinstance(lo, list) and isinstance(hi, list)


def _humanize_spike_def_id(spike_def_id: str) -> str | None:
    if not spike_def_id.startswith("SPIKE_"):
        return None

    parts = [p for p in spike_def_id.split("_")[1:] if p]
    if not parts:
        return None

    sign = ""
    if parts[-1] in {"POS", "ABS"}:
        sign = parts[-1]
        parts = parts[:-1]

    base = "-".join(p.lower() for p in parts) if parts else spike_def_id.lower()
    if sign == "POS":
        return f"{base} (+)"
    if sign == "ABS":
        return f"{base} (abs)"
    return base


def _humanize_metric_name(metric: str) -> str | None:
    if metric == "chi":
        return "χ"
    if metric.startswith("delta_b_"):
        return f"Δb_{metric.removeprefix('delta_b_')}"
    return None


def _format_series_label(raw: str, label_map: dict[str, str] | None) -> str:
    if label_map and raw in label_map:
        return label_map[raw]

    if ":" in raw:
        lhs, rhs = raw.split(":", 1)
        lhs_h = _humanize_spike_def_id(lhs) or lhs
        rhs_h = _humanize_metric_name(rhs) or rhs
        return f"{lhs_h}: {rhs_h}"

    return _humanize_spike_def_id(raw) or _humanize_metric_name(raw) or raw


def simple_line_plot(
    *,
    x,
    ys: dict[str, list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_pdf: Path,
    out_png: Path,
    yerr: dict[str, list[float] | tuple[list[float], list[float]]] | None = None,
    meta: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    label_map: dict[str, str] | None = None,
    legend_mode: LegendMode = "auto",
    legend_fontsize: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    series_labels = [_format_series_label(k, label_map) for k in ys.keys()]
    resolved_mode: LegendMode = legend_mode
    if resolved_mode == "auto":
        long_label = max((len(s) for s in series_labels), default=0) > 24
        resolved_mode = "right" if (len(ys) > 4 or long_label) else "inside"

    for name, y in ys.items():
        label = _format_series_label(name, label_map)
        if yerr is not None and name in yerr:
            err = yerr[name]
            if _is_asymmetric_yerr(err):
                lo, hi = err  # type: ignore[misc]
                ax.errorbar(x, y, yerr=[lo, hi], label=label, marker="o", capsize=3)
            else:
                ax.errorbar(x, y, yerr=err, label=label, marker="o", capsize=3)
        else:
            ax.plot(x, y, label=label, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if meta:
        txt = ", ".join(f"{k}={v}" for k, v in meta.items())
        ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=7, va="bottom")

    if resolved_mode == "none":
        fig.tight_layout()
    elif resolved_mode == "inside":
        ax.legend(fontsize=legend_fontsize, frameon=True)
        fig.tight_layout()
    elif resolved_mode == "right":
        # Reserve enough right margin so legend text is not clipped.
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=legend_fontsize,
            frameon=True,
            borderaxespad=0.0,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.get_window_extent(renderer=renderer)
        leg_bbox = legend.get_window_extent(renderer=renderer)
        legend.remove()

        width_frac = float(leg_bbox.width / max(1.0, fig_bbox.width))
        margin = 0.02
        rect_right = max(0.5, 1.0 - width_frac - margin)
        fig.tight_layout(rect=(0.0, 0.0, rect_right, 1.0))
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=legend_fontsize,
            frameon=True,
            borderaxespad=0.0,
        )
    elif resolved_mode == "below":
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.27)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, max(1, len(ys))),
            fontsize=legend_fontsize,
            frameon=True,
            borderaxespad=0.0,
        )
    else:
        raise ValueError(f"Unknown legend_mode: {legend_mode}")

    save_figure(fig=fig, out_pdf=out_pdf, out_png=out_png, provenance=provenance or meta, dpi=200)
    plt.close(fig)


def triptych_line_plot(
    *,
    x,
    ys_by_panel: dict[str, dict[str, list[float]]],
    title: str,
    xlabel: str,
    out_pdf: Path,
    out_png: Path,
    yerr: dict[str, dict[str, list[float] | tuple[list[float], list[float]]]] | None = None,
    meta: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    legend_fontsize: int = 8,
) -> None:
    panel_keys = list(ys_by_panel.keys())
    if not panel_keys:
        raise ValueError("ys_by_panel must contain at least one panel")

    n_panels = len(panel_keys)
    fig_w = 3.6 * float(n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 3.2), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, panel_key in zip(axes, panel_keys, strict=True):
        ys = ys_by_panel[panel_key]
        for series_key, y in ys.items():
            label = _format_series_label(series_key, None)
            panel_yerr = None
            if yerr is not None and panel_key in yerr and series_key in yerr[panel_key]:
                panel_yerr = yerr[panel_key][series_key]

            if panel_yerr is None:
                ax.plot(x, y, label=label, marker="o")
            else:
                if _is_asymmetric_yerr(panel_yerr):
                    lo, hi = panel_yerr  # type: ignore[misc]
                    ax.errorbar(x, y, yerr=[lo, hi], label=label, marker="o", capsize=3)
                else:
                    ax.errorbar(x, y, yerr=panel_yerr, label=label, marker="o", capsize=3)

        panel_title = _humanize_metric_name(panel_key) or panel_key
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("Δb")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(labels),
            fontsize=legend_fontsize,
            frameon=True,
        )

    fig.suptitle(title, y=0.995)
    if meta:
        txt = ", ".join(f"{k}={v}" for k, v in meta.items())
        fig.text(0.01, 0.01, txt, fontsize=7, va="bottom", ha="left")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    save_figure(fig=fig, out_pdf=out_pdf, out_png=out_png, provenance=provenance or meta, dpi=200)
    plt.close(fig)
