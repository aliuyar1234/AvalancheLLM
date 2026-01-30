from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..canon import get_canon
from ..datasets.sampling import select_token_windows
from ..errors import DependencyError
from ..io.artifacts import RunWriter
from ..io.jsonl import JsonlLogger
from ..metrics.signatures import chi_susceptibility
from ..model.hooks import GainIntervention, get_mlp_layers
from ..model.loader import load_model
from ..plotting.figures import simple_line_plot
from ..plotting.tables import write_table
from ..raster.extract import extract_rasters
from ..raster.nulls import within_layer_time_permutation


def _load_run_record(run_dir: Path) -> dict[str, Any]:
    canon = get_canon()
    rr_path = run_dir / str(canon["OUTPUT"]["RUN_RECORD_JSON"])
    if not rr_path.is_file():
        raise DependencyError(f"Missing run record file: {rr_path}")
    return json.loads(rr_path.read_text(encoding="utf-8"))


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size <= 0:
        raise RuntimeError("Cannot compute bootstrap CI on empty sample")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Cannot compute bootstrap CI on non-finite sample")
    lo = float(np.quantile(values, 0.025))
    hi = float(np.quantile(values, 0.975))
    return (lo, hi)


def _branching_counts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-sequence branching counts for pooled branching estimates.

    x: occupancy raster [N, L, T]
    Returns: (denom, num_time, num_depth), each shape [N]
    """
    if x.ndim != 3:
        raise ValueError("x must be [N, L, T]")
    denom = np.sum(x, axis=(1, 2), dtype=np.int64)
    num_time = np.sum(x[:, :, :-1] * x[:, :, 1:], axis=(1, 2), dtype=np.int64)
    num_depth = np.sum(x[:, :-1, :] * x[:, 1:, :], axis=(1, 2), dtype=np.int64)
    return denom, num_time, num_depth


def _pooled_branching_from_counts(
    denom: np.ndarray, num_time: np.ndarray, num_depth: np.ndarray
) -> tuple[float, float, float]:
    denom_total = int(np.sum(denom, dtype=np.int64))
    if denom_total <= 0:
        raise RuntimeError("Zero occupancy denominator in branching computation")
    num_time_total = int(np.sum(num_time, dtype=np.int64))
    num_depth_total = int(np.sum(num_depth, dtype=np.int64))
    b_time = float(num_time_total) / float(denom_total)
    b_depth = float(num_depth_total) / float(denom_total)
    return (b_time, b_depth, float(b_time + b_depth))


def run_phase6_b(
    *,
    writer: RunWriter,
    events: JsonlLogger,
    phase_log: JsonlLogger,
    config: dict[str, Any],
    args: Any,
    dep_cal: Path,
    dep_gstar: Path,
    dep_nulls: Path,
    dep_s05: Path,
) -> None:
    canon = get_canon()
    pipeline = config.get("pipeline", {})
    seq_len = int(pipeline.get("seq_len", canon["CONST"]["SEQ_LEN_TOKENS"]))

    rr_cal = _load_run_record(dep_cal)
    rr_gstar = _load_run_record(dep_gstar)
    rr_nulls = _load_run_record(dep_nulls)
    rr_s05 = _load_run_record(dep_s05)

    # Phase 1 artifacts (mu/sigma)
    cal_results = dep_cal / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    cal_npz = cal_results / str(canon["OUTPUT"]["CALIBRATION_STATS_NPZ_BASENAME"])
    if not cal_npz.is_file():
        raise DependencyError("Missing Phase1 calibration_stats.npz dependency")
    cal = np.load(cal_npz)
    mu = cal["mu"]
    sigma = cal["sigma"]

    # Phase 4 tau table (rate matched on Dataset A calibration slice)
    nulls_results = dep_nulls / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    tau_path = nulls_results / str(canon["OUTPUT"]["TAU_RATE_MATCHED_PARQUET_BASENAME"])
    if not tau_path.is_file():
        raise DependencyError("Missing Phase4 tau table dependency")
    tau_df = pd.read_parquet(tau_path)
    for col in ("spike_def_id", "target_rate", "g", "layer", "tau"):
        if col not in tau_df.columns:
            raise DependencyError(f"Tau table missing required column: {col}")

    # gstar per (spike_def_id, target_rate)
    gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    if not gstar_path.is_file():
        raise DependencyError("Missing gstar.json dependency")
    gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
    by_cond = gstar_obj.get("by_condition", {})
    if not isinstance(by_cond, dict):
        raise DependencyError("Invalid gstar.json format")

    model_role = str(config.get("model_role") or canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"])
    model_id = str(config.get("model_selected", {}).get("hf_id", ""))
    if not model_id:
        raise RuntimeError("Missing model_selected.hf_id in resolved config")
    lm = load_model(model_id, device=args.device, dtype=args.dtype)
    layers = get_mlp_layers(lm.model)
    if not layers:
        raise RuntimeError("Model has no MLP layers")
    d_ff = getattr(layers[0].gate_proj, "out_features", None)
    writer.run_record["model"].update(
        {
            "hf_id": lm.hf_id,
            "model_revision": lm.model_revision,
            "tokenizer_revision": lm.tokenizer_revision,
            "model_role": model_role,
            "n_layers": int(len(layers)),
            "d_ff": int(d_ff) if isinstance(d_ff, int) else None,
        }
    )
    if mu.shape[0] != len(layers) or sigma.shape[0] != len(layers):
        raise RuntimeError("mu/sigma layer dimension does not match model layers")

    # Dataset B evaluation slice
    ds_b = config.get("datasets", {}).get("B", {})
    hf_id_b = str(ds_b.get("hf_id", ""))
    hf_cfg_b = ds_b.get("config")
    split_b = str(ds_b.get("split", ""))
    if not hf_id_b or not split_b:
        raise RuntimeError("Missing datasets.B hf_id/split in resolved config")
    n_windows_b = int(canon["CONST"]["N_WINDOWS_B_EVAL"])
    selected_b, meta_b = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["B"]),
        hf_id=hf_id_b,
        hf_config=str(hf_cfg_b) if hf_cfg_b else None,
        split=split_b,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_windows_b,
        run_id=writer.run_id,
    )
    b_windows = [w.input_ids for w in selected_b]
    b_seq_ids = np.array([w.chunk_index for w in selected_b], dtype=np.int32)
    writer.run_record.setdefault("dataset", {}).setdefault("slices", {})["B_eval"] = meta_b
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256_B_eval"] = str(meta_b["dataset_slice_sha256"])
    writer.flush_run_record()

    tol = float(canon["CONST"]["RATE_MATCH_TOL_ABS"])
    baseline = float(canon["CONST"]["GAIN_BASELINE"])
    reps = int(canon["CONST"]["BOOTSTRAP_REPS"])
    seed_base = int(canon["CONST"]["BOOTSTRAP_SEED"])

    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rate_labels = [str(float(x)) for x in pipeline.get("target_rates", [])]

    def _tau_vec(*, spike_def_id: str, target_rate: str, g: float) -> np.ndarray:
        sub = tau_df[
            (tau_df["spike_def_id"] == spike_def_id)
            & (tau_df["target_rate"] == float(target_rate))
            & (tau_df["g"] == float(g))
        ]
        if sub.shape[0] != len(layers):
            raise DependencyError("Tau table missing required layer rows for a condition")
        sub = sub.sort_values(["layer"], kind="mergesort")
        layers_idx = sub["layer"].to_numpy(dtype=np.int64, copy=True)
        if not np.array_equal(layers_idx, np.arange(len(layers), dtype=np.int64)):
            raise DependencyError("Tau table layer indices are not contiguous 0..L-1")
        vec = sub["tau"].to_numpy(dtype=np.float64, copy=True)
        if vec.shape != (len(layers),):
            raise DependencyError("Invalid tau vector shape")
        return vec

    def _bootstrap_branching(
        denom: np.ndarray, num_time: np.ndarray, num_depth: np.ndarray, *, seed: int
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        point = _pooled_branching_from_counts(denom, num_time, num_depth)
        rng = np.random.default_rng(seed)
        n_seq = int(denom.shape[0])
        bt: list[float] = []
        bd: list[float] = []
        btot: list[float] = []
        for _ in range(reps):
            idx = rng.integers(0, n_seq, size=n_seq)
            d = np.sum(denom[idx], dtype=np.int64)
            if int(d) <= 0:
                continue
            nt = np.sum(num_time[idx], dtype=np.int64)
            nd = np.sum(num_depth[idx], dtype=np.int64)
            bt.append(float(nt) / float(d))
            bd.append(float(nd) / float(d))
            btot.append(float(nt + nd) / float(d))
        if not btot:
            raise RuntimeError("Bootstrap produced no valid branching samples")
        bt_lo, bt_hi = _bootstrap_ci(np.asarray(bt, dtype=np.float64))
        bd_lo, bd_hi = _bootstrap_ci(np.asarray(bd, dtype=np.float64))
        btot_lo, btot_hi = _bootstrap_ci(np.asarray(btot, dtype=np.float64))
        return (
            float(point[0]),
            float(point[1]),
            float(point[2]),
            float(bt_lo),
            float(bt_hi),
            float(bd_lo),
            float(bd_hi),
            float(btot_lo),
            float(btot_hi),
        )

    # Cache nll per gain (independent of spike_def_id / target_rate)
    nll_cache: dict[float, tuple[float, float, float, float, float, float]] = {}

    def _nll_mean_and_ppl_with_ci(g: float, *, seed: int) -> tuple[float, float, float, float, float, float]:
        key = float(g)
        if key in nll_cache:
            return nll_cache[key]

        nll_sums: list[float] = []
        token_counts: list[int] = []
        for ids in b_windows:
            input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
            attn = torch.ones_like(input_ids)
            with torch.no_grad():
                with GainIntervention(layers, gain=key):
                    out = lm.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError("Model did not return logits")
            if logits.ndim != 3:
                raise RuntimeError("Unexpected logits shape")
            logits_shift = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            loss = F.cross_entropy(
                logits_shift.reshape(-1, logits_shift.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            nll_sums.append(float(loss.detach().cpu().item()))
            token_counts.append(int(labels.numel()))

        toks = np.asarray(token_counts, dtype=np.int64)
        sums = np.asarray(nll_sums, dtype=np.float64)
        total_tokens = int(np.sum(toks, dtype=np.int64))
        if total_tokens <= 0:
            raise RuntimeError("No tokens scored for perplexity")
        total_nll = float(np.sum(sums, dtype=np.float64))
        nll_mean = total_nll / float(total_tokens)
        ppl = float(np.exp(nll_mean))

        rng = np.random.default_rng(seed)
        n_seq = int(toks.shape[0])
        bs_nll: list[float] = []
        bs_ppl: list[float] = []
        for _ in range(reps):
            idx = rng.integers(0, n_seq, size=n_seq)
            tok = int(np.sum(toks[idx], dtype=np.int64))
            if tok <= 0:
                continue
            nll = float(np.sum(sums[idx], dtype=np.float64)) / float(tok)
            bs_nll.append(nll)
            bs_ppl.append(float(np.exp(nll)))
        if not bs_ppl:
            raise RuntimeError("Bootstrap produced no valid perplexity samples")
        nll_lo, nll_hi = _bootstrap_ci(np.asarray(bs_nll, dtype=np.float64))
        ppl_lo, ppl_hi = _bootstrap_ci(np.asarray(bs_ppl, dtype=np.float64))

        nll_cache[key] = (float(nll_mean), float(ppl), float(nll_lo), float(nll_hi), float(ppl_lo), float(ppl_hi))
        return nll_cache[key]

    rows: list[dict[str, Any]] = []
    cond_id = 0
    for sd in spike_defs:
        for tr in target_rate_labels:
            gstar_key = f"{sd}|{float(tr)}"
            if gstar_key not in by_cond:
                raise DependencyError(f"gstar missing for {gstar_key}")
            gstar_val = float(by_cond[gstar_key]["gstar"])
            for gcond, g in [("g1", baseline), ("gstar", gstar_val)]:
                tau_vec = _tau_vec(spike_def_id=sd, target_rate=tr, g=g)
                ras = extract_rasters(
                    model=lm.model,
                    layers=layers,
                    token_windows=b_windows,
                    gain=float(g),
                    spike_def_id=sd,
                    mu=mu,
                    sigma=sigma,
                    tau_by_layer=tau_vec,
                    device=args.device,
                    seq_len=seq_len,
                )
                x = ras.x.astype(np.uint8, copy=False)
                denom, num_time, num_depth = _branching_counts(x)
                (
                    b_time,
                    b_depth,
                    b_tot,
                    b_time_ci_low,
                    b_time_ci_high,
                    b_depth_ci_low,
                    b_depth_ci_high,
                    b_tot_ci_low,
                    b_tot_ci_high,
                ) = _bootstrap_branching(denom, num_time, num_depth, seed=seed_base + 1000 + cond_id)

                x_perm = np.zeros_like(x, dtype=np.uint8)
                for n in range(len(b_windows)):
                    out = within_layer_time_permutation(
                        ras.a[n], run_id=writer.run_id, cond_id=cond_id, seq_id=int(b_seq_ids[n])
                    )
                    x_perm[n] = out.x_perm.astype(np.uint8)
                denom_p, num_time_p, num_depth_p = _branching_counts(x_perm)
                b_time_p, b_depth_p, b_tot_p = _pooled_branching_from_counts(denom_p, num_time_p, num_depth_p)

                # Bootstrap delta_b_tot from pooled branching under resampled sequences.
                rng = np.random.default_rng(seed_base + 2000 + cond_id)
                n_seq = int(denom.shape[0])
                delta_bs: list[float] = []
                for _ in range(reps):
                    idx = rng.integers(0, n_seq, size=n_seq)
                    d_real = int(np.sum(denom[idx], dtype=np.int64))
                    d_null = int(np.sum(denom_p[idx], dtype=np.int64))
                    if d_real <= 0 or d_null <= 0:
                        continue
                    nt_real = int(np.sum(num_time[idx], dtype=np.int64))
                    nd_real = int(np.sum(num_depth[idx], dtype=np.int64))
                    nt_null = int(np.sum(num_time_p[idx], dtype=np.int64))
                    nd_null = int(np.sum(num_depth_p[idx], dtype=np.int64))
                    btot_real = float(nt_real + nd_real) / float(d_real)
                    btot_null = float(nt_null + nd_null) / float(d_null)
                    delta_bs.append(float(btot_real - btot_null))
                if not delta_bs:
                    raise RuntimeError("Bootstrap produced no valid delta_b_tot samples")
                delta_b_tot = float(b_tot - b_tot_p)
                delta_lo, delta_hi = _bootstrap_ci(np.asarray(delta_bs, dtype=np.float64))

                chi = float(chi_susceptibility(ras.a.astype(np.uint16)))
                y = np.sum(ras.a.astype(np.uint16), axis=(1, 2), dtype=np.float64)
                rng = np.random.default_rng(seed_base + 3000 + cond_id)
                chi_bs: list[float] = []
                for _ in range(reps):
                    idx = rng.integers(0, y.size, size=y.size)
                    yy = y[idx]
                    mean = float(np.mean(yy))
                    var = float(np.var(yy))
                    chi_bs.append(var / (mean + float(canon["CONST"]["CHI_EPS"])))
                chi_lo, chi_hi = _bootstrap_ci(np.asarray(chi_bs, dtype=np.float64))

                g_seed = int(round(float(g) * 10000.0))
                nll_mean, ppl, nll_lo, nll_hi, ppl_lo, ppl_hi = _nll_mean_and_ppl_with_ci(
                    float(g), seed=seed_base + 4000 + g_seed
                )

                rows.append(
                    {
                        "run_id": writer.run_id,
                        "stage_id": writer.phase_id,
                        "model_id": model_id,
                        "model_role": model_role,
                        "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["B"]),
                        "spike_def_id": sd,
                        "target_rate": float(tr),
                        "g_condition": gcond,
                        "g": float(g),
                        "b_time": float(b_time),
                        "b_depth": float(b_depth),
                        "b_tot": float(b_tot),
                        "delta_b_tot": float(delta_b_tot),
                        "chi": float(chi),
                        "n_sequences": int(len(b_windows)),
                        "ppl": float(ppl),
                        "nll_mean": float(nll_mean),
                        "opt_b_time_ci_low": float(b_time_ci_low),
                        "opt_b_time_ci_high": float(b_time_ci_high),
                        "opt_b_depth_ci_low": float(b_depth_ci_low),
                        "opt_b_depth_ci_high": float(b_depth_ci_high),
                        "opt_b_tot_ci_low": float(b_tot_ci_low),
                        "opt_b_tot_ci_high": float(b_tot_ci_high),
                        "opt_delta_b_tot_ci_low": float(delta_lo),
                        "opt_delta_b_tot_ci_high": float(delta_hi),
                        "opt_chi_ci_low": float(chi_lo),
                        "opt_chi_ci_high": float(chi_hi),
                        "opt_nll_mean_ci_low": float(nll_lo),
                        "opt_nll_mean_ci_high": float(nll_hi),
                        "opt_ppl_ci_low": float(ppl_lo),
                        "opt_ppl_ci_high": float(ppl_hi),
                        "config_hash": writer.config_hash,
                        "code_version": writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata"),
                    }
                )
                cond_id += 1

    df = pd.DataFrame(rows)

    # Write results/metrics.parquet (machine-readable)
    results_dir = writer.run_dir / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    if metrics_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {metrics_path}")
    df.to_parquet(metrics_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=metrics_path.relative_to(writer.run_dir).as_posix())

    # Table T02_GENERALIZATION
    out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T02_GENERALIZATION"]))
    out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T02_GENERALIZATION"]))
    write_table(df=df, out_csv=out_csv, out_parquet=out_parq)
    writer.register_artifact(logical_name="table_T02_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="table_T02_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())

    # Figure F06_GENERALIZATION_B (one representative condition)
    fig_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F06_GENERALIZATION_B"]))
    fig_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F06_GENERALIZATION_B"]))
    pick_sd = spike_defs[0]
    pick_tr = float(target_rate_labels[0])
    sub = df[(df["spike_def_id"] == pick_sd) & (df["target_rate"] == pick_tr)].copy()
    sub = sub.sort_values(["g_condition"], kind="mergesort")
    xs = [0, 1]
    labels = ["g1", "gstar"]
    ppl_line = [float(sub[sub["g_condition"] == lab]["ppl"].iloc[0]) for lab in labels]
    btot_line = [float(sub[sub["g_condition"] == lab]["b_tot"].iloc[0]) for lab in labels]
    ppl_err = [
        max(
            float(sub[sub["g_condition"] == lab]["ppl"].iloc[0] - sub[sub["g_condition"] == lab]["opt_ppl_ci_low"].iloc[0]),
            float(sub[sub["g_condition"] == lab]["opt_ppl_ci_high"].iloc[0] - sub[sub["g_condition"] == lab]["ppl"].iloc[0]),
        )
        for lab in labels
    ]
    btot_err = [
        max(
            float(sub[sub["g_condition"] == lab]["b_tot"].iloc[0] - sub[sub["g_condition"] == lab]["opt_b_tot_ci_low"].iloc[0]),
            float(sub[sub["g_condition"] == lab]["opt_b_tot_ci_high"].iloc[0] - sub[sub["g_condition"] == lab]["b_tot"].iloc[0]),
        )
        for lab in labels
    ]
    simple_line_plot(
        x=xs,
        ys={"ppl": ppl_line, "b_tot": btot_line},
        title="Generalization on Dataset B",
        xlabel="condition (g1 vs gstar)",
        ylabel="value",
        out_pdf=fig_pdf,
        out_png=fig_png,
        yerr={"ppl": ppl_err, "b_tot": btot_err},
        meta={"labels": ",".join(labels), "spike_def_id": pick_sd, "target_rate": pick_tr, "tol": tol},
        provenance={
            "run_id": writer.run_id,
            "model_id": model_id,
            "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["B"]),
            "spike_def_id": pick_sd,
            "target_rate": pick_tr,
            "g_condition": "g1|gstar",
            "config_hash": writer.config_hash,
        },
    )
    writer.register_artifact(logical_name="F06_GENERALIZATION_B_pdf", relative_path=fig_pdf.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="F06_GENERALIZATION_B_png", relative_path=fig_png.relative_to(writer.run_dir).as_posix())

    writer.run_record["dependencies"] = [
        str(rr_cal.get("run_id")),
        str(rr_gstar.get("run_id")),
        str(rr_nulls.get("run_id")),
        str(rr_s05.get("run_id")),
    ]
    writer.run_record["conditions"] = [f"{r['spike_def_id']}|{r['target_rate']}|{r['g_condition']}" for r in rows]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 5})
    phase_log.emit("artifacts_written", {"count": 5})
