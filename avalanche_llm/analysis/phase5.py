from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..canon import get_canon
from ..datasets.sampling import select_token_windows
from ..errors import DependencyError
from ..io.artifacts import RunWriter
from ..io.jsonl import JsonlLogger
from ..metrics.avalanches import avalanche_size_stats, avalanches_from_a
from ..metrics.signatures import crackling_fit_diagnostics, chi_susceptibility
from ..events.rate_match_calibrate import calibrate_tau_and_optionally_rasters
from ..model.hooks import get_mlp_layers
from ..model.loader import load_model
from ..plotting.figures import simple_line_plot, triptych_line_plot
from ..plotting.savefig import save_figure
from ..plotting.tables import write_table
from ..raster.extract import extract_rasters
from ..raster.nulls import within_layer_time_circular_shift, within_layer_time_permutation


def _load_run_record(run_dir: Path) -> dict[str, Any]:
    canon = get_canon()
    rr_path = run_dir / str(canon["OUTPUT"]["RUN_RECORD_JSON"])
    if not rr_path.is_file():
        raise DependencyError(f"Missing run record file: {rr_path}")
    return json.loads(rr_path.read_text(encoding="utf-8"))


def run_phase5(
    *,
    writer: RunWriter,
    events: JsonlLogger,
    phase_log: JsonlLogger,
    config: dict[str, Any],
    args: Any,
    dep_cal: Path,
    dep_gain_grid: Path,
    dep_gstar: Path,
    dep_rasters: Path,
    dep_nulls: Path,
) -> None:
    canon = get_canon()
    pipeline = config.get("pipeline", {})
    seq_len = int(pipeline.get("seq_len", canon["CONST"]["SEQ_LEN_TOKENS"]))

    rr_cal = _load_run_record(dep_cal)
    rr_gain_grid = _load_run_record(dep_gain_grid)
    rr_gstar = _load_run_record(dep_gstar)
    rr_rasters = _load_run_record(dep_rasters)
    rr_nulls = _load_run_record(dep_nulls)

    ds_a = config.get("datasets", {}).get("A", {})
    hf_id = str(ds_a.get("hf_id", ""))
    hf_cfg = ds_a.get("config")
    split = str(ds_a.get("split", ""))
    if not hf_id or not split:
        raise RuntimeError("Missing datasets.A hf_id/split in resolved config")

    # Phase 1 artifacts
    cal_results = dep_cal / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    cal_npz = cal_results / str(canon["OUTPUT"]["CALIBRATION_STATS_NPZ_BASENAME"])
    cal_rates = cal_results / str(canon["OUTPUT"]["RATE_TARGETS_JSON_BASENAME"])
    if not cal_npz.is_file() or not cal_rates.is_file():
        raise DependencyError("Missing Phase1 calibration artifacts")
    cal = np.load(cal_npz)
    mu = cal["mu"]
    sigma = cal["sigma"]
    rate_targets = json.loads(cal_rates.read_text(encoding="utf-8"))

    # Phase 4 tau table
    nulls_results = dep_nulls / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    tau_path = nulls_results / str(canon["OUTPUT"]["TAU_RATE_MATCHED_PARQUET_BASENAME"])
    if not tau_path.is_file():
        raise DependencyError("Missing Phase4 tau table dependency")
    tau_df = pd.read_parquet(tau_path)
    for col in ("spike_def_id", "target_rate", "g", "layer", "tau"):
        if col not in tau_df.columns:
            raise DependencyError(f"Tau table missing required column: {col}")

    # Phase 3/4 raster artifacts required for null sanity checks and paper figures.
    phase3_results = dep_rasters / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    raster_path = phase3_results / str(canon["OUTPUT"]["RASTER_NPZ_BASENAME"])
    raster_metrics_path = phase3_results / str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    if not raster_path.is_file() or not raster_metrics_path.is_file():
        raise DependencyError("Missing Phase3 raster dependencies")

    null_npz_path = nulls_results / str(canon["OUTPUT"]["NULL_NPZ_BASENAME"])
    null_metrics_path = nulls_results / str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    if not null_npz_path.is_file() or not null_metrics_path.is_file():
        raise DependencyError("Missing Phase4 null raster dependencies")

    # Null sanity check: within-layer permutation preserves per-layer marginals exactly.
    rz = np.load(raster_path)
    keys = canon["ENUM"]["NPZ_KEY"]
    a3 = rz[str(keys["A_COUNT"])].astype(np.uint16)
    cond_id3 = rz[str(keys["COND_ID"])].astype(np.int32)
    seq_id3 = rz[str(keys["SEQ_ID"])].astype(np.int32)
    m3 = (
        pd.read_parquet(raster_metrics_path)
        .sort_values(["cond_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    for col in ("cond_id", "spike_def_id", "target_rate", "g_condition", "g", "null_id"):
        if col not in m3.columns:
            raise DependencyError(f"Phase3 metrics.parquet missing required column: {col}")

    nz = np.load(null_npz_path)
    a_null = nz[str(keys["A_COUNT"])].astype(np.uint16)
    cond_id_null = nz[str(keys["COND_ID"])].astype(np.int32)
    seq_id_null = nz[str(keys["SEQ_ID"])].astype(np.int32)
    m4 = (
        pd.read_parquet(null_metrics_path)
        .sort_values(["cond_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    for col in ("cond_id", "spike_def_id", "target_rate", "g_condition", "g", "null_id"):
        if col not in m4.columns:
            raise DependencyError(f"Phase4 metrics.parquet missing required column: {col}")

    if not np.array_equal(seq_id3.astype(np.int32, copy=False), seq_id_null.astype(np.int32, copy=False)):
        raise DependencyError("Phase4 null NPZ seq_id does not match Phase3 rasters seq_id")

    axis_by_cond3 = {int(cid): int(i) for i, cid in enumerate(cond_id3.tolist())}
    axis_by_cond4 = {int(cid): int(i) for i, cid in enumerate(cond_id_null.tolist())}
    null_none_id = str(canon["ENUM"]["NULL_ID"]["NULL_NONE"])
    perm_null_id = str(canon["ENUM"]["NULL_ID"]["NULL_RASTER_WITHIN_LAYER_TIME_PERM"])
    shift_null_id = str(canon["ENUM"]["NULL_ID"]["NULL_RASTER_WITHIN_LAYER_TIME_CIRC_SHIFT"])
    token_null_id = str(canon["ENUM"]["NULL_ID"]["NULL_TOKEN_SHUFFLE_INPUT"])

    if int(np.sum(m4["null_id"] == token_null_id)) <= 0:
        raise DependencyError("Phase4 nulls missing required token shuffle null_id rows")
    if int(np.sum(m4["null_id"] == shift_null_id)) <= 0:
        raise DependencyError("Phase4 nulls missing required circular shift null_id rows")

    for _, row_real in m3.iterrows():
        if str(row_real["null_id"]) != null_none_id:
            continue
        target_match = pd.Series(
            np.isclose(
                m4["target_rate"].to_numpy(dtype=np.float64, copy=False),
                float(row_real["target_rate"]),
                atol=1.0e-12,
                rtol=0.0,
            ),
            index=m4.index,
        )
        g_match = pd.Series(
            np.isclose(
                m4["g"].to_numpy(dtype=np.float64, copy=False),
                float(row_real["g"]),
                atol=1.0e-12,
                rtol=0.0,
            ),
            index=m4.index,
        )
        mask = (
            (m4["null_id"] == perm_null_id)
            & (m4["spike_def_id"] == row_real["spike_def_id"])
            & (m4["g_condition"] == row_real["g_condition"])
            & target_match
            & g_match
        )
        if int(np.sum(mask)) != 1:
            raise DependencyError(
                "Could not locate unique Phase4 perm null condition for a Phase3 condition "
                f"(spike_def_id={row_real['spike_def_id']}, target_rate={row_real['target_rate']}, "
                f"g_condition={row_real['g_condition']}, g={row_real['g']})"
            )
        row_null = m4[mask].iloc[0]
        axis_real = axis_by_cond3.get(int(row_real["cond_id"]))
        axis_null = axis_by_cond4.get(int(row_null["cond_id"]))
        if axis_real is None or axis_null is None:
            raise DependencyError("NPZ cond_id axis mapping missing for Phase3 or Phase4")
        for n in range(int(seq_id3.size)):
            for l in range(int(a3.shape[2])):
                real = a3[axis_real, n, l, :]
                perm = a_null[axis_null, n, l, :]
                if not np.array_equal(np.sort(real), np.sort(perm)):
                    raise RuntimeError("Phase4 within-layer permutation null failed marginal preservation check")
    phase4_perm_null_marginals_exact = True

    for _, row_real in m3.iterrows():
        if str(row_real["null_id"]) != null_none_id:
            continue
        target_match = pd.Series(
            np.isclose(
                m4["target_rate"].to_numpy(dtype=np.float64, copy=False),
                float(row_real["target_rate"]),
                atol=1.0e-12,
                rtol=0.0,
            ),
            index=m4.index,
        )
        g_match = pd.Series(
            np.isclose(
                m4["g"].to_numpy(dtype=np.float64, copy=False),
                float(row_real["g"]),
                atol=1.0e-12,
                rtol=0.0,
            ),
            index=m4.index,
        )
        mask = (
            (m4["null_id"] == shift_null_id)
            & (m4["spike_def_id"] == row_real["spike_def_id"])
            & (m4["g_condition"] == row_real["g_condition"])
            & target_match
            & g_match
        )
        if int(np.sum(mask)) != 1:
            raise DependencyError(
                "Could not locate unique Phase4 circular shift null condition for a Phase3 condition "
                f"(spike_def_id={row_real['spike_def_id']}, target_rate={row_real['target_rate']}, "
                f"g_condition={row_real['g_condition']}, g={row_real['g']})"
            )
        row_null = m4[mask].iloc[0]
        axis_real = axis_by_cond3.get(int(row_real["cond_id"]))
        axis_null = axis_by_cond4.get(int(row_null["cond_id"]))
        if axis_real is None or axis_null is None:
            raise DependencyError("NPZ cond_id axis mapping missing for Phase3 or Phase4")
        for n in range(int(seq_id3.size)):
            for l in range(int(a3.shape[2])):
                real = a3[axis_real, n, l, :]
                shift = a_null[axis_null, n, l, :]
                if not np.array_equal(np.sort(real), np.sort(shift)):
                    raise RuntimeError("Phase4 within-layer circular shift null failed marginal preservation check")
    phase4_shift_null_marginals_exact = True

    # Reconstruct Dataset A calibration slice (Phase1 run_id)
    slices = rr_cal.get("dataset", {}).get("slices", {})
    smeta = slices.get("A_calibration")
    if not isinstance(smeta, dict):
        raise DependencyError("Phase1 run_record missing dataset.slices.A_calibration metadata")
    n_windows = int(smeta.get("n_windows", 0))
    if n_windows <= 0:
        raise DependencyError("Invalid Phase1 calibration slice window count")

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

    selected, ds_meta = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_windows,
        run_id=str(rr_cal.get("run_id")),
    )
    if ds_meta["dataset_slice_sha256"] != rr_cal.get("hashes", {}).get("dataset_slice_sha256"):
        raise DependencyError("Dataset slice hash mismatch vs Phase1 run_record")
    token_windows = [w.input_ids for w in selected]
    seq_ids = np.array([w.chunk_index for w in selected], dtype=np.int32)

    writer.run_record.setdefault("dataset", {}).setdefault("slices", {})["A_calibration"] = smeta
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256"] = str(ds_meta["dataset_slice_sha256"])
    writer.flush_run_record()

    if mu.shape[0] != len(layers) or sigma.shape[0] != len(layers):
        raise RuntimeError("mu/sigma layer dimension does not match model layers")

    tol = float(canon["CONST"]["RATE_MATCH_TOL_ABS"])
    tau0 = float(canon["CONST"]["TAU0_BASELINE"])
    adj = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_4N"])
    seed0 = int(canon["CONST"]["BOOTSTRAP_SEED"])
    reps = int(canon["CONST"]["BOOTSTRAP_REPS"])
    chi_eps = float(canon["CONST"]["CHI_EPS"])
    null_none = str(canon["ENUM"]["NULL_ID"]["NULL_NONE"])

    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rate_labels = [str(float(x)) for x in pipeline.get("target_rates", [])]
    gains = [float(g) for g in pipeline.get("gain_grid", [])]

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

    def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
        if values.size <= 0 or not np.all(np.isfinite(values)):
            raise RuntimeError("Cannot compute bootstrap CI on empty or non-finite sample")
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

    cond_rows: list[dict[str, Any]] = []
    aval_rows: list[dict[str, Any]] = []
    cond_id = 0
    for sd in spike_defs:
        r_star_map = rate_targets.get("r_star_l", {}).get(sd, {})
        if not isinstance(r_star_map, dict):
            raise DependencyError("Phase1 rate_targets missing r_star_l for spike_def")
        for tr in target_rate_labels:
            target_vec = np.asarray(r_star_map.get(tr), dtype=np.float64)
            if target_vec.shape != (len(layers),):
                raise DependencyError("Phase1 r_star_l has unexpected shape")
            for g in gains:
                tau_vec = _tau_vec(spike_def_id=sd, target_rate=tr, g=g)
                ras = extract_rasters(
                    model=lm.model,
                    layers=layers,
                    token_windows=token_windows,
                    gain=float(g),
                    spike_def_id=sd,
                    mu=mu,
                    sigma=sigma,
                    tau_by_layer=tau_vec,
                    device=args.device,
                    seq_len=seq_len,
                )
                achieved = np.asarray(ras.achieved_rate_by_layer, dtype=np.float64)
                max_err = float(np.max(np.abs(achieved - target_vec)))
                if max_err > tol:
                    raise RuntimeError(f"Rate match acceptance failed: max_abs_err={max_err} tol={tol}")
                achieved_mean = float(np.mean(achieved))

                x = ras.x.astype(np.uint8, copy=False)
                denom_r, num_time_r, num_depth_r = _branching_counts(x)
                b_time, b_depth, b_tot = _pooled_branching_from_counts(denom_r, num_time_r, num_depth_r)

                # Within-layer permutation + circular-shift nulls for delta b.
                N = int(len(token_windows))
                denom_p = np.zeros((N,), dtype=np.int64)
                num_time_p = np.zeros((N,), dtype=np.int64)
                num_depth_p = np.zeros((N,), dtype=np.int64)
                denom_s = np.zeros((N,), dtype=np.int64)
                num_time_s = np.zeros((N,), dtype=np.int64)
                num_depth_s = np.zeros((N,), dtype=np.int64)
                for n in range(N):
                    out_p = within_layer_time_permutation(
                        ras.a[n], run_id=writer.run_id, cond_id=cond_id, seq_id=int(seq_ids[n])
                    )
                    x_p = out_p.x_perm.astype(np.uint8, copy=False)
                    denom_p[n] = int(np.sum(x_p, dtype=np.int64))
                    num_time_p[n] = int(np.sum(x_p[:, :-1] * x_p[:, 1:], dtype=np.int64))
                    num_depth_p[n] = int(np.sum(x_p[:-1, :] * x_p[1:, :], dtype=np.int64))

                    out_s = within_layer_time_circular_shift(
                        ras.a[n], run_id=writer.run_id, cond_id=cond_id, seq_id=int(seq_ids[n])
                    )
                    x_s = out_s.x_perm.astype(np.uint8, copy=False)
                    denom_s[n] = int(np.sum(x_s, dtype=np.int64))
                    num_time_s[n] = int(np.sum(x_s[:, :-1] * x_s[:, 1:], dtype=np.int64))
                    num_depth_s[n] = int(np.sum(x_s[:-1, :] * x_s[1:, :], dtype=np.int64))

                b_time_p, b_depth_p, b_tot_p = _pooled_branching_from_counts(denom_p, num_time_p, num_depth_p)
                b_time_s, b_depth_s, b_tot_s = _pooled_branching_from_counts(denom_s, num_time_s, num_depth_s)
                delta_b_time = float(b_time - b_time_p)
                delta_b_depth = float(b_depth - b_depth_p)
                delta_b_tot = float(b_tot - b_tot_p)
                delta_b_time_shift = float(b_time - b_time_s)
                delta_b_depth_shift = float(b_depth - b_depth_s)
                delta_b_tot_shift = float(b_tot - b_tot_s)

                chi = float(chi_susceptibility(ras.a.astype(np.uint16)))
                if not np.isfinite(chi):
                    raise RuntimeError("Non-finite chi encountered")

                # Bootstrap CIs over sequences (paired across real and nulls).
                boot_seed_hex = hashlib.sha256(
                    f"{seed0}|{writer.run_id}|{cond_id}|branch_boot".encode("utf-8")
                ).hexdigest()
                boot_seed = int(boot_seed_hex[:8], 16)
                rng = np.random.default_rng(boot_seed)
                idx = rng.integers(0, N, size=(reps, N))

                def _pooled_bs(denom: np.ndarray, num_time: np.ndarray, num_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    denom_tot = denom[idx].sum(axis=1).astype(np.float64)
                    num_time_tot = num_time[idx].sum(axis=1).astype(np.float64)
                    num_depth_tot = num_depth[idx].sum(axis=1).astype(np.float64)
                    b_time_bs = num_time_tot / denom_tot
                    b_depth_bs = num_depth_tot / denom_tot
                    return (b_time_bs, b_depth_bs, b_time_bs + b_depth_bs)

                b_time_bs, b_depth_bs, b_tot_bs = _pooled_bs(denom_r, num_time_r, num_depth_r)
                b_time_p_bs, b_depth_p_bs, b_tot_p_bs = _pooled_bs(denom_p, num_time_p, num_depth_p)
                b_time_s_bs, b_depth_s_bs, b_tot_s_bs = _pooled_bs(denom_s, num_time_s, num_depth_s)

                db_time_bs = b_time_bs - b_time_p_bs
                db_depth_bs = b_depth_bs - b_depth_p_bs
                db_tot_bs = b_tot_bs - b_tot_p_bs

                db_time_shift_bs = b_time_bs - b_time_s_bs
                db_depth_shift_bs = b_depth_bs - b_depth_s_bs
                db_tot_shift_bs = b_tot_bs - b_tot_s_bs

                b_time_lo, b_time_hi = _bootstrap_ci(b_time_bs)
                b_depth_lo, b_depth_hi = _bootstrap_ci(b_depth_bs)
                b_tot_lo, b_tot_hi = _bootstrap_ci(b_tot_bs)

                db_time_lo, db_time_hi = _bootstrap_ci(db_time_bs)
                db_depth_lo, db_depth_hi = _bootstrap_ci(db_depth_bs)
                db_tot_lo, db_tot_hi = _bootstrap_ci(db_tot_bs)

                db_time_shift_lo, db_time_shift_hi = _bootstrap_ci(db_time_shift_bs)
                db_depth_shift_lo, db_depth_shift_hi = _bootstrap_ci(db_depth_shift_bs)
                db_tot_shift_lo, db_tot_shift_hi = _bootstrap_ci(db_tot_shift_bs)

                y = np.sum(ras.a.astype(np.uint16), axis=(1, 2), dtype=np.float64)
                y_bs = y[idx]
                mean = y_bs.mean(axis=1)
                var = y_bs.var(axis=1)
                chi_bs = var / (mean + chi_eps)
                chi_lo, chi_hi = _bootstrap_ci(chi_bs)

                comps_all = []
                for n in range(len(token_windows)):
                    comps = avalanches_from_a(ras.a[n], adjacency_id=adj)
                    comps_all.extend(comps)
                    for ci, comp in enumerate(comps):
                        aval_rows.append(
                            {
                                "cond_id": int(cond_id),
                                "seq_id": int(seq_ids[n]),
                                "component_id": int(ci),
                                "size": int(comp.size),
                                "span_tokens": int(comp.span_tokens),
                                "span_layers": int(comp.span_layers),
                                "t_min": int(comp.t_min) if comp.t_min is not None else None,
                                "t_max": int(comp.t_max) if comp.t_max is not None else None,
                                "l_min": int(comp.l_min) if comp.l_min is not None else None,
                                "l_max": int(comp.l_max) if comp.l_max is not None else None,
                            }
                        )

                aval_stats = avalanche_size_stats(comps_all)
                crack_seed_hex = hashlib.sha256(f"{seed0}|{writer.run_id}|{cond_id}".encode("utf-8")).hexdigest()
                crack_seed = int(crack_seed_hex[:8], 16)
                crack = crackling_fit_diagnostics(comps_all, seed=crack_seed)
                if not crack.passed:
                    raise RuntimeError(
                        "Crackling fit quality gate failed "
                        f"(spike_def_id={sd} target_rate={tr} g={g} "
                        f"n_pts={crack.n_duration_points} ci_width={crack.ci_width} r2={crack.r2})"
                    )

                row = {
                    "cond_id": int(cond_id),
                    "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                    "spike_def_id": sd,
                    "target_rate": float(tr),
                    "g": float(g),
                    "null_id": null_none,
                    "tau0_baseline": tau0,
                    "rate_match_tol_abs": tol,
                    "achieved_rate_mean": achieved_mean,
                    "achieved_rate_max_abs_err": max_err,
                    "b_time": b_time,
                    "b_depth": b_depth,
                    "b_tot": b_tot,
                    "b_time_perm": b_time_p,
                    "b_depth_perm": b_depth_p,
                    "b_tot_perm": b_tot_p,
                    "delta_b_time": delta_b_time,
                    "delta_b_depth": delta_b_depth,
                    "delta_b_tot": delta_b_tot,
                    "chi": chi,
                    "opt_phase4_perm_null_marginals_exact": bool(phase4_perm_null_marginals_exact),
                    "opt_phase4_shift_null_marginals_exact": bool(phase4_shift_null_marginals_exact),
                    "opt_b_time_ci_low": float(b_time_lo),
                    "opt_b_time_ci_high": float(b_time_hi),
                    "opt_b_depth_ci_low": float(b_depth_lo),
                    "opt_b_depth_ci_high": float(b_depth_hi),
                    "opt_b_tot_ci_low": float(b_tot_lo),
                    "opt_b_tot_ci_high": float(b_tot_hi),
                    "opt_delta_b_time_ci_low": float(db_time_lo),
                    "opt_delta_b_time_ci_high": float(db_time_hi),
                    "opt_delta_b_depth_ci_low": float(db_depth_lo),
                    "opt_delta_b_depth_ci_high": float(db_depth_hi),
                    "opt_delta_b_tot_ci_low": float(db_tot_lo),
                    "opt_delta_b_tot_ci_high": float(db_tot_hi),
                    "opt_chi_ci_low": float(chi_lo),
                    "opt_chi_ci_high": float(chi_hi),
                    "opt_b_time_shift": float(b_time_s),
                    "opt_b_depth_shift": float(b_depth_s),
                    "opt_b_tot_shift": float(b_tot_s),
                    "opt_delta_b_time_shift": float(delta_b_time_shift),
                    "opt_delta_b_depth_shift": float(delta_b_depth_shift),
                    "opt_delta_b_tot_shift": float(delta_b_tot_shift),
                    "opt_delta_b_time_shift_ci_low": float(db_time_shift_lo),
                    "opt_delta_b_time_shift_ci_high": float(db_time_shift_hi),
                    "opt_delta_b_depth_shift_ci_low": float(db_depth_shift_lo),
                    "opt_delta_b_depth_shift_ci_high": float(db_depth_shift_hi),
                    "opt_delta_b_tot_shift_ci_low": float(db_tot_shift_lo),
                    "opt_delta_b_tot_shift_ci_high": float(db_tot_shift_hi),
                    "crackling_gamma": float(crack.gamma),
                    "crackling_ci_low": float(crack.ci_low),
                    "crackling_ci_high": float(crack.ci_high),
                    "opt_crackling_ci_width": float(crack.ci_width),
                    "opt_crackling_n_duration_points": int(crack.n_duration_points),
                    "opt_crackling_n_avalanches_used": int(crack.n_avalanches_used),
                    "opt_crackling_r2": float(crack.r2),
                    "opt_crackling_pass": bool(crack.passed),
                    "n_sequences": int(len(token_windows)),
                    "n_avalanches": int(aval_stats["n_avalanches"]),
                    "avalanche_size_mean": float(aval_stats["size_mean"]),
                    "avalanche_size_median": float(aval_stats["size_median"]),
                    "avalanche_span_tokens_mean": float(aval_stats["span_tokens_mean"]),
                    "avalanche_span_layers_mean": float(aval_stats["span_layers_mean"]),
                }
                if not np.isfinite(row["b_tot"]) or not np.isfinite(row["delta_b_tot"]):
                    raise RuntimeError("Non-finite branching metrics encountered")
                cond_rows.append(row)
                cond_id += 1

    expected_rows = len(spike_defs) * len(target_rate_labels) * len(gains)
    if len(cond_rows) != expected_rows:
        raise RuntimeError("Condition grid coverage mismatch in Phase5 analysis")

    # Write results artifacts
    results_dir = writer.run_dir / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    aval_path = results_dir / str(canon["OUTPUT"]["AVALANCHES_PARQUET_BASENAME"])
    if metrics_path.exists() or aval_path.exists():
        raise RuntimeError("Refusing to overwrite existing results files")
    pd.DataFrame(cond_rows).to_parquet(metrics_path, index=False)
    pd.DataFrame(aval_rows).to_parquet(aval_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=metrics_path.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="avalanches_parquet", relative_path=aval_path.relative_to(writer.run_dir).as_posix())

    # T01 summary table
    code_version = writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata")
    dataset_a = str(canon["ENUM"]["DATASET_ROLE"]["A"])
    t01_rows = []
    for r in cond_rows:
        row = {
            "run_id": writer.run_id,
            "stage_id": writer.phase_id,
            "model_id": model_id,
            "model_role": model_role,
            "dataset_role": dataset_a,
            "spike_def_id": r["spike_def_id"],
            "target_rate": r["target_rate"],
            "g": r["g"],
            "tau0_baseline": r["tau0_baseline"],
            "rate_match_tol_abs": r["rate_match_tol_abs"],
            "achieved_rate_mean": r["achieved_rate_mean"],
            "achieved_rate_max_abs_err": r["achieved_rate_max_abs_err"],
            "b_time": r["b_time"],
            "b_depth": r["b_depth"],
            "b_tot": r["b_tot"],
            "b_time_perm": r["b_time_perm"],
            "b_depth_perm": r["b_depth_perm"],
            "b_tot_perm": r["b_tot_perm"],
            "delta_b_time": r["delta_b_time"],
            "delta_b_depth": r["delta_b_depth"],
            "delta_b_tot": r["delta_b_tot"],
            "chi": r["chi"],
            "crackling_gamma": r["crackling_gamma"],
            "crackling_ci_low": r["crackling_ci_low"],
            "crackling_ci_high": r["crackling_ci_high"],
            "n_sequences": r["n_sequences"],
            "n_avalanches": r["n_avalanches"],
            "avalanche_size_mean": r["avalanche_size_mean"],
            "avalanche_size_median": r["avalanche_size_median"],
            "avalanche_span_tokens_mean": r["avalanche_span_tokens_mean"],
            "avalanche_span_layers_mean": r["avalanche_span_layers_mean"],
            "config_hash": writer.config_hash,
            "code_version": code_version,
        }
        for k, v in r.items():
            if isinstance(k, str) and k.startswith("opt_"):
                row[k] = v
        t01_rows.append(row)
    t01_df = pd.DataFrame(t01_rows)
    if not np.all(t01_df["achieved_rate_max_abs_err"].to_numpy(dtype=np.float64) <= tol):
        raise RuntimeError("Rate match tolerance check failed in T01 output")

    out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T01_SUMMARY"]))
    out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T01_SUMMARY"]))
    write_table(df=t01_df, out_csv=out_csv, out_parquet=out_parq)
    writer.register_artifact(logical_name="table_T01_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="table_T01_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())
    t01_df_art = pd.read_parquet(out_parq)

    # Appendix tables: T04 tail fits, T05 crackling diagnostics, T06 ablations.
    metrics_df = pd.read_parquet(metrics_path)
    aval_df = pd.read_parquet(aval_path)
    if "cond_id" not in aval_df.columns or "size" not in aval_df.columns:
        raise RuntimeError("avalanches.parquet missing required columns for tail fits")
    if "cond_id" not in metrics_df.columns:
        raise RuntimeError("metrics.parquet missing cond_id column for tail fits")

    xmin_pct = float(canon["CONST"]["TAIL_FIT_XMIN_PERCENTILE"])
    min_tail = int(canon["CONST"]["TAIL_FIT_MIN_TAIL_SAMPLES"])

    def _tail_fit_row(sizes: np.ndarray) -> dict[str, Any]:
        x = np.asarray(sizes, dtype=np.float64)
        x = x[np.isfinite(x) & (x > 0.0)]
        if x.size <= 0:
            raise RuntimeError("No positive avalanche sizes for tail fits")
        xmin = float(np.quantile(x, xmin_pct))
        tail = x[x >= xmin]
        n = int(tail.size)
        if n < min_tail:
            raise RuntimeError(f"Tail fit has too few samples: n_tail={n} min_tail={min_tail}")
        # Continuous approximations (descriptive only).
        alpha = 1.0 + float(n) / float(np.sum(np.log(tail / xmin)))
        lam = float(n) / float(np.sum(tail - xmin))
        logs = np.log(tail)
        mu_ln = float(np.mean(logs))
        sig_ln = float(np.std(logs))
        ll_pl = float(n * np.log(alpha - 1.0) + (alpha - 1.0) * n * np.log(xmin) - alpha * float(np.sum(np.log(tail))))
        ll_exp = float(n * np.log(lam) - lam * float(np.sum(tail - xmin)))
        ll_ln = float(
            -float(np.sum(np.log(tail * sig_ln * np.sqrt(2.0 * np.pi))))
            - float(np.sum(((logs - mu_ln) ** 2) / (2.0 * sig_ln * sig_ln)))
        )
        return {
            "xmin": float(xmin),
            "n_tail": int(n),
            "alpha_powerlaw": float(alpha),
            "lambda_exponential": float(lam),
            "lognorm_mu": float(mu_ln),
            "lognorm_sigma": float(sig_ln),
            "ll_powerlaw": float(ll_pl),
            "ll_exponential": float(ll_exp),
            "ll_lognormal": float(ll_ln),
            "llr_powerlaw_vs_lognormal": float(ll_pl - ll_ln),
            "llr_powerlaw_vs_exponential": float(ll_pl - ll_exp),
        }

    tail_rows: list[dict[str, Any]] = []
    meta_cols = ["cond_id", "spike_def_id", "target_rate", "g"]
    for c in meta_cols:
        if c not in metrics_df.columns:
            raise RuntimeError(f"metrics.parquet missing required column for tail fits: {c}")
    for _, mrow in metrics_df.sort_values(["cond_id"], kind="mergesort").iterrows():
        cid = int(mrow["cond_id"])
        sizes = aval_df[aval_df["cond_id"] == cid]["size"].to_numpy(dtype=np.float64, copy=False)
        fit = _tail_fit_row(sizes)
        tail_rows.append(
            {
                "run_id": writer.run_id,
                "stage_id": writer.phase_id,
                "model_id": model_id,
                "model_role": model_role,
                "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                "spike_def_id": str(mrow["spike_def_id"]),
                "target_rate": float(mrow["target_rate"]),
                "g": float(mrow["g"]),
                **fit,
                "config_hash": writer.config_hash,
                "code_version": code_version,
            }
        )

    t04_df = pd.DataFrame(tail_rows)
    out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T04_TAIL_FITS"]))
    out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T04_TAIL_FITS"]))
    write_table(df=t04_df, out_csv=out_csv, out_parquet=out_parq)
    writer.register_artifact(logical_name="table_T04_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="table_T04_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())

    # Crackling diagnostics table (derived from T01)
    required_t05 = [
        "opt_crackling_ci_width",
        "opt_crackling_n_duration_points",
        "opt_crackling_n_avalanches_used",
        "opt_crackling_r2",
        "opt_crackling_pass",
    ]
    for c in required_t05:
        if c not in t01_df_art.columns:
            raise RuntimeError(f"T01 missing required crackling diagnostics column: {c}")
    t05_df = pd.DataFrame(
        {
            "run_id": t01_df_art["run_id"],
            "stage_id": t01_df_art["stage_id"],
            "model_id": t01_df_art["model_id"],
            "model_role": t01_df_art["model_role"],
            "dataset_role": t01_df_art["dataset_role"],
            "spike_def_id": t01_df_art["spike_def_id"],
            "target_rate": t01_df_art["target_rate"],
            "g": t01_df_art["g"],
            "crackling_gamma": t01_df_art["crackling_gamma"],
            "crackling_ci_low": t01_df_art["crackling_ci_low"],
            "crackling_ci_high": t01_df_art["crackling_ci_high"],
            "crackling_ci_width": t01_df_art["opt_crackling_ci_width"],
            "n_duration_points": t01_df_art["opt_crackling_n_duration_points"],
            "n_avalanches_used": t01_df_art["opt_crackling_n_avalanches_used"],
            "r2": t01_df_art["opt_crackling_r2"],
            "pass": t01_df_art["opt_crackling_pass"],
            "config_hash": t01_df_art["config_hash"],
            "code_version": t01_df_art["code_version"],
        }
    )
    if not bool(t05_df["pass"].astype(bool).all()):
        raise RuntimeError("Crackling diagnostics pass gate failed for at least one condition")
    out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T05_CRACKLING_DIAGNOSTICS"]))
    out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T05_CRACKLING_DIAGNOSTICS"]))
    write_table(df=t05_df, out_csv=out_csv, out_parquet=out_parq)
    writer.register_artifact(logical_name="table_T05_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="table_T05_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())

    # Ablations table (rate-matched)
    gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    if not gstar_path.is_file():
        raise DependencyError("Missing gstar.json dependency")
    gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
    by_cond = gstar_obj.get("by_condition", {})
    if not isinstance(by_cond, dict):
        raise DependencyError("Invalid gstar.json format")

    inter = canon["ENUM"]["INTERVENTION_ID"]
    tgt = canon["ENUM"]["GAIN_TARGET"]
    baseline_gain = float(canon["CONST"]["GAIN_BASELINE"])

    # Layer bands as thirds of the depth.
    L = int(len(layers))
    cut1 = max(1, L // 3)
    cut2 = max(cut1 + 1, (2 * L) // 3)
    bands = {
        str(inter["MLP_BAND_EARLY"]): list(range(0, cut1)),
        str(inter["MLP_BAND_MID"]): list(range(cut1, cut2)),
        str(inter["MLP_BAND_LATE"]): list(range(cut2, L)),
    }

    def _b_tot_and_delta(*, a: np.ndarray, x: np.ndarray, cond_id_seed: int) -> tuple[float, float, float]:
        denom = float(np.sum(x, dtype=np.int64))
        if denom <= 0.0:
            raise RuntimeError("Zero occupancy denominator in ablation branching computation")
        num_time = float(np.sum(x[:, :, :-1] * x[:, :, 1:], dtype=np.int64))
        num_depth = float(np.sum(x[:, :-1, :] * x[:, 1:, :], dtype=np.int64))
        b_tot = float((num_time + num_depth) / denom)

        denom_p = 0.0
        num_time_p = 0.0
        num_depth_p = 0.0
        for n in range(int(x.shape[0])):
            out = within_layer_time_permutation(a[n], run_id=writer.run_id, cond_id=cond_id_seed, seq_id=int(seq_ids[n]))
            xp = out.x_perm.astype(np.uint8, copy=False)
            denom_p += float(np.sum(xp, dtype=np.int64))
            num_time_p += float(np.sum(xp[:, :-1] * xp[:, 1:], dtype=np.int64))
            num_depth_p += float(np.sum(xp[:-1, :] * xp[1:, :], dtype=np.int64))
        b_tot_p = float((num_time_p + num_depth_p) / denom_p) if denom_p > 0.0 else float("nan")
        delta_b_tot = float(b_tot - b_tot_p)
        chi = float(chi_susceptibility(a.astype(np.uint16)))
        return (b_tot, delta_b_tot, chi)

    ab_rows: list[dict[str, Any]] = []
    ab_cond_id = int(expected_rows)
    for sd in spike_defs:
        r_star_map = rate_targets.get("r_star_l", {}).get(sd, {})
        if not isinstance(r_star_map, dict):
            raise DependencyError("Phase1 rate_targets missing r_star_l for spike_def")
        for tr in target_rate_labels:
            tr_f = float(tr)
            key = f"{sd}|{tr_f}"
            if key not in by_cond:
                raise DependencyError(f"gstar missing for {key}")
            gstar_val = float(by_cond[key]["gstar"])
            target_vec = np.asarray(r_star_map.get(tr), dtype=np.float64)
            if target_vec.shape != (len(layers),):
                raise DependencyError("Phase1 r_star_l has unexpected shape")

            # Baseline (g1) reference: reuse the main T01 row (MLP_GLOBAL).
            base_row = t01_df_art[(t01_df_art["spike_def_id"] == sd) & (t01_df_art["target_rate"] == tr_f) & (t01_df_art["g"] == baseline_gain)]
            if base_row.shape[0] != 1:
                raise RuntimeError("Could not locate unique baseline row for ablations")
            base_row = base_row.iloc[0].to_dict()
            ab_rows.append(
                {
                    "run_id": writer.run_id,
                    "stage_id": writer.phase_id,
                    "model_id": model_id,
                    "model_role": model_role,
                    "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                    "spike_def_id": sd,
                    "target_rate": tr_f,
                    "g_condition": "g1",
                    "g": float(baseline_gain),
                    "intervention_id": str(inter["MLP_GLOBAL"]),
                    "gain_target": str(tgt["MLP"]),
                    "b_tot": float(base_row["b_tot"]),
                    "delta_b_tot": float(base_row["delta_b_tot"]),
                    "chi": float(base_row["chi"]),
                    "achieved_rate_max_abs_err": float(base_row["achieved_rate_max_abs_err"]),
                    "n_sequences": int(base_row["n_sequences"]),
                    "config_hash": writer.config_hash,
                    "code_version": code_version,
                }
            )

            # gstar for MLP_GLOBAL: reuse main T01 row.
            gstar_row = t01_df_art[(t01_df_art["spike_def_id"] == sd) & (t01_df_art["target_rate"] == tr_f) & (t01_df_art["g"] == gstar_val)]
            if gstar_row.shape[0] != 1:
                raise RuntimeError("Could not locate unique gstar row for ablations")
            gstar_row = gstar_row.iloc[0].to_dict()
            ab_rows.append(
                {
                    "run_id": writer.run_id,
                    "stage_id": writer.phase_id,
                    "model_id": model_id,
                    "model_role": model_role,
                    "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                    "spike_def_id": sd,
                    "target_rate": tr_f,
                    "g_condition": "gstar",
                    "g": float(gstar_val),
                    "intervention_id": str(inter["MLP_GLOBAL"]),
                    "gain_target": str(tgt["MLP"]),
                    "b_tot": float(gstar_row["b_tot"]),
                    "delta_b_tot": float(gstar_row["delta_b_tot"]),
                    "chi": float(gstar_row["chi"]),
                    "achieved_rate_max_abs_err": float(gstar_row["achieved_rate_max_abs_err"]),
                    "n_sequences": int(gstar_row["n_sequences"]),
                    "config_hash": writer.config_hash,
                    "code_version": code_version,
                }
            )

            # Attention gain (negative control) at gstar (rate-matched via fresh tau calibration).
            tau_cal, rasters = calibrate_tau_and_optionally_rasters(
                model=lm.model,
                layers=layers,
                token_windows=token_windows,
                gain=float(gstar_val),
                gain_target=str(tgt["ATTN"]),
                spike_def_id=sd,
                mu=mu,
                sigma=sigma,
                r_targets_by_target_rate={tr: target_vec},
                device=args.device,
                seq_len=seq_len,
                collect_rasters=True,
            )
            assert rasters is not None
            a_attn = rasters.a_by_target_rate[tr]
            x_attn = rasters.x_by_target_rate[tr]
            achieved_attn = np.asarray(tau_cal.achieved_rate_by_target_rate[tr], dtype=np.float64)
            max_err_attn = float(np.max(np.abs(achieved_attn - target_vec)))
            b_tot, delta_b_tot, chi = _b_tot_and_delta(a=a_attn, x=x_attn, cond_id_seed=ab_cond_id)
            ab_rows.append(
                {
                    "run_id": writer.run_id,
                    "stage_id": writer.phase_id,
                    "model_id": model_id,
                    "model_role": model_role,
                    "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                    "spike_def_id": sd,
                    "target_rate": tr_f,
                    "g_condition": "gstar",
                    "g": float(gstar_val),
                    "intervention_id": str(inter["ATTN_GLOBAL"]),
                    "gain_target": str(tgt["ATTN"]),
                    "b_tot": float(b_tot),
                    "delta_b_tot": float(delta_b_tot),
                    "chi": float(chi),
                    "achieved_rate_max_abs_err": float(max_err_attn),
                    "n_sequences": int(x_attn.shape[0]),
                    "config_hash": writer.config_hash,
                    "code_version": code_version,
                }
            )
            ab_cond_id += 1

            # Layer-local MLP gain bands at gstar (rate-matched via fresh tau calibration).
            for ab_id, layer_idxs in bands.items():
                gain_by_layer = {int(li): float(gstar_val) for li in layer_idxs}
                tau_cal, rasters = calibrate_tau_and_optionally_rasters(
                    model=lm.model,
                    layers=layers,
                    token_windows=token_windows,
                    gain=float(gstar_val),
                    gain_target=str(tgt["MLP"]),
                    gain_by_layer=gain_by_layer,
                    default_gain=float(baseline_gain),
                    spike_def_id=sd,
                    mu=mu,
                    sigma=sigma,
                    r_targets_by_target_rate={tr: target_vec},
                    device=args.device,
                    seq_len=seq_len,
                    collect_rasters=True,
                )
                assert rasters is not None
                a_loc = rasters.a_by_target_rate[tr]
                x_loc = rasters.x_by_target_rate[tr]
                achieved_loc = np.asarray(tau_cal.achieved_rate_by_target_rate[tr], dtype=np.float64)
                max_err_loc = float(np.max(np.abs(achieved_loc - target_vec)))
                b_tot, delta_b_tot, chi = _b_tot_and_delta(a=a_loc, x=x_loc, cond_id_seed=ab_cond_id)
                ab_rows.append(
                    {
                        "run_id": writer.run_id,
                        "stage_id": writer.phase_id,
                        "model_id": model_id,
                        "model_role": model_role,
                        "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                        "spike_def_id": sd,
                        "target_rate": tr_f,
                        "g_condition": "gstar",
                        "g": float(gstar_val),
                        "intervention_id": str(ab_id),
                        "gain_target": str(tgt["MLP"]),
                        "b_tot": float(b_tot),
                        "delta_b_tot": float(delta_b_tot),
                        "chi": float(chi),
                        "achieved_rate_max_abs_err": float(max_err_loc),
                        "n_sequences": int(x_loc.shape[0]),
                        "config_hash": writer.config_hash,
                        "code_version": code_version,
                    }
                )
                ab_cond_id += 1

    t06_df = pd.DataFrame(ab_rows)
    if not np.all(t06_df["achieved_rate_max_abs_err"].to_numpy(dtype=np.float64) <= tol):
        raise RuntimeError("Rate match tolerance check failed in T06 ablations output")
    out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T06_ABLATIONS"]))
    out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T06_ABLATIONS"]))
    write_table(df=t06_df, out_csv=out_csv, out_parquet=out_parq)
    writer.register_artifact(logical_name="table_T06_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="table_T06_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())

    # Figures F01-F05 and F08
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # F01: choose a Phase3 raster example with at least one component
    phase3_results = dep_rasters / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    raster_path = phase3_results / str(canon["OUTPUT"]["RASTER_NPZ_BASENAME"])
    raster_metrics_path = phase3_results / str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    if not raster_path.is_file() or not raster_metrics_path.is_file():
        raise DependencyError("Missing Phase3 raster dependencies for F01")
    rz = np.load(raster_path)
    keys = canon["ENUM"]["NPZ_KEY"]
    a3 = rz[str(keys["A_COUNT"])].astype(np.uint16)
    x3 = rz[str(keys["X_OCCUPANCY"])].astype(np.uint8)
    cond_id3 = rz[str(keys["COND_ID"])].astype(np.int32)
    seq_id3 = rz[str(keys["SEQ_ID"])].astype(np.int32)
    m3 = pd.read_parquet(raster_metrics_path).sort_values(["cond_id"], kind="mergesort").reset_index(drop=True)
    axis_by_cond3 = {int(cid): int(i) for i, cid in enumerate(cond_id3.tolist())}

    chosen = None
    chosen_comps = None
    for _, row in m3.iterrows():
        cid = int(row["cond_id"])
        axis = axis_by_cond3.get(cid)
        if axis is None:
            continue
        for n in range(int(seq_id3.size)):
            comps = avalanches_from_a(a3[axis, n], adjacency_id=adj)
            if comps:
                chosen = (row.to_dict(), axis, n)
                chosen_comps = comps
                break
        if chosen is not None:
            break
    if chosen is None or chosen_comps is None:
        raise RuntimeError("Could not find a Phase3 raster example with a connected component")

    cond_meta, axis, n = chosen
    img = x3[axis, n].astype(np.float32)
    out_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F01_RASTER_EXAMPLE"]))
    out_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F01_RASTER_EXAMPLE"]))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xlabel("token index")
    ax.set_ylabel("layer index")
    ax.set_title("Raster example")
    largest = max(chosen_comps, key=lambda c: c.size)
    if largest.t_min is not None and largest.t_max is not None and largest.l_min is not None and largest.l_max is not None:
        rect = plt.Rectangle(
            (largest.t_min - 0.5, largest.l_min - 0.5),
            (largest.t_max - largest.t_min + 1),
            (largest.l_max - largest.l_min + 1),
            fill=False,
            color="red",
            linewidth=2,
        )
        ax.add_patch(rect)
    meta_txt = f"run_id={writer.run_id}, spike_def_id={cond_meta.get('spike_def_id')}, target_rate={cond_meta.get('target_rate')}, g={cond_meta.get('g')}"
    ax.text(0.01, 0.01, meta_txt, transform=ax.transAxes, fontsize=7, va="bottom")
    fig.tight_layout()
    save_figure(
        fig=fig,
        out_pdf=out_pdf,
        out_png=out_png,
        provenance={
            "run_id": writer.run_id,
            "model_id": model_id,
            "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
            "spike_def_id": str(cond_meta.get("spike_def_id")),
            "target_rate": str(cond_meta.get("target_rate")),
            "g": str(cond_meta.get("g")),
            "config_hash": writer.config_hash,
        },
        dpi=200,
    )
    plt.close(fig)
    writer.register_artifact(logical_name="F01_RASTER_EXAMPLE_pdf", relative_path=out_pdf.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="F01_RASTER_EXAMPLE_png", relative_path=out_png.relative_to(writer.run_dir).as_posix())

    # Shared view for line plots
    pick_sd = spike_defs[0]
    pick_tr = float(target_rate_labels[0])
    series = t01_df_art[(t01_df_art["spike_def_id"] == pick_sd) & (t01_df_art["target_rate"] == pick_tr)].copy()
    series = series.sort_values(["g"], kind="mergesort")
    xs = series["g"].tolist()

    def _plot(
        fig_id: str,
        ys: dict[str, list[float]],
        title: str,
        ylabel: str,
        meta: dict[str, Any],
        *,
        yerr: dict[str, list[float] | tuple[list[float], list[float]]] | None = None,
    ) -> None:
        fig_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"][fig_id]))
        fig_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"][fig_id]))
        simple_line_plot(
            x=xs,
            ys=ys,
            title=title,
            xlabel="gain g",
            ylabel=ylabel,
            out_pdf=fig_pdf,
            out_png=fig_png,
            yerr=yerr,
            meta=meta,
            provenance={
                "run_id": writer.run_id,
                "model_id": model_id,
                "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                "spike_def_id": str(meta.get("spike_def_id", "multiple")),
                "target_rate": str(meta.get("target_rate", "multiple")),
                "g": str(meta.get("g", "grid")),
                "config_hash": writer.config_hash,
            },
        )
        writer.register_artifact(logical_name=f"{fig_id}_pdf", relative_path=fig_pdf.relative_to(writer.run_dir).as_posix())
        writer.register_artifact(logical_name=f"{fig_id}_png", relative_path=fig_png.relative_to(writer.run_dir).as_posix())

    # F02 rate match
    _plot(
        "F02_RATE_MATCH_CHECK",
        {"max_abs_err": series["achieved_rate_max_abs_err"].tolist(), "tol": [tol for _ in xs]},
        "Rate match check",
        "max abs rate error",
        {"spike_def_id": pick_sd, "target_rate": pick_tr, "run_id": writer.run_id},
    )

    # F03 branching curves
    ys = {}
    yerr = {}
    for sd in spike_defs:
        sub = t01_df_art[(t01_df_art["spike_def_id"] == sd) & (t01_df_art["target_rate"] == pick_tr)].copy()
        sub = sub.sort_values(["g"], kind="mergesort")
        ys[f"{sd}:b_time"] = sub["b_time"].tolist()
        ys[f"{sd}:b_depth"] = sub["b_depth"].tolist()
        ys[f"{sd}:b_tot"] = sub["b_tot"].tolist()
        if "opt_b_time_ci_low" in sub.columns and "opt_b_time_ci_high" in sub.columns:
            y = sub["b_time"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_b_time_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_b_time_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr[f"{sd}:b_time"] = ((y - lo).tolist(), (hi - y).tolist())
        if "opt_b_depth_ci_low" in sub.columns and "opt_b_depth_ci_high" in sub.columns:
            y = sub["b_depth"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_b_depth_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_b_depth_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr[f"{sd}:b_depth"] = ((y - lo).tolist(), (hi - y).tolist())
        if "opt_b_tot_ci_low" in sub.columns and "opt_b_tot_ci_high" in sub.columns:
            y = sub["b_tot"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_b_tot_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_b_tot_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr[f"{sd}:b_tot"] = ((y - lo).tolist(), (hi - y).tolist())
    _plot(
        "F03_BRANCHING_CURVES",
        ys,
        "Branching curves (rate-matched)",
        "branching value",
        {"target_rate": pick_tr, "run_id": writer.run_id},
        yerr=yerr or None,
    )

    # F04 delta b (triptych: time/depth/tot; legend only shows spike defs)
    ys_by_panel: dict[str, dict[str, list[float]]] = {"delta_b_time": {}, "delta_b_depth": {}, "delta_b_tot": {}}
    yerr_by_panel: dict[str, dict[str, list[float] | tuple[list[float], list[float]]]] = {
        "delta_b_time": {},
        "delta_b_depth": {},
        "delta_b_tot": {},
    }
    for sd in spike_defs:
        sub = t01_df_art[(t01_df_art["spike_def_id"] == sd) & (t01_df_art["target_rate"] == pick_tr)].copy()
        sub = sub.sort_values(["g"], kind="mergesort")
        ys_by_panel["delta_b_time"][sd] = sub["delta_b_time"].tolist()
        ys_by_panel["delta_b_depth"][sd] = sub["delta_b_depth"].tolist()
        ys_by_panel["delta_b_tot"][sd] = sub["delta_b_tot"].tolist()
        if "opt_delta_b_time_ci_low" in sub.columns and "opt_delta_b_time_ci_high" in sub.columns:
            y = sub["delta_b_time"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_delta_b_time_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_delta_b_time_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr_by_panel["delta_b_time"][sd] = ((y - lo).tolist(), (hi - y).tolist())
        if "opt_delta_b_depth_ci_low" in sub.columns and "opt_delta_b_depth_ci_high" in sub.columns:
            y = sub["delta_b_depth"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_delta_b_depth_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_delta_b_depth_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr_by_panel["delta_b_depth"][sd] = ((y - lo).tolist(), (hi - y).tolist())
        if "opt_delta_b_tot_ci_low" in sub.columns and "opt_delta_b_tot_ci_high" in sub.columns:
            y = sub["delta_b_tot"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_delta_b_tot_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_delta_b_tot_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr_by_panel["delta_b_tot"][sd] = ((y - lo).tolist(), (hi - y).tolist())

    fig_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F04_NULL_DELTAB"]))
    fig_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F04_NULL_DELTAB"]))
    triptych_line_plot(
        x=xs,
        ys_by_panel=ys_by_panel,
        title="Delta b vs within-layer permutation null",
        xlabel="gain g",
        out_pdf=fig_pdf,
        out_png=fig_png,
        yerr=yerr_by_panel or None,
        meta={"target_rate": pick_tr, "run_id": writer.run_id},
        provenance={
            "run_id": writer.run_id,
            "model_id": model_id,
            "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
            "spike_def_id": "multiple",
            "target_rate": str(pick_tr),
            "g": "grid",
            "config_hash": writer.config_hash,
        },
    )
    writer.register_artifact(logical_name="F04_NULL_DELTAB_pdf", relative_path=fig_pdf.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="F04_NULL_DELTAB_png", relative_path=fig_png.relative_to(writer.run_dir).as_posix())

    # F05 gstar selection annotated
    gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    if not gstar_path.is_file():
        raise DependencyError("Missing gstar.json dependency")
    gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
    gstar_key = f"{pick_sd}|{pick_tr}"
    if gstar_key not in gstar_obj.get("by_condition", {}):
        raise DependencyError("gstar.json missing selected condition key")
    gstar_val = float(gstar_obj["by_condition"][gstar_key]["gstar"])

    fig_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F05_GSTAR_SELECTION"]))
    fig_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F05_GSTAR_SELECTION"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, series["b_tot"].tolist(), marker="o", label="b_tot")
    ax.axvline(gstar_val, color="red", linestyle="--", label=f"gstar={gstar_val}")
    ax.set_title("gstar selection (b_tot near 1)")
    ax.set_xlabel("gain g")
    ax.set_ylabel("b_tot")
    ax.legend()
    ax.text(0.01, 0.01, f"run_id={writer.run_id}, key={gstar_key}", transform=ax.transAxes, fontsize=7, va="bottom")
    fig.tight_layout()
    fig_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig=fig,
        out_pdf=fig_pdf,
        out_png=fig_png,
        provenance={
            "run_id": writer.run_id,
            "model_id": model_id,
            "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
            "spike_def_id": pick_sd,
            "target_rate": pick_tr,
            "g": "grid",
            "config_hash": writer.config_hash,
        },
        dpi=200,
    )
    plt.close(fig)
    writer.register_artifact(logical_name="F05_GSTAR_SELECTION_pdf", relative_path=fig_pdf.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="F05_GSTAR_SELECTION_png", relative_path=fig_png.relative_to(writer.run_dir).as_posix())

    # F08 spike definition robustness (b_tot)
    ys = {}
    for sd in spike_defs:
        sub = t01_df_art[(t01_df_art["spike_def_id"] == sd) & (t01_df_art["target_rate"] == pick_tr)].copy()
        sub = sub.sort_values(["g"], kind="mergesort")
        ys[sd] = sub["b_tot"].tolist()
    _plot(
        "F08_SPIKEDEF_ROBUST",
        ys,
        "Spike definition robustness",
        "b_tot",
        {"target_rate": pick_tr, "run_id": writer.run_id},
    )

    # F09 chi curves (appendix)
    ys = {}
    yerr = {}
    for sd in spike_defs:
        sub = t01_df_art[(t01_df_art["spike_def_id"] == sd) & (t01_df_art["target_rate"] == pick_tr)].copy()
        sub = sub.sort_values(["g"], kind="mergesort")
        ys[f"{sd}:chi"] = sub["chi"].tolist()
        if "opt_chi_ci_low" in sub.columns and "opt_chi_ci_high" in sub.columns:
            y = sub["chi"].to_numpy(dtype=np.float64, copy=False)
            lo = sub["opt_chi_ci_low"].to_numpy(dtype=np.float64, copy=False)
            hi = sub["opt_chi_ci_high"].to_numpy(dtype=np.float64, copy=False)
            yerr[f"{sd}:chi"] = ((y - lo).tolist(), (hi - y).tolist())
    _plot(
        "F09_CHI_CURVES",
        ys,
        "Susceptibility chi vs gain (rate-matched)",
        "chi",
        {"target_rate": pick_tr, "run_id": writer.run_id},
        yerr=yerr or None,
    )

    # F10 null comparison (appendix): permutation vs circular shift delta_b_tot
    sub = t01_df_art[(t01_df_art["spike_def_id"] == pick_sd) & (t01_df_art["target_rate"] == pick_tr)].copy()
    sub = sub.sort_values(["g"], kind="mergesort")
    if "opt_delta_b_tot_shift" not in sub.columns:
        raise RuntimeError("T01 missing required opt_delta_b_tot_shift column for F10")
    ys = {
        "perm:delta_b_tot": sub["delta_b_tot"].tolist(),
        "circ_shift:delta_b_tot": sub["opt_delta_b_tot_shift"].tolist(),
    }
    yerr = {}
    if "opt_delta_b_tot_ci_low" in sub.columns and "opt_delta_b_tot_ci_high" in sub.columns:
        y = sub["delta_b_tot"].to_numpy(dtype=np.float64, copy=False)
        lo = sub["opt_delta_b_tot_ci_low"].to_numpy(dtype=np.float64, copy=False)
        hi = sub["opt_delta_b_tot_ci_high"].to_numpy(dtype=np.float64, copy=False)
        yerr["perm:delta_b_tot"] = ((y - lo).tolist(), (hi - y).tolist())
    if "opt_delta_b_tot_shift_ci_low" in sub.columns and "opt_delta_b_tot_shift_ci_high" in sub.columns:
        y = sub["opt_delta_b_tot_shift"].to_numpy(dtype=np.float64, copy=False)
        lo = sub["opt_delta_b_tot_shift_ci_low"].to_numpy(dtype=np.float64, copy=False)
        hi = sub["opt_delta_b_tot_shift_ci_high"].to_numpy(dtype=np.float64, copy=False)
        yerr["circ_shift:delta_b_tot"] = ((y - lo).tolist(), (hi - y).tolist())
    _plot(
        "F10_NULL_COMPARE",
        ys,
        "Null comparison: Δb_tot (perm vs circ-shift)",
        "delta b_tot",
        {"spike_def_id": pick_sd, "target_rate": pick_tr, "run_id": writer.run_id},
        yerr=yerr or None,
    )

    # F11 ablations (appendix): delta_b_tot under different interventions at gstar
    pick = t06_df[(t06_df["spike_def_id"] == pick_sd) & (t06_df["target_rate"] == pick_tr)].copy()
    pick = pick[pick["g_condition"] == "gstar"].copy()
    pick = pick.sort_values(["intervention_id"], kind="mergesort")
    if pick.shape[0] <= 0:
        raise RuntimeError("No ablation rows found for F11")

    labels = pick["intervention_id"].astype(str).tolist()
    vals = pick["delta_b_tot"].to_numpy(dtype=np.float64, copy=False).tolist()

    fig_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F11_ABLATIONS"]))
    fig_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F11_ABLATIONS"]))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = list(range(len(labels)))
    ax.bar(xs, vals)
    ax.set_xticks(xs, labels, rotation=30, ha="right")
    ax.set_ylabel("delta_b_tot (vs within-layer perm null)")
    ax.set_title("Ablations at gstar (rate-matched)")
    ax.text(
        0.01,
        0.01,
        f"run_id={writer.run_id}, spike_def_id={pick_sd}, target_rate={pick_tr}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    fig.tight_layout()
    save_figure(
        fig=fig,
        out_pdf=fig_pdf,
        out_png=fig_png,
        provenance={
            "run_id": writer.run_id,
            "model_id": model_id,
            "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
            "spike_def_id": str(pick_sd),
            "target_rate": str(pick_tr),
            "g_condition": "gstar",
            "config_hash": writer.config_hash,
        },
        dpi=200,
    )
    plt.close(fig)
    writer.register_artifact(logical_name="F11_ABLATIONS_pdf", relative_path=fig_pdf.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="F11_ABLATIONS_png", relative_path=fig_png.relative_to(writer.run_dir).as_posix())

    writer.run_record["dependencies"] = [
        str(rr_cal.get("run_id")),
        str(rr_gain_grid.get("run_id")),
        str(rr_gstar.get("run_id")),
        str(rr_rasters.get("run_id")),
        str(rr_nulls.get("run_id")),
    ]
    writer.run_record["conditions"] = [f"{r['spike_def_id']}|{r['target_rate']}|{r['g']}" for r in cond_rows]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 16})
    phase_log.emit("artifacts_written", {"count": 16})
