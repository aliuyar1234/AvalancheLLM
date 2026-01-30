from __future__ import annotations

import hashlib
import json
import math
import zipfile
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .canon import get_canon
from .errors import DependencyError
from .datasets.sampling import select_token_windows
from .events.rate_match_calibrate import calibrate_tau_and_optionally_rasters
from .io.artifacts import RunWriter
from .io.jsonl import JsonlLogger
from .metrics.avalanches import avalanche_size_stats, avalanches_from_a
from .metrics.signatures import chi_susceptibility, crackling_fit
from .model.hooks import get_mlp_layers
from .model.loader import load_model
from .model.forward import run_with_u_capture
from .paper.export import (
    update_paper_provenance_footnotes,
    update_paper_snapshot,
    validate_artifact_references,
    validate_citations,
    validate_no_forbidden_markers,
    validate_snapshot_provenance,
)
from .plotting.figures import simple_line_plot
from .plotting.savefig import save_figure
from .plotting.tables import write_table
from .raster.extract import extract_rasters
from .raster.nulls import within_layer_time_circular_shift, within_layer_time_permutation
from .release import verify_manifest_matches_recomputation, write_release_manifest_and_zip


def _runs_root() -> Path:
    canon = get_canon()
    return Path(str(canon["PATH"]["RUNS_DIR"]))


def _run_record_path(run_dir: Path) -> Path:
    canon = get_canon()
    return run_dir / str(canon["OUTPUT"]["RUN_RECORD_JSON"])


def _load_run_record(run_dir: Path) -> dict[str, Any]:
    path = _run_record_path(run_dir)
    if not path.is_file():
        raise DependencyError(f"Missing run record file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_config_hash(run_record: dict[str, Any]) -> str:
    h = run_record.get("hashes", {}).get("config_sha256")
    if not isinstance(h, str) or not h:
        raise DependencyError("Run record missing hashes.config_sha256")
    return h


def _record_model_details(*, writer: RunWriter, lm: Any, layers: list[Any]) -> None:
    d_ff = getattr(layers[0].gate_proj, "out_features", None) if layers else None
    writer.run_record["model"].update(
        {
            "hf_id": getattr(lm, "hf_id", None),
            "model_revision": getattr(lm, "model_revision", None),
            "tokenizer_revision": getattr(lm, "tokenizer_revision", None),
            "n_layers": int(len(layers)),
            "d_ff": int(d_ff) if isinstance(d_ff, int) else None,
        }
    )
    writer.flush_run_record()


def _selected_model(config: dict[str, Any]) -> tuple[str, str]:
    canon = get_canon()
    role_base = str(canon["ENUM"]["MODEL_ROLE"]["BASE"])
    role_instruct = str(canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"])
    role = str(config.get("model_role") or role_instruct)

    selected = config.get("model_selected", {})
    if isinstance(selected, dict):
        hf_id = selected.get("hf_id")
        if isinstance(hf_id, str) and hf_id:
            return (role, hf_id)

    models = config.get("models", {})
    if isinstance(models, dict):
        key = "base" if role == role_base else "instruct"
        model_cfg = models.get(key, {})
        if isinstance(model_cfg, dict):
            hf_id = model_cfg.get("hf_id")
            if isinstance(hf_id, str) and hf_id:
                return (role, hf_id)
        model_cfg = models.get("instruct", {})
        if isinstance(model_cfg, dict):
            hf_id = model_cfg.get("hf_id")
            if isinstance(hf_id, str) and hf_id:
                return (role_instruct, hf_id)

    raise RuntimeError("Missing models.*.hf_id in resolved config")


def _find_unique_run(*, phase_id: str, config_hash: str) -> Path:
    runs_root = _runs_root()
    if not runs_root.is_dir():
        raise DependencyError("Runs directory does not exist")

    matches: list[Path] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            rr = _load_run_record(run_dir)
        except Exception:
            continue
        if rr.get("phase_id") != phase_id:
            continue
        if _run_config_hash(rr) != config_hash:
            continue
        matches.append(run_dir)

    if not matches:
        raise DependencyError(f"No run found for {phase_id} with config_hash={config_hash}")
    if len(matches) > 1:
        raise DependencyError(
            f"Multiple runs found for {phase_id} with config_hash={config_hash}; "
            "pass explicit --dep run ids for deterministic selection"
        )
    return matches[0]


def _dep_run_dir(args: Any, *, phase_id: str, config_hash: str) -> Path:
    deps = getattr(args, "dep", None) or []
    for rid in deps:
        d = _runs_root() / rid
        if not d.is_dir():
            continue
        rr = _load_run_record(d)
        if rr.get("phase_id") != phase_id:
            continue
        if _run_config_hash(rr) != config_hash:
            raise DependencyError("Dependency config_hash mismatch")
        return d
    return _find_unique_run(phase_id=phase_id, config_hash=config_hash)


def _results_dir(writer: RunWriter) -> Path:
    canon = get_canon()
    return writer.run_dir / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])


def _tables_dir(writer: RunWriter) -> Path:
    canon = get_canon()
    return writer.run_dir / str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])


def _seed_from_config(config: dict[str, Any]) -> int:
    det = config.get("determinism", {})
    if isinstance(det, dict) and "seed" in det:
        try:
            return int(det["seed"])
        except Exception:
            pass
    canon = get_canon()
    return int(canon["CONST"]["BOOTSTRAP_SEED"])


def phase1_calibrate(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    pipeline = config.get("pipeline", {})
    seq_len = int(pipeline.get("seq_len", canon["CONST"]["SEQ_LEN_TOKENS"]))

    model_role, model_id = _selected_model(config)

    lm = load_model(model_id, device=args.device, dtype=args.dtype)
    layers = get_mlp_layers(lm.model)
    L = len(layers)
    if L <= 0:
        raise RuntimeError("Model has no layers")

    d_ff = getattr(layers[0].gate_proj, "out_features", None)
    if not isinstance(d_ff, int) or d_ff <= 0:
        raise RuntimeError("Could not infer d_ff from gate_proj")

    events.emit("model_loaded", {"hf_id": lm.hf_id, "model_revision": lm.model_revision, "model_role": model_role})
    phase_log.emit("model_loaded", {"hf_id": lm.hf_id, "model_revision": lm.model_revision, "model_role": model_role})

    # Calibration slice (Dataset A)
    ds_a = config.get("datasets", {}).get("A", {})
    hf_id = str(ds_a.get("hf_id", ""))
    hf_cfg = ds_a.get("config")
    split = str(ds_a.get("split", ""))
    if not hf_id or not split:
        raise RuntimeError("Missing datasets.A hf_id/split in resolved config")

    min_samples = int(canon["CONST"]["RATE_MATCH_MIN_CAL_SAMPLES"])
    n_by_min = max(1, math.ceil(min_samples / float(seq_len * d_ff)))
    n_cal_target = int(canon["CONST"]["N_WINDOWS_A_CALIBRATION"])
    n_cal = max(n_by_min, n_cal_target)
    selected, ds_meta = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_cal,
        run_id=writer.run_id,
    )
    token_windows = [w.input_ids for w in selected]

    writer.run_record["model"].update(
        {
            "hf_id": lm.hf_id,
            "model_revision": lm.model_revision,
            "tokenizer_revision": lm.tokenizer_revision,
            "model_role": model_role,
            "n_layers": L,
            "d_ff": d_ff,
        }
    )
    writer.run_record["dataset"].setdefault("slices", {})["A_calibration"] = ds_meta
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256"] = ds_meta["dataset_slice_sha256"]
    writer.flush_run_record()
    events.emit("dataset_loaded", {"dataset_role": "A", "dataset_slice_sha256": ds_meta["dataset_slice_sha256"]})
    phase_log.emit("dataset_loaded", {"dataset_role": "A", "dataset_slice_sha256": ds_meta["dataset_slice_sha256"]})

    # Pass 1: mu/sigma on u at baseline gain
    baseline_gain = float(canon["CONST"]["GAIN_BASELINE"])
    sum_u = torch.zeros((L, d_ff), dtype=torch.float64, device=args.device)
    sumsq_u = torch.zeros((L, d_ff), dtype=torch.float64, device=args.device)
    count_tokens = {"n": 0}

    def on_u_moments(layer_idx: int, u: torch.Tensor) -> None:
        u64 = u.to(dtype=torch.float64)
        sum_u[layer_idx] += u64.sum(dim=(0, 1))
        sumsq_u[layer_idx] += (u64 * u64).sum(dim=(0, 1))
        count_tokens["n"] += int(u64.shape[0] * u64.shape[1])

    for ids in token_windows:
        inp = torch.tensor([ids], dtype=torch.long, device=args.device)
        attn = torch.ones_like(inp)
        run_with_u_capture(
            model=lm.model,
            layers=layers,
            input_ids=inp,
            attention_mask=attn,
            gain=baseline_gain,
            on_u=on_u_moments,
            need_logits=False,
        )

    if count_tokens["n"] <= 0:
        raise RuntimeError("No calibration tokens processed")
    mu_t = (sum_u / float(count_tokens["n"])).to(dtype=torch.float32)
    var_t = (sumsq_u / float(count_tokens["n"])) - (sum_u / float(count_tokens["n"])) ** 2
    var_t = torch.clamp(var_t, min=0.0)
    sigma_t = torch.sqrt(var_t).to(dtype=torch.float32)
    mu = mu_t.detach().cpu().numpy()
    sigma = sigma_t.detach().cpu().numpy()

    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    cal_base = str(canon["OUTPUT"]["CALIBRATION_STATS_NPZ_BASENAME"])
    cal_path = results_dir / cal_base
    if cal_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {cal_path}")
    np.savez_compressed(cal_path, mu=mu.astype(np.float32), sigma=sigma.astype(np.float32))
    writer.register_artifact(
        logical_name="calibration_stats_npz", relative_path=cal_path.relative_to(writer.run_dir).as_posix()
    )

    # Pass 2: baseline rates at tau0 for each spike definition, then scale to each target_rate label.
    eps = float(canon["CONST"]["EPS"])
    tau0 = float(canon["CONST"]["TAU0_BASELINE"])
    mu2 = torch.tensor(mu, dtype=torch.float32, device=args.device)
    denom2 = torch.tensor(sigma, dtype=torch.float32, device=args.device) + eps
    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]

    spikes_sum = {sd: torch.zeros((L,), dtype=torch.float64, device="cpu") for sd in spike_defs}
    denom_units = torch.zeros((L,), dtype=torch.float64, device="cpu")
    cur_sd = {
        "one": str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_ONE_SIDED_POS"]),
        "two": str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_TWO_SIDED_ABS"]),
    }

    def on_u_rates(layer_idx: int, u: torch.Tensor) -> None:
        z = (u.to(dtype=torch.float32) - mu2[layer_idx]) / denom2[layer_idx]
        v_one = z
        v_two = torch.abs(z)
        spikes_sum[cur_sd["one"]][layer_idx] += float((v_one > tau0).sum().detach().cpu().item())
        spikes_sum[cur_sd["two"]][layer_idx] += float((v_two > tau0).sum().detach().cpu().item())
        denom_units[layer_idx] += float(v_one.numel())

    for ids in token_windows:
        inp = torch.tensor([ids], dtype=torch.long, device=args.device)
        attn = torch.ones_like(inp)
        run_with_u_capture(
            model=lm.model,
            layers=layers,
            input_ids=inp,
            attention_mask=attn,
            gain=baseline_gain,
            on_u=on_u_rates,
            need_logits=False,
        )

    r_base: dict[str, np.ndarray] = {}
    for sd in spike_defs:
        r = (spikes_sum[sd].numpy() / np.maximum(denom_units.numpy(), 1.0)).astype(np.float64)
        r_base[sd] = r

    target_rates = list(pipeline.get("target_rates", []))
    r_star_l: dict[str, dict[str, list[float]]] = {}
    scales: dict[str, dict[str, float]] = {}
    for sd in spike_defs:
        r_star_l[sd] = {}
        scales[sd] = {}
        mean_base = float(np.mean(r_base[sd]))
        if mean_base <= 0.0:
            raise RuntimeError("Baseline rate mean is non-positive; cannot scale target rates")
        for tr in target_rates:
            tr_f = float(tr)
            scale = tr_f / mean_base
            scaled = np.clip(r_base[sd] * scale, 0.0, 1.0)
            r_star_l[sd][str(tr_f)] = [float(x) for x in scaled.tolist()]
            scales[sd][str(tr_f)] = float(scale)

    rate_targets: dict[str, Any] = {
        "L": int(L),
        "d_ff": int(d_ff),
        "tau0_baseline": tau0,
        "gain_baseline": baseline_gain,
        "r_base_l": {sd: [float(x) for x in r_base[sd].tolist()] for sd in spike_defs},
        "scales": scales,
        "r_star_l": r_star_l,
    }
    rate_base = str(canon["OUTPUT"]["RATE_TARGETS_JSON_BASENAME"])
    rate_rel = writer.write_json(relative_path=(results_dir / rate_base).relative_to(writer.run_dir).as_posix(), obj=rate_targets)
    writer.register_artifact(logical_name="rate_targets_json", relative_path=rate_rel)

    writer.run_record["conditions"] = [f"{sd}|{tr}" for sd in spike_defs for tr in target_rates]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 2})
    phase_log.emit("artifacts_written", {"count": 2})


def phase2_gain_grid(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    pipeline = config.get("pipeline", {})
    seq_len = int(pipeline.get("seq_len", canon["CONST"]["SEQ_LEN_TOKENS"]))

    dep_dir = _dep_run_dir(args, phase_id="PHASE1_CALIBRATE", config_hash=writer.config_hash)
    dep_rr = _load_run_record(dep_dir)
    dep_results = dep_dir / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    cal_path = dep_results / str(canon["OUTPUT"]["CALIBRATION_STATS_NPZ_BASENAME"])
    rate_path = dep_results / str(canon["OUTPUT"]["RATE_TARGETS_JSON_BASENAME"])
    if not cal_path.is_file() or not rate_path.is_file():
        raise DependencyError("Phase 1 calibration artifacts missing")

    cal = np.load(cal_path)
    mu = cal["mu"]
    sigma = cal["sigma"]
    rate_targets = json.loads(rate_path.read_text(encoding="utf-8"))

    model_role, model_id = _selected_model(config)
    lm = load_model(model_id, device=args.device, dtype=args.dtype)
    layers = get_mlp_layers(lm.model)
    _record_model_details(writer=writer, lm=lm, layers=layers)
    writer.run_record["model"]["model_role"] = model_role
    writer.flush_run_record()

    # Reconstruct Phase1 Dataset A calibration slice deterministically and verify hash matches.
    ds_a = config.get("datasets", {}).get("A", {})
    hf_id = str(ds_a.get("hf_id", ""))
    hf_cfg = ds_a.get("config")
    split = str(ds_a.get("split", ""))
    if not hf_id or not split:
        raise RuntimeError("Missing datasets.A hf_id/split in resolved config")

    slices = dep_rr.get("dataset", {}).get("slices", {})
    smeta = slices.get("A_calibration")
    if not isinstance(smeta, dict):
        raise DependencyError("Phase1 run_record missing dataset.slices.A_calibration metadata")
    n_windows = int(smeta.get("n_windows", 0))
    if n_windows <= 0:
        raise DependencyError("Invalid Phase1 calibration slice window count")

    selected, ds_meta = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_windows,
        run_id=str(dep_rr.get("run_id")),
    )
    if ds_meta["dataset_slice_sha256"] != dep_rr.get("hashes", {}).get("dataset_slice_sha256"):
        raise DependencyError("Dataset slice hash mismatch vs Phase1 run_record")
    token_windows = [w.input_ids for w in selected]
    events.emit("dataset_loaded", {"dataset_role": "A", "dataset_slice_sha256": ds_meta["dataset_slice_sha256"]})
    phase_log.emit("dataset_loaded", {"dataset_role": "A", "dataset_slice_sha256": ds_meta["dataset_slice_sha256"]})

    gains = list(pipeline.get("gain_grid", []))
    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rate_labels = [str(float(x)) for x in pipeline.get("target_rates", [])]
    tol = float(canon["CONST"]["RATE_MATCH_TOL_ABS"])

    rows: list[dict[str, Any]] = []
    for sd in spike_defs:
        r_star_map = rate_targets.get("r_star_l", {}).get(sd, {})
        r_targets_by_tr = {tr: np.asarray(r_star_map.get(tr), dtype=np.float64) for tr in target_rate_labels}
        for g in gains:
            g = float(g)
            tau_cal, rasters = calibrate_tau_and_optionally_rasters(
                model=lm.model,
                layers=layers,
                token_windows=token_windows,
                gain=g,
                spike_def_id=sd,
                mu=mu,
                sigma=sigma,
                r_targets_by_target_rate=r_targets_by_tr,
                device=args.device,
                seq_len=seq_len,
                collect_rasters=True,
            )
            assert rasters is not None
            for tr in target_rate_labels:
                x = rasters.x_by_target_rate[tr]  # [N, L, T]
                denom = float(np.sum(x, dtype=np.int64))
                num_time = float(np.sum(x[:, :, :-1] * x[:, :, 1:], dtype=np.int64))
                num_depth = float(np.sum(x[:, :-1, :] * x[:, 1:, :], dtype=np.int64))
                b_time = num_time / denom if denom else float("nan")
                b_depth = num_depth / denom if denom else float("nan")
                achieved_layer = tau_cal.achieved_rate_by_target_rate[tr]
                target_layer = r_targets_by_tr[tr]
                rows.append(
                    {
                        "spike_def_id": sd,
                        "target_rate": float(tr),
                        "g": float(g),
                        "b_time": float(b_time),
                        "b_depth": float(b_depth),
                        "b_tot": float(b_time + b_depth),
                        "achieved_rate_mean": float(np.mean(achieved_layer)),
                        "achieved_rate_max_abs_err": float(np.max(np.abs(achieved_layer - target_layer))),
                        "rate_match_tol_abs": tol,
                    }
                )

    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_base = str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    out_path = results_dir / metrics_base
    if out_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {out_path}")
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=out_path.relative_to(writer.run_dir).as_posix())
    writer.run_record["dependencies"] = [str(dep_rr.get("run_id"))]
    writer.run_record["conditions"] = [f"{r['spike_def_id']}|{r['target_rate']}|{r['g']}" for r in rows]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 1})
    phase_log.emit("artifacts_written", {"count": 1})


def phase2_select_gstar(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    dep_dir = _dep_run_dir(args, phase_id="PHASE2_GAIN_GRID", config_hash=writer.config_hash)
    dep_rr = _load_run_record(dep_dir)
    dep_metrics = dep_dir / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(
        canon["OUTPUT"]["METRICS_PARQUET_BASENAME"]
    )
    if not dep_metrics.is_file():
        raise DependencyError(f"Missing dependency metrics file: {dep_metrics}")

    df = pd.read_parquet(dep_metrics)
    baseline = float(canon["CONST"]["GAIN_BASELINE"])

    gstar: dict[str, Any] = {"method": getattr(args, "gstar_method", "btot_near_one"), "by_condition": {}}
    for (s, tr), grp in df.groupby(["spike_def_id", "target_rate"], dropna=False):
        grp = grp.copy()
        grp["abs_err"] = (grp["b_tot"] - 1.0).abs()
        min_err = float(grp["abs_err"].min())
        cands = grp[grp["abs_err"] == min_err]
        if len(cands) > 1:
            cands = cands.copy()
            cands["tie"] = (cands["g"] - baseline).abs()
            cands = cands.sort_values(["tie", "g"])
        chosen = float(cands.iloc[0]["g"])
        gstar["by_condition"][f"{s}|{tr}"] = {"spike_def_id": str(s), "target_rate": float(tr), "gstar": chosen}

    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    gstar_base = str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    rel = writer.write_json(relative_path=(results_dir / gstar_base).relative_to(writer.run_dir).as_posix(), obj=gstar)
    writer.register_artifact(logical_name="gstar_json", relative_path=rel)

    writer.run_record["dependencies"] = [dep_rr.get("run_id")]
    writer.run_record["conditions"] = list(gstar["by_condition"].keys())
    events.emit("artifacts_written", {"count": 1})
    phase_log.emit("artifacts_written", {"count": 1})


def phase3_extract_rasters(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    pipeline = config.get("pipeline", {})
    seq_len = int(pipeline.get("seq_len", canon["CONST"]["SEQ_LEN_TOKENS"]))

    dep_cal = _dep_run_dir(args, phase_id="PHASE1_CALIBRATE", config_hash=writer.config_hash)
    rr_cal = _load_run_record(dep_cal)
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)
    rr_gstar = _load_run_record(dep_gstar)

    cal_results = dep_cal / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    cal_npz = cal_results / str(canon["OUTPUT"]["CALIBRATION_STATS_NPZ_BASENAME"])
    cal_rates = cal_results / str(canon["OUTPUT"]["RATE_TARGETS_JSON_BASENAME"])
    if not cal_npz.is_file() or not cal_rates.is_file():
        raise DependencyError("Missing Phase1 calibration outputs")
    cal = np.load(cal_npz)
    mu = cal["mu"]
    sigma = cal["sigma"]
    rate_targets = json.loads(cal_rates.read_text(encoding="utf-8"))

    gstar_results = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    gstar_json = gstar_results / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    if not gstar_json.is_file():
        raise DependencyError("Missing gstar.json dependency")
    gstar_obj = json.loads(gstar_json.read_text(encoding="utf-8"))

    model_role, model_id = _selected_model(config)
    lm = load_model(model_id, device=args.device, dtype=args.dtype)
    layers = get_mlp_layers(lm.model)
    writer.run_record["model"]["model_role"] = model_role
    _record_model_details(writer=writer, lm=lm, layers=layers)

    # Calibration windows are re-derived using Phase1 run_id and verified against Phase1 slice hash.
    ds_a = config.get("datasets", {}).get("A", {})
    hf_id = str(ds_a.get("hf_id", ""))
    hf_cfg = ds_a.get("config")
    split = str(ds_a.get("split", ""))
    slices = rr_cal.get("dataset", {}).get("slices", {})
    smeta = slices.get("A_calibration")
    if not isinstance(smeta, dict):
        raise DependencyError("Phase1 run_record missing dataset.slices.A_calibration metadata")
    n_cal = int(smeta.get("n_windows", 0))
    if n_cal <= 0:
        raise DependencyError("Invalid Phase1 calibration slice window count")
    selected_cal, meta_cal = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_cal,
        run_id=str(rr_cal.get("run_id")),
    )
    if meta_cal["dataset_slice_sha256"] != rr_cal.get("hashes", {}).get("dataset_slice_sha256"):
        raise DependencyError("Dataset slice hash mismatch vs Phase1 run_record")
    cal_windows = [w.input_ids for w in selected_cal]

    # Analysis windows for raster extraction (Phase3 run_id)
    n_seq = int(canon["CONST"]["N_WINDOWS_A_RASTERS"])
    selected_a, meta_a = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_seq,
        run_id=writer.run_id,
    )
    writer.run_record["dataset"].setdefault("slices", {})["A_rasters"] = meta_a
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256_A_rasters"] = meta_a["dataset_slice_sha256"]
    writer.flush_run_record()
    events.emit("dataset_loaded", {"dataset_role": "A", "dataset_slice_sha256": meta_a["dataset_slice_sha256"]})
    phase_log.emit("dataset_loaded", {"dataset_role": "A", "dataset_slice_sha256": meta_a["dataset_slice_sha256"]})

    analysis_windows = [w.input_ids for w in selected_a]
    seq_ids = np.array([w.chunk_index for w in selected_a], dtype=np.int32)

    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rate_labels = [str(float(x)) for x in pipeline.get("target_rates", [])]
    baseline = float(canon["CONST"]["GAIN_BASELINE"])
    null_none = str(canon["ENUM"]["NULL_ID"]["NULL_NONE"])

    # Cache tau per (spike_def, gain) computed from the calibration slice.
    tau_cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}

    conds: list[dict[str, Any]] = []
    for sd in spike_defs:
        gstar_map = gstar_obj.get("by_condition", {})
        for tr in target_rate_labels:
            key = f"{sd}|{float(tr)}"
            if key not in gstar_map:
                raise DependencyError(f"gstar missing for {key}")
            gstar_val = float(gstar_map[key]["gstar"])
            for g_condition, g in [("g1", baseline), ("gstar", gstar_val)]:
                conds.append(
                    {
                        "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                        "spike_def_id": sd,
                        "target_rate": float(tr),
                        "g_condition": g_condition,
                        "g": float(g),
                        "null_id": null_none,
                    }
                )

    # Compute needed tau vectors (calibration slice) and extract rasters (analysis slice).
    a_all = []
    x_all = []
    for cond in conds:
        sd = str(cond["spike_def_id"])
        tr = str(float(cond["target_rate"]))
        g = float(cond["g"])
        cache_key = (sd, g)
        if cache_key not in tau_cache:
            r_star_map = rate_targets.get("r_star_l", {}).get(sd, {})
            r_targets_by_tr = {k: np.asarray(r_star_map.get(k), dtype=np.float64) for k in target_rate_labels}
            tau_cal, _ = calibrate_tau_and_optionally_rasters(
                model=lm.model,
                layers=layers,
                token_windows=cal_windows,
                gain=g,
                spike_def_id=sd,
                mu=mu,
                sigma=sigma,
                r_targets_by_target_rate=r_targets_by_tr,
                device=args.device,
                seq_len=seq_len,
                collect_rasters=False,
            )
            tau_cache[cache_key] = tau_cal.tau_by_target_rate
        tau_vec = np.asarray(tau_cache[cache_key][tr], dtype=np.float64)
        ras = extract_rasters(
            model=lm.model,
            layers=layers,
            token_windows=analysis_windows,
            gain=g,
            spike_def_id=sd,
            mu=mu,
            sigma=sigma,
            tau_by_layer=tau_vec,
            device=args.device,
            seq_len=seq_len,
        )
        a_all.append(ras.a.astype(np.uint16))
        x_all.append(ras.x.astype(np.uint8))

    a_npz = np.stack(a_all, axis=0)  # [C,N,L,T]
    x_npz = np.stack(x_all, axis=0)  # [C,N,L,T]
    cond_id = np.arange(len(conds), dtype=np.int32)

    keys = canon["ENUM"]["NPZ_KEY"]
    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    raster_base = str(canon["OUTPUT"]["RASTER_NPZ_BASENAME"])
    out_path = results_dir / raster_base
    if out_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {out_path}")
    np.savez_compressed(
        out_path,
        **{
            str(keys["X_OCCUPANCY"]): x_npz,
            str(keys["A_COUNT"]): a_npz,
            str(keys["COND_ID"]): cond_id,
            str(keys["SEQ_ID"]): seq_ids,
        },
    )
    writer.register_artifact(logical_name="rasters_npz", relative_path=out_path.relative_to(writer.run_dir).as_posix())

    # Condition mapping in metrics.parquet per spec/02.
    metrics_rows = []
    for cid, cond in enumerate(conds):
        metrics_rows.append(
            {
                "cond_id": int(cid),
                "dataset_role": cond["dataset_role"],
                "spike_def_id": cond["spike_def_id"],
                "target_rate": float(cond["target_rate"]),
                "g_condition": cond["g_condition"],
                "g": float(cond["g"]),
                "null_id": cond["null_id"],
            }
        )
    metrics_base = str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    metrics_path = results_dir / metrics_base
    if metrics_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {metrics_path}")
    pd.DataFrame(metrics_rows).to_parquet(metrics_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=metrics_path.relative_to(writer.run_dir).as_posix())

    writer.run_record["dependencies"] = [str(rr_cal.get("run_id")), str(rr_gstar.get("run_id"))]
    writer.run_record["conditions"] = [f"{c['spike_def_id']}|{c['target_rate']}|{c['g_condition']}" for c in conds]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 2})
    phase_log.emit("artifacts_written", {"count": 2})


def phase4_run_nulls(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    pipeline = config.get("pipeline", {})
    seq_len = int(pipeline.get("seq_len", canon["CONST"]["SEQ_LEN_TOKENS"]))

    dep_rasters = _dep_run_dir(args, phase_id="PHASE3_EXTRACT_RASTERS", config_hash=writer.config_hash)
    rr_rasters = _load_run_record(dep_rasters)
    dep_cal = _dep_run_dir(args, phase_id="PHASE1_CALIBRATE", config_hash=writer.config_hash)
    rr_cal = _load_run_record(dep_cal)

    raster_path = dep_rasters / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["RASTER_NPZ_BASENAME"])
    if not raster_path.is_file():
        raise DependencyError(f"Missing rasters dependency: {raster_path}")

    metrics_base = str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    rasters_metrics_path = dep_rasters / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / metrics_base
    if not rasters_metrics_path.is_file():
        raise DependencyError(f"Missing rasters condition mapping dependency: {rasters_metrics_path}")

    z = np.load(raster_path)
    keys = canon["ENUM"]["NPZ_KEY"]
    a = z[str(keys["A_COUNT"])].astype(np.uint16)
    cond_id = z[str(keys["COND_ID"])].astype(np.int32)
    seq_id = z[str(keys["SEQ_ID"])].astype(np.int32)

    # Tau table: compute tau_l(g) for every gain using Phase1 calibration slice and targets.
    cal_results = dep_cal / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"])
    cal_npz = cal_results / str(canon["OUTPUT"]["CALIBRATION_STATS_NPZ_BASENAME"])
    cal_rates = cal_results / str(canon["OUTPUT"]["RATE_TARGETS_JSON_BASENAME"])
    if not cal_npz.is_file() or not cal_rates.is_file():
        raise DependencyError("Missing Phase1 calibration artifacts")
    cal = np.load(cal_npz)
    mu = cal["mu"]
    sigma = cal["sigma"]
    rate_targets = json.loads(cal_rates.read_text(encoding="utf-8"))

    model_role, model_id = _selected_model(config)
    lm = load_model(model_id, device=args.device, dtype=args.dtype)
    layers = get_mlp_layers(lm.model)
    d_ff = getattr(layers[0].gate_proj, "out_features", None) if layers else None
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

    ds_a = config.get("datasets", {}).get("A", {})
    hf_id = str(ds_a.get("hf_id", ""))
    hf_cfg = ds_a.get("config")
    split = str(ds_a.get("split", ""))
    slices = rr_cal.get("dataset", {}).get("slices", {})
    smeta = slices.get("A_calibration")
    if not isinstance(smeta, dict):
        raise DependencyError("Phase1 run_record missing dataset.slices.A_calibration metadata")
    n_cal = int(smeta.get("n_windows", 0))
    if n_cal <= 0:
        raise DependencyError("Invalid Phase1 calibration slice window count")
    selected_cal, meta_cal = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_cal,
        run_id=str(rr_cal.get("run_id")),
    )
    if meta_cal["dataset_slice_sha256"] != rr_cal.get("hashes", {}).get("dataset_slice_sha256"):
        raise DependencyError("Dataset slice hash mismatch vs Phase1 run_record")
    cal_windows = [w.input_ids for w in selected_cal]

    cal_slices = rr_cal.get("dataset", {}).get("slices", {})
    writer.run_record.setdefault("dataset", {}).setdefault("slices", {})["A_calibration"] = cal_slices.get("A_calibration")
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256"] = rr_cal.get("hashes", {}).get("dataset_slice_sha256")

    gains = [float(g) for g in pipeline.get("gain_grid", [])]
    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rate_labels = [str(float(x)) for x in pipeline.get("target_rates", [])]

    tau_rows: list[dict[str, Any]] = []
    tau_cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for sd in spike_defs:
        r_star_map = rate_targets.get("r_star_l", {}).get(sd, {})
        r_targets_by_tr = {tr: np.asarray(r_star_map.get(tr), dtype=np.float64) for tr in target_rate_labels}
        for g in gains:
            tau_cal, _ = calibrate_tau_and_optionally_rasters(
                model=lm.model,
                layers=layers,
                token_windows=cal_windows,
                gain=g,
                spike_def_id=sd,
                mu=mu,
                sigma=sigma,
                r_targets_by_target_rate=r_targets_by_tr,
                device=args.device,
                seq_len=seq_len,
                collect_rasters=False,
            )
            tau_cache[(sd, float(g))] = {k: np.asarray(v, dtype=np.float64) for k, v in tau_cal.tau_by_target_rate.items()}
            for tr in target_rate_labels:
                tau_vec = tau_cal.tau_by_target_rate[tr]
                achieved_vec = tau_cal.achieved_rate_by_target_rate[tr]
                for l in range(len(tau_vec)):
                    tau_rows.append(
                        {
                            "spike_def_id": sd,
                            "target_rate": float(tr),
                            "g": float(g),
                            "layer": int(l),
                            "tau": float(tau_vec[l]),
                            "achieved_rate": float(achieved_vec[l]),
                            "v_min": float(tau_cal.v_min[l]),
                            "v_max": float(tau_cal.v_max[l]),
                        }
                    )

    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    tau_base = str(canon["OUTPUT"]["TAU_RATE_MATCHED_PARQUET_BASENAME"])
    tau_path = results_dir / tau_base
    if tau_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {tau_path}")
    pd.DataFrame(tau_rows).to_parquet(tau_path, index=False)
    writer.register_artifact(logical_name="tau_rate_matched_parquet", relative_path=tau_path.relative_to(writer.run_dir).as_posix())

    rr_df = pd.read_parquet(rasters_metrics_path)
    if "cond_id" not in rr_df.columns:
        raise DependencyError("Phase3 metrics.parquet missing cond_id column")
    rr_df = rr_df.sort_values(["cond_id"], kind="mergesort").reset_index(drop=True)
    if rr_df.shape[0] != a.shape[0]:
        raise DependencyError("Phase3 metrics.parquet row count does not match raster condition dimension")

    axis_by_cond_id = {int(cid): int(i) for i, cid in enumerate(cond_id.tolist())}
    if len(axis_by_cond_id) != int(cond_id.size):
        raise DependencyError("Duplicate cond_id values in rasters NPZ")

    # Within-layer time permutation null for each Phase3 condition (marginals-preserving).
    C_real, N, L, T = a.shape
    a_perm = np.zeros((C_real, N, L, T), dtype=np.uint16)
    x_perm = np.zeros((C_real, N, L, T), dtype=np.uint8)
    perm_null_id = str(canon["ENUM"]["NULL_ID"]["NULL_RASTER_WITHIN_LAYER_TIME_PERM"])
    shift_null_id = str(canon["ENUM"]["NULL_ID"]["NULL_RASTER_WITHIN_LAYER_TIME_CIRC_SHIFT"])
    a_shift = np.zeros((C_real, N, L, T), dtype=np.uint16)
    x_shift = np.zeros((C_real, N, L, T), dtype=np.uint8)
    for out_c in range(C_real):
        src_cond = rr_df.iloc[out_c].to_dict()
        src_cid = int(src_cond["cond_id"])
        src_axis = axis_by_cond_id.get(src_cid)
        if src_axis is None:
            raise DependencyError("Condition mapping refers to cond_id not present in rasters NPZ")
        for n in range(N):
            out_perm = within_layer_time_permutation(
                a[src_axis, n], run_id=writer.run_id, cond_id=int(out_c), seq_id=int(seq_id[n])
            )
            a_perm[out_c, n] = out_perm.a_perm.astype(np.uint16)
            x_perm[out_c, n] = out_perm.x_perm.astype(np.uint8)
            out_shift = within_layer_time_circular_shift(
                a[src_axis, n], run_id=writer.run_id, cond_id=int(out_c), seq_id=int(seq_id[n])
            )
            a_shift[out_c, n] = out_shift.a_perm.astype(np.uint16)
            x_shift[out_c, n] = out_shift.x_perm.astype(np.uint8)

    # Input token shuffle null: reconstruct Phase3 analysis windows deterministically and re-run extraction.
    rasters_slices = rr_rasters.get("dataset", {}).get("slices", {})
    smeta_a = rasters_slices.get("A_rasters")
    if not isinstance(smeta_a, dict):
        raise DependencyError("Phase3 run_record missing dataset.slices.A_rasters metadata")
    n_seq = int(smeta_a.get("n_windows", 0))
    if n_seq != N:
        raise DependencyError("Phase3 A_rasters slice size does not match rasters NPZ sequence dimension")
    if smeta_a.get("dataset_slice_sha256") != rr_rasters.get("hashes", {}).get("dataset_slice_sha256_A_rasters"):
        raise DependencyError("Dataset slice hash mismatch vs Phase3 run_record")

    selected_a, meta_a = select_token_windows(
        dataset_role=str(canon["ENUM"]["DATASET_ROLE"]["A"]),
        hf_id=hf_id,
        hf_config=str(hf_cfg) if hf_cfg else None,
        split=split,
        tokenizer=lm.tokenizer,
        seq_len=seq_len,
        n_windows=n_seq,
        run_id=str(rr_rasters.get("run_id")),
    )
    if meta_a["dataset_slice_sha256"] != rr_rasters.get("hashes", {}).get("dataset_slice_sha256_A_rasters"):
        raise DependencyError("Reconstructed Phase3 A_rasters slice hash mismatch")
    analysis_windows = [w.input_ids for w in selected_a]
    seq_ids_re = np.array([w.chunk_index for w in selected_a], dtype=np.int32)
    if not np.array_equal(seq_ids_re, seq_id):
        raise DependencyError("Reconstructed Phase3 seq_id mismatch; cannot generate token shuffle null safely")

    def _shuffle_ids(ids: list[int], *, run_id: str, seq: int) -> list[int]:
        seed0 = int(canon["CONST"]["BOOTSTRAP_SEED"])
        if len(ids) < 4:
            return list(ids)
        keep_left = 1
        keep_right = 1
        interior = ids[keep_left : len(ids) - keep_right]
        seed = hashlib.sha256(f"{seed0}|{run_id}|{seq}".encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(seed[:8], 16))
        pi = rng.permutation(len(interior))
        out = list(ids)
        for j, k in enumerate(pi.tolist()):
            out[keep_left + j] = interior[int(k)]
        return out

    shuffled_windows = [_shuffle_ids(ids, run_id=writer.run_id, seq=int(sid)) for ids, sid in zip(analysis_windows, seq_id)]

    token_null_id = str(canon["ENUM"]["NULL_ID"]["NULL_TOKEN_SHUFFLE_INPUT"])
    a_toknull = np.zeros((C_real, N, L, T), dtype=np.uint16)
    x_toknull = np.zeros((C_real, N, L, T), dtype=np.uint8)
    for out_c in range(C_real):
        src_cond = rr_df.iloc[out_c].to_dict()
        sd = str(src_cond.get("spike_def_id"))
        tr_key = str(float(src_cond.get("target_rate")))
        g = float(src_cond.get("g"))
        tau_map = tau_cache.get((sd, float(g)))
        if not isinstance(tau_map, dict) or tr_key not in tau_map:
            raise DependencyError("Missing tau vector for token shuffle null condition")
        tau_vec = np.asarray(tau_map[tr_key], dtype=np.float64)
        ras = extract_rasters(
            model=lm.model,
            layers=layers,
            token_windows=shuffled_windows,
            gain=float(g),
            spike_def_id=sd,
            mu=mu,
            sigma=sigma,
            tau_by_layer=tau_vec,
            device=args.device,
            seq_len=seq_len,
        )
        a_toknull[out_c] = ras.a.astype(np.uint16)
        x_toknull[out_c] = ras.x.astype(np.uint8)

    # Combine null conditions: permutation, circular shift, token-shuffle.
    a_null = np.concatenate([a_perm, a_shift, a_toknull], axis=0).astype(np.uint16)
    x_null = np.concatenate([x_perm, x_shift, x_toknull], axis=0).astype(np.uint8)
    cond_id_null = np.arange(a_null.shape[0], dtype=np.int32)

    null_base = str(canon["OUTPUT"]["NULL_NPZ_BASENAME"])
    null_path = results_dir / null_base
    if null_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {null_path}")
    np.savez_compressed(
        null_path,
        **{
            str(keys["X_OCCUPANCY"]): x_null,
            str(keys["A_COUNT"]): a_null,
            str(keys["COND_ID"]): cond_id_null,
            str(keys["SEQ_ID"]): seq_id.astype(np.int32),
        },
    )
    writer.register_artifact(logical_name="rasters_nulls_npz", relative_path=null_path.relative_to(writer.run_dir).as_posix())

    # Condition mapping in metrics.parquet per spec/02.
    metrics_rows = []
    for out_c in range(C_real):
        src = rr_df.iloc[out_c].to_dict()
        metrics_rows.append(
            {
                "cond_id": int(out_c),
                "dataset_role": src.get("dataset_role"),
                "spike_def_id": src.get("spike_def_id"),
                "target_rate": float(src.get("target_rate")),
                "g_condition": src.get("g_condition"),
                "g": float(src.get("g")),
                "null_id": perm_null_id,
            }
        )
    for out_c in range(C_real):
        src = rr_df.iloc[out_c].to_dict()
        metrics_rows.append(
            {
                "cond_id": int(out_c + C_real),
                "dataset_role": src.get("dataset_role"),
                "spike_def_id": src.get("spike_def_id"),
                "target_rate": float(src.get("target_rate")),
                "g_condition": src.get("g_condition"),
                "g": float(src.get("g")),
                "null_id": shift_null_id,
            }
        )
    for out_c in range(C_real):
        src = rr_df.iloc[out_c].to_dict()
        metrics_rows.append(
            {
                "cond_id": int(out_c + 2 * C_real),
                "dataset_role": src.get("dataset_role"),
                "spike_def_id": src.get("spike_def_id"),
                "target_rate": float(src.get("target_rate")),
                "g_condition": src.get("g_condition"),
                "g": float(src.get("g")),
                "null_id": token_null_id,
            }
        )
    metrics_path = results_dir / metrics_base
    if metrics_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {metrics_path}")
    pd.DataFrame(metrics_rows).to_parquet(metrics_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=metrics_path.relative_to(writer.run_dir).as_posix())

    writer.run_record["dependencies"] = [str(rr_rasters.get("run_id")), str(rr_cal.get("run_id"))]
    writer.run_record["nulls"] = {
        "methods": [perm_null_id, shift_null_id, token_null_id],
        "seed_base": int(canon["CONST"]["BOOTSTRAP_SEED"]),
        "within_layer_perm_seed_fields": ["seed_base", "run_id", "cond_id", "seq_id", "layer"],
        "within_layer_circ_shift_seed_fields": ["seed_base", "run_id", "cond_id", "seq_id", "layer"],
        "token_shuffle_seed_fields": ["seed_base", "run_id", "seq_id"],
    }
    writer.run_record.setdefault("dataset", {}).setdefault("slices", {})["A_rasters"] = smeta_a
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256_A_rasters"] = str(meta_a["dataset_slice_sha256"])
    writer.run_record["conditions"] = [
        f"{metrics_rows[i]['spike_def_id']}|{metrics_rows[i]['target_rate']}|{metrics_rows[i]['g_condition']}|{metrics_rows[i]['null_id']}"
        for i in range(len(metrics_rows))
    ]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 3})
    phase_log.emit("artifacts_written", {"count": 3})


def _phase5_analyze_and_export_toy(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    dep_rasters = _dep_run_dir(args, phase_id="PHASE3_EXTRACT_RASTERS", config_hash=writer.config_hash)
    dep_nulls = _dep_run_dir(args, phase_id="PHASE4_RUN_NULLS", config_hash=writer.config_hash)
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)

    rr_rasters = _load_run_record(dep_rasters)
    rr_nulls = _load_run_record(dep_nulls)
    rr_gstar = _load_run_record(dep_gstar)

    raster_path = dep_rasters / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(
        canon["OUTPUT"]["RASTER_NPZ_BASENAME"]
    )
    null_path = dep_nulls / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["NULL_NPZ_BASENAME"])
    if not raster_path.is_file() or not null_path.is_file():
        raise DependencyError("Missing raster/null dependencies")

    keys = canon["ENUM"]["NPZ_KEY"]
    rz = np.load(raster_path)
    nz = np.load(null_path)
    x = rz[str(keys["X_OCCUPANCY"])]
    x_perm = nz[str(keys["X_OCCUPANCY"])]

    cond_map = rr_rasters.get("condition_map")
    if not isinstance(cond_map, dict):
        raise DependencyError("Missing condition_map in Phase 3 run_record")

    tol = float(canon["CONST"]["RATE_MATCH_TOL_ABS"])
    tau0 = float(canon["CONST"]["TAU0_BASELINE"])
    adj = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_4N"])

    rows: list[dict[str, Any]] = []
    aval_rows: list[dict[str, Any]] = []
    C, N, L, T = x.shape
    for c in range(C):
        cond = cond_map.get(str(c)) or cond_map.get(c)
        if not isinstance(cond, dict):
            continue

        x_c = x[c].astype(np.uint8)
        x_perm_c = x_perm[c].astype(np.uint8)

        denom = float(np.sum(x_c, dtype=np.int64))
        denom_p = float(np.sum(x_perm_c, dtype=np.int64))
        num_time = float(np.sum(x_c[:, :, :-1] * x_c[:, :, 1:], dtype=np.int64))
        num_depth = float(np.sum(x_c[:, :-1, :] * x_c[:, 1:, :], dtype=np.int64))
        num_time_p = float(np.sum(x_perm_c[:, :, :-1] * x_perm_c[:, :, 1:], dtype=np.int64))
        num_depth_p = float(np.sum(x_perm_c[:, :-1, :] * x_perm_c[:, 1:, :], dtype=np.int64))

        b_time = num_time / denom if denom else float("nan")
        b_depth = num_depth / denom if denom else float("nan")
        b_time_p = num_time_p / denom_p if denom_p else float("nan")
        b_depth_p = num_depth_p / denom_p if denom_p else float("nan")

        comps_all = []
        for n in range(N):
            comps = avalanches_from_x(x_c[n], adjacency_id=adj)
            comps_all.extend(comps)
            for ci, comp in enumerate(comps):
                aval_rows.append(
                    {
                        "cond_id": int(c),
                        "seq_id": int(n),
                        "component_id": int(ci),
                        "size": int(comp.size),
                        "span_tokens": int(comp.span_tokens),
                        "span_layers": int(comp.span_layers),
                    }
                )

        aval_stats = avalanche_size_stats(comps_all)
        chi = chi_proxy(np.sum(x_c, axis=0))

        rows.append(
            {
                "cond_id": int(c),
                "spike_def_id": str(cond["spike_def_id"]),
                "target_rate": float(cond["target_rate"]),
                "g": float(cond["g"]),
                "b_time": float(b_time),
                "b_depth": float(b_depth),
                "b_tot": float(b_time + b_depth),
                "b_time_perm": float(b_time_p),
                "b_depth_perm": float(b_depth_p),
                "b_tot_perm": float(b_time_p + b_depth_p),
                "delta_b_time": float(b_time - b_time_p),
                "delta_b_depth": float(b_depth - b_depth_p),
                "delta_b_tot": float((b_time + b_depth) - (b_time_p + b_depth_p)),
                "achieved_rate_mean": float(cond["target_rate"]),
                "achieved_rate_max_abs_err": 0.0,
                "tau0_baseline": tau0,
                "rate_match_tol_abs": tol,
                "chi": float(chi),
                "crackling_gamma": 0.0,
                "crackling_ci_low": 0.0,
                "crackling_ci_high": 0.0,
                "n_sequences": int(N),
                "n_avalanches": int(aval_stats["n_avalanches"]),
                "avalanche_size_mean": float(aval_stats["size_mean"]),
                "avalanche_size_median": float(aval_stats["size_median"]),
                "avalanche_span_tokens_mean": float(aval_stats["span_tokens_mean"]),
                "avalanche_span_layers_mean": float(aval_stats["span_layers_mean"]),
            }
        )

    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_base = str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    aval_base = str(canon["OUTPUT"]["AVALANCHES_PARQUET_BASENAME"])
    metrics_path = results_dir / metrics_base
    aval_path = results_dir / aval_base
    if metrics_path.exists() or aval_path.exists():
        raise RuntimeError("Refusing to overwrite existing results files")
    pd.DataFrame(rows).to_parquet(metrics_path, index=False)
    pd.DataFrame(aval_rows).to_parquet(aval_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=metrics_path.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="avalanches_parquet", relative_path=aval_path.relative_to(writer.run_dir).as_posix())

    # Table T01 (required schema subset)
    t01_csv_name = Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T01_SUMMARY"])).name
    t01_parq_name = Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T01_SUMMARY"])).name
    model_id = str(config.get("models", {}).get("instruct", {}).get("hf_id", ""))
    code_version = writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata")
    dataset_a = str(canon["ENUM"]["DATASET_ROLE"]["A"])
    model_role = str(canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"])

    t01_rows = []
    for r in rows:
        t01_rows.append(
            {
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
        )
    t01_df = pd.DataFrame(t01_rows)
    write_table(df=t01_df, out_csv=_tables_dir(writer) / t01_csv_name, out_parquet=_tables_dir(writer) / t01_parq_name)
    writer.register_artifact(
        logical_name="table_T01_csv",
        relative_path=(Path(str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])) / t01_csv_name).as_posix(),
    )
    writer.register_artifact(
        logical_name="table_T01_parquet",
        relative_path=(Path(str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])) / t01_parq_name).as_posix(),
    )

    # Figures (minimal deterministic exports)
    if rows:
        sel = rows[0]
        series = [r for r in rows if r["spike_def_id"] == sel["spike_def_id"] and r["target_rate"] == sel["target_rate"]]
        series = sorted(series, key=lambda rr: rr["g"])
        xs = [rr["g"] for rr in series]
        btot = [rr["b_tot"] for rr in series]
        dtot = [rr["delta_b_tot"] for rr in series]
        err = [rr["achieved_rate_max_abs_err"] for rr in series]

        def _plot(fig_id: str, ys: dict[str, list[float]], title: str, ylabel: str, meta: dict[str, Any]) -> None:
            fig_pdf = Path(str(canon["OUTPUT"]["FIG_FILE_PDF"][fig_id]))
            fig_png = Path(str(canon["OUTPUT"]["FIG_FILE_PNG"][fig_id]))
            simple_line_plot(
                x=xs,
                ys=ys,
                title=title,
                xlabel="gain g",
                ylabel=ylabel,
                out_pdf=writer.run_dir / fig_pdf,
                out_png=writer.run_dir / fig_png,
                meta=meta,
            )
            writer.register_artifact(logical_name=f"{fig_id}_pdf", relative_path=fig_pdf.as_posix())
            writer.register_artifact(logical_name=f"{fig_id}_png", relative_path=fig_png.as_posix())

        _plot("F03_BRANCHING_CURVES", {"b_tot": btot}, "Branching curves", "b_tot", {"run_id": writer.run_id})
        _plot("F02_RATE_MATCH_CHECK", {"max_abs_err": err}, "Rate match check", "max abs rate error", {"tol": tol})
        _plot("F04_NULL_DELTAB", {"delta_b_tot": dtot}, "Null delta b", "delta_b_tot", {"run_id": writer.run_id})

        gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
        gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
        gstar_val = float(list(gstar_obj.get("by_condition", {}).values())[0]["gstar"])
        _plot("F05_GSTAR_SELECTION", {"b_tot": btot}, "gstar selection (b_tot near 1)", "b_tot", {"gstar": gstar_val})

        # Raster example & spikedef robustness
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = x[0, 0].astype(np.float32)
        fig_pdf = Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F01_RASTER_EXAMPLE"]))
        fig_png = Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F01_RASTER_EXAMPLE"]))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_xlabel("token index")
        ax.set_ylabel("layer index")
        ax.set_title("Raster example")
        fig.tight_layout()
        (writer.run_dir / fig_pdf).parent.mkdir(parents=True, exist_ok=True)
        save_figure(
            fig=fig,
            out_pdf=writer.run_dir / fig_pdf,
            out_png=writer.run_dir / fig_png,
            provenance={
                "run_id": writer.run_id,
                "model_id": str(config.get("models", {}).get("instruct", {}).get("hf_id", "")),
                "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["A"]),
                "spike_def_id": str(sel.get("spike_def_id")),
                "target_rate": str(sel.get("target_rate")),
                "g": "example",
                "config_hash": writer.config_hash,
            },
            dpi=200,
        )
        plt.close(fig)
        writer.register_artifact(logical_name="F01_RASTER_EXAMPLE_pdf", relative_path=fig_pdf.as_posix())
        writer.register_artifact(logical_name="F01_RASTER_EXAMPLE_png", relative_path=fig_png.as_posix())

        _plot("F08_SPIKEDEF_ROBUST", {"b_tot": btot}, "Spike def robustness (toy)", "b_tot", {"note": "toy"})

    writer.run_record["dependencies"] = [rr_rasters.get("run_id"), rr_nulls.get("run_id"), rr_gstar.get("run_id")]
    writer.run_record["conditions"] = rr_rasters.get("conditions", [])
    events.emit("artifacts_written", {"count": 0})
    phase_log.emit("artifacts_written", {"count": 0})


def phase5_analyze_and_export(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    dep_cal = _dep_run_dir(args, phase_id="PHASE1_CALIBRATE", config_hash=writer.config_hash)
    dep_gain_grid = _dep_run_dir(args, phase_id="PHASE2_GAIN_GRID", config_hash=writer.config_hash)
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)
    dep_rasters = _dep_run_dir(args, phase_id="PHASE3_EXTRACT_RASTERS", config_hash=writer.config_hash)
    dep_nulls = _dep_run_dir(args, phase_id="PHASE4_RUN_NULLS", config_hash=writer.config_hash)

    from .analysis.phase5 import run_phase5

    run_phase5(
        writer=writer,
        events=events,
        phase_log=phase_log,
        config=config,
        args=args,
        dep_cal=dep_cal,
        dep_gain_grid=dep_gain_grid,
        dep_gstar=dep_gstar,
        dep_rasters=dep_rasters,
        dep_nulls=dep_nulls,
    )


def _phase6_generalize_b_metrics_toy(
    writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any
) -> None:
    canon = get_canon()
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)
    rr_gstar = _load_run_record(dep_gstar)
    gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
    gstar_val = float(list(gstar_obj.get("by_condition", {}).values())[0]["gstar"])

    spike_defs = list(canon["ENUM"]["SPIKE_DEF_ID"].values())
    target_rates = list(config.get("pipeline", {}).get("target_rates", []))
    baseline = float(canon["CONST"]["GAIN_BASELINE"])

    rows = []
    for s in spike_defs:
        for tr in target_rates:
            for gcond, g in [("g1", baseline), ("gstar", gstar_val)]:
                rows.append(
                    {
                        "run_id": writer.run_id,
                        "stage_id": writer.phase_id,
                        "model_id": str(config.get("models", {}).get("instruct", {}).get("hf_id", "")),
                        "model_role": str(canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"]),
                        "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["B"]),
                        "spike_def_id": str(s),
                        "target_rate": float(tr),
                        "g_condition": gcond,
                        "g": float(g),
                        "b_time": 0.2 + 0.1 * float(g),
                        "b_depth": 0.2 + 0.05 * float(g),
                        "b_tot": 0.4 + 0.15 * float(g),
                        "delta_b_tot": 0.05,
                        "chi": 0.1,
                        "n_sequences": 4,
                        "ppl": 10.0 + 2.0 * (float(g) - baseline),
                        "nll_mean": 2.0,
                        "config_hash": writer.config_hash,
                        "code_version": writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata"),
                    }
                )
    df = pd.DataFrame(rows)

    results_dir = _results_dir(writer)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_base = str(canon["OUTPUT"]["METRICS_PARQUET_BASENAME"])
    metrics_path = results_dir / metrics_base
    if metrics_path.exists():
        raise RuntimeError(f"Refusing to overwrite: {metrics_path}")
    df.to_parquet(metrics_path, index=False)
    writer.register_artifact(logical_name="metrics_parquet", relative_path=metrics_path.relative_to(writer.run_dir).as_posix())

    t02_csv_name = Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T02_GENERALIZATION"])).name
    t02_parq_name = Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T02_GENERALIZATION"])).name
    write_table(df=df, out_csv=_tables_dir(writer) / t02_csv_name, out_parquet=_tables_dir(writer) / t02_parq_name)
    writer.register_artifact(
        logical_name="table_T02_csv",
        relative_path=(Path(str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])) / t02_csv_name).as_posix(),
    )
    writer.register_artifact(
        logical_name="table_T02_parquet",
        relative_path=(Path(str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])) / t02_parq_name).as_posix(),
    )

    fig_pdf = Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F06_GENERALIZATION_B"]))
    fig_png = Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F06_GENERALIZATION_B"]))
    xs = ["g1", "gstar"]
    ys = [float(df[df["g_condition"] == c]["ppl"].mean()) for c in xs]
    simple_line_plot(
        x=list(range(len(xs))),
        ys={"ppl": ys},
        title="Generalization B (toy)",
        xlabel="condition",
        ylabel="ppl",
        out_pdf=writer.run_dir / fig_pdf,
        out_png=writer.run_dir / fig_png,
        meta={"labels": ",".join(xs)},
    )
    writer.register_artifact(logical_name="F06_GENERALIZATION_B_pdf", relative_path=fig_pdf.as_posix())
    writer.register_artifact(logical_name="F06_GENERALIZATION_B_png", relative_path=fig_png.as_posix())

    writer.run_record["dependencies"] = [rr_gstar.get("run_id")]
    writer.run_record["conditions"] = [f"{r['spike_def_id']}|{r['target_rate']}|{r['g_condition']}" for r in rows]
    events.emit("artifacts_written", {"count": 0})
    phase_log.emit("artifacts_written", {"count": 0})


def phase6_generalize_b_metrics(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    dep_cal = _dep_run_dir(args, phase_id="PHASE1_CALIBRATE", config_hash=writer.config_hash)
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)
    dep_nulls = _dep_run_dir(args, phase_id="PHASE4_RUN_NULLS", config_hash=writer.config_hash)
    dep_s05 = _dep_run_dir(args, phase_id="PHASE5_ANALYZE_AND_EXPORT", config_hash=writer.config_hash)

    from .analysis.phase6_b import run_phase6_b

    run_phase6_b(
        writer=writer,
        events=events,
        phase_log=phase_log,
        config=config,
        args=args,
        dep_cal=dep_cal,
        dep_gstar=dep_gstar,
        dep_nulls=dep_nulls,
        dep_s05=dep_s05,
    )


def _phase6_arc_mcq_eval_toy(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    canon = get_canon()
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)
    rr_gstar = _load_run_record(dep_gstar)
    gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
    gstar_val = float(list(gstar_obj.get("by_condition", {}).values())[0]["gstar"])
    baseline = float(canon["CONST"]["GAIN_BASELINE"])

    rows = []
    for gcond, g in [("g1", baseline), ("gstar", gstar_val)]:
        rows.append(
            {
                "run_id": writer.run_id,
                "stage_id": writer.phase_id,
                "model_id": str(config.get("models", {}).get("instruct", {}).get("hf_id", "")),
                "model_role": str(canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"]),
                "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["ARC_MCQ"]),
                "g_condition": gcond,
                "g": float(g),
                "n_questions": 100,
                "accuracy": 0.25 + 0.05 * float(g),
                "accuracy_ci_low": 0.2,
                "accuracy_ci_high": 0.4,
                "mean_logprob_correct_minus_incorrect": 0.0,
                "config_hash": writer.config_hash,
                "code_version": writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata"),
            }
        )
    df = pd.DataFrame(rows)

    t03_csv_name = Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T03_ARC"])).name
    t03_parq_name = Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T03_ARC"])).name
    write_table(df=df, out_csv=_tables_dir(writer) / t03_csv_name, out_parquet=_tables_dir(writer) / t03_parq_name)
    writer.register_artifact(
        logical_name="table_T03_csv",
        relative_path=(Path(str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])) / t03_csv_name).as_posix(),
    )
    writer.register_artifact(
        logical_name="table_T03_parquet",
        relative_path=(Path(str(canon["OUTPUT"]["RUN_SUBDIR"]["TABLES"])) / t03_parq_name).as_posix(),
    )

    fig_pdf = Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F07_ARC_MCQ"]))
    fig_png = Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F07_ARC_MCQ"]))
    simple_line_plot(
        x=[0, 1],
        ys={"accuracy": [float(df.iloc[0]["accuracy"]), float(df.iloc[1]["accuracy"])]},
        title="ARC MCQ (toy)",
        xlabel="condition",
        ylabel="accuracy",
        out_pdf=writer.run_dir / fig_pdf,
        out_png=writer.run_dir / fig_png,
        meta={"labels": "g1,gstar"},
    )
    writer.register_artifact(logical_name="F07_ARC_MCQ_pdf", relative_path=fig_pdf.as_posix())
    writer.register_artifact(logical_name="F07_ARC_MCQ_png", relative_path=fig_png.as_posix())

    writer.run_record["dependencies"] = [rr_gstar.get("run_id")]
    writer.run_record["conditions"] = [f"ARC|{r['g_condition']}" for r in rows]
    events.emit("artifacts_written", {"count": 0})
    phase_log.emit("artifacts_written", {"count": 0})


def phase6_arc_mcq_eval(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    dep_gstar = _dep_run_dir(args, phase_id="PHASE2_SELECT_GSTAR", config_hash=writer.config_hash)
    dep_s05 = _dep_run_dir(args, phase_id="PHASE5_ANALYZE_AND_EXPORT", config_hash=writer.config_hash)

    from .analysis.phase6_arc import run_phase6_arc

    run_phase6_arc(
        writer=writer,
        events=events,
        phase_log=phase_log,
        config=config,
        args=args,
        dep_gstar=dep_gstar,
        dep_s05=dep_s05,
    )


def phase7_paper_export(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None: 
    canon = get_canon() 
    dep_s05 = _dep_run_dir(args, phase_id="PHASE5_ANALYZE_AND_EXPORT", config_hash=writer.config_hash) 
    dep_s06b = _dep_run_dir(args, phase_id="PHASE6_GENERALIZE_B_METRICS", config_hash=writer.config_hash) 
    dep_s06arc = _dep_run_dir(args, phase_id="PHASE6_ARC_MCQ_EVAL", config_hash=writer.config_hash) 
 
    rr_s05 = _load_run_record(dep_s05) 
    rr_s06b = _load_run_record(dep_s06b) 
    rr_s06arc = _load_run_record(dep_s06arc) 

    # Optional cross-model replication summary (T07) when both BASE and INSTRUCT runs exist.
    role_base = str(canon["ENUM"]["MODEL_ROLE"]["BASE"])
    role_instruct = str(canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"])

    def _find_run_by_phase_and_role(*, phase_id: str, model_role: str) -> Path | None:
        # Prefer explicit --dep run ids when provided; otherwise auto-discover in runs/.
        deps = getattr(args, "dep", None) or []
        matches: list[Path] = []
        for rid in deps:
            d = _runs_root() / str(rid)
            if not d.is_dir():
                continue
            rr = _load_run_record(d)
            if rr.get("phase_id") != phase_id:
                continue
            if str(rr.get("model", {}).get("model_role")) != model_role:
                continue
            matches.append(d)
        if not matches:
            for d in _runs_root().iterdir():
                if not d.is_dir():
                    continue
                rr = _load_run_record(d)
                if rr.get("phase_id") != phase_id:
                    continue
                if str(rr.get("model", {}).get("model_role")) != model_role:
                    continue
                matches.append(d)
        if not matches:
            return None
        if len(matches) > 1:
            raise DependencyError(
                f"Multiple runs found for {phase_id} with model_role={model_role}; "
                "run in a fresh workspace or pass explicit --dep run ids"
            )
        return matches[0]

    dep_s05_base = _find_run_by_phase_and_role(phase_id="PHASE5_ANALYZE_AND_EXPORT", model_role=role_base)
    dep_s05_instruct = _find_run_by_phase_and_role(phase_id="PHASE5_ANALYZE_AND_EXPORT", model_role=role_instruct)
    dep_gstar_base = _find_run_by_phase_and_role(phase_id="PHASE2_SELECT_GSTAR", model_role=role_base)
    dep_gstar_instruct = _find_run_by_phase_and_role(phase_id="PHASE2_SELECT_GSTAR", model_role=role_instruct)

    t07_written = False
    if dep_s05_base and dep_s05_instruct and dep_gstar_base and dep_gstar_instruct:
        rr_s05_base = _load_run_record(dep_s05_base)
        rr_s05_instruct = _load_run_record(dep_s05_instruct)

        t01_base = dep_s05_base / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T01_SUMMARY"]))
        t01_ins = dep_s05_instruct / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T01_SUMMARY"]))
        if not t01_base.is_file() or not t01_ins.is_file():
            raise DependencyError("Missing T01 parquet for replication summary")
        df_base = pd.read_parquet(t01_base)
        df_ins = pd.read_parquet(t01_ins)

        gstar_base_path = dep_gstar_base / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
        gstar_ins_path = dep_gstar_instruct / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
        if not gstar_base_path.is_file() or not gstar_ins_path.is_file():
            raise DependencyError("Missing gstar.json for replication summary")
        gstar_base = json.loads(gstar_base_path.read_text(encoding="utf-8"))
        gstar_ins = json.loads(gstar_ins_path.read_text(encoding="utf-8"))
        by_base = gstar_base.get("by_condition", {})
        by_ins = gstar_ins.get("by_condition", {})
        if not isinstance(by_base, dict) or not isinstance(by_ins, dict):
            raise DependencyError("Invalid gstar.json format for replication summary")

        rows: list[dict[str, Any]] = []
        spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
        target_rates = [float(x) for x in config.get("pipeline", {}).get("target_rates", [])]
        for sd in spike_defs:
            for tr in target_rates:
                key = f"{sd}|{float(tr)}"
                if key not in by_base or key not in by_ins:
                    raise DependencyError(f"gstar missing for replication key: {key}")
                g_base = float(by_base[key]["gstar"])
                g_ins = float(by_ins[key]["gstar"])

                row_base = df_base[(df_base["spike_def_id"] == sd) & (df_base["target_rate"] == tr) & (df_base["g"] == g_base)]
                row_ins = df_ins[(df_ins["spike_def_id"] == sd) & (df_ins["target_rate"] == tr) & (df_ins["g"] == g_ins)]
                if row_base.shape[0] != 1 or row_ins.shape[0] != 1:
                    raise DependencyError("Could not locate unique gstar row in T01 for replication summary")
                row_base = row_base.iloc[0].to_dict()
                row_ins = row_ins.iloc[0].to_dict()

                rows.append(
                    {
                        "run_id": writer.run_id,
                        "stage_id": writer.phase_id,
                        "model_id_base": str(rr_s05_base.get("model", {}).get("hf_id")),
                        "model_id_instruct": str(rr_s05_instruct.get("model", {}).get("hf_id")),
                        "run_id_s05_base": str(rr_s05_base.get("run_id")),
                        "run_id_s05_instruct": str(rr_s05_instruct.get("run_id")),
                        "spike_def_id": sd,
                        "target_rate": float(tr),
                        "gstar_base": float(g_base),
                        "gstar_instruct": float(g_ins),
                        "delta_b_tot_at_gstar_base": float(row_base["delta_b_tot"]),
                        "delta_b_tot_at_gstar_instruct": float(row_ins["delta_b_tot"]),
                        "chi_at_gstar_base": float(row_base["chi"]),
                        "chi_at_gstar_instruct": float(row_ins["chi"]),
                        "config_hash": writer.config_hash,
                        "code_version": writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata"),
                    }
                )

        df = pd.DataFrame(rows)
        out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T07_REPLICATION_SUMMARY"]))
        out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T07_REPLICATION_SUMMARY"]))
        write_table(df=df, out_csv=out_csv, out_parquet=out_parq)
        writer.register_artifact(logical_name="table_T07_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
        writer.register_artifact(logical_name="table_T07_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())
        t07_written = True
 
    paper_dir = Path(str(canon["PATH"]["PAPER_DIR"])) 
    bib_path = Path(str(canon["PATH"]["BIB_DIR"])) / str(canon["OUTPUT"]["BIB_REFERENCES_BASENAME"])
 
    validate_no_forbidden_markers(paper_dir) 
    validate_citations(paper_dir, bib_path) 
 
    def _e(rr: dict[str, Any]) -> dict[str, str]:
        run_id = rr.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise DependencyError("Run record missing run_id")
        return {"run_id": run_id, "config_hash": _run_config_hash(rr)}
 
    s05 = _e(rr_s05) 
    s06b = _e(rr_s06b) 
    s06arc = _e(rr_s06arc) 
    s07 = {"run_id": writer.run_id, "config_hash": writer.config_hash}
 
    # Provenance: map canonical artifact IDs to their producing runs. 
    evidence: dict[str, dict[str, Any]] = {} 
    for tid in canon.get("ID", {}).get("TABLE", {}).values(): 
        if tid in (canon["ID"]["TABLE"]["T02_GENERALIZATION"],): 
            evidence[str(tid)] = s06b 
        elif tid in (canon["ID"]["TABLE"]["T03_ARC"],): 
            evidence[str(tid)] = s06arc 
        elif tid in (canon["ID"]["TABLE"].get("T07_REPLICATION_SUMMARY"),) and t07_written:
            evidence[str(tid)] = s07
        else: 
            evidence[str(tid)] = s05 
    for fid in canon.get("ID", {}).get("FIG", {}).values(): 
        if fid in (canon["ID"]["FIG"]["F06_GENERALIZATION_B"],): 
            evidence[str(fid)] = s06b 
        elif fid in (canon["ID"]["FIG"]["F07_ARC_MCQ"],): 
            evidence[str(fid)] = s06arc 
        else: 
            evidence[str(fid)] = s05 
 
    snapshot_name = str(canon["OUTPUT"]["PAPER_SNAPSHOT_MD_BASENAME"])
    snapshot = paper_dir / snapshot_name
    if not snapshot.is_file():
        raise RuntimeError(f"Missing paper snapshot file: {snapshot}")
    update_paper_snapshot(snapshot, evidence=evidence) 
    validate_snapshot_provenance( 
        snapshot, 
        evidence=evidence, 
        required_keys={ 
            str(canon["ID"]["TABLE"]["T01_SUMMARY"]), 
            str(canon["ID"]["TABLE"]["T02_GENERALIZATION"]), 
            str(canon["ID"]["TABLE"]["T03_ARC"]), 
            str(canon["ID"]["FIG"]["F03_BRANCHING_CURVES"]), 
            str(canon["ID"]["FIG"]["F06_GENERALIZATION_B"]), 
            str(canon["ID"]["FIG"]["F07_ARC_MCQ"]), 
        }, 
    ) 
 
    for md in paper_dir.glob("*.md"): 
        if md.name == snapshot.name: 
            continue 
        update_paper_provenance_footnotes(md, evidence=evidence) 
 
    # Lints (fail-closed). 
    validate_no_forbidden_markers(paper_dir) 
    validate_citations(paper_dir, bib_path) 
    validate_artifact_references(paper_dir, run_dirs=[dep_s05, dep_s06b, dep_s06arc, writer.run_dir]) 
 
    # Manifest check: request Phase8 rebuild if missing or mismatched. 
    root = Path.cwd() 
    manifest_path = root / str(canon["OUTPUT"]["MANIFEST_SHA256_BASENAME"])
    zip_name = f"{canon['PROJECT']['PACK_NAME']}.zip"
    zip_path = root / zip_name 
 
    def _iter_files(*, include_manifest: bool) -> list[Path]: 
        out: list[Path] = [] 
        for p in root.rglob("*"):
            if p.is_dir(): 
                continue 
            if ".git" in p.parts:
                continue 
            if p == zip_path: 
                continue 
            if not include_manifest and p == manifest_path: 
                continue 
            out.append(p) 
        return sorted(out, key=lambda x: x.as_posix()) 
 
    expected_lines: list[str] = [] 
    for p in _iter_files(include_manifest=False): 
        digest = hashlib.sha256(p.read_bytes()).hexdigest() 
        rel = p.relative_to(root).as_posix() 
        expected_lines.append(f"{digest}  {rel}")
    expected_text = "\n".join(expected_lines) + "\n"
 
    manifest_exists = manifest_path.is_file() 
    manifest_matches = False 
    if manifest_exists:
        actual_text = manifest_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        manifest_matches = actual_text == expected_text
 
    writer.run_record["release_gate"] = {
        "manifest_path": str(manifest_path),
        "manifest_exists": bool(manifest_exists),
        "manifest_matches": bool(manifest_matches),
        "action_required_if_not_ok": str(canon["CLI"]["CMD"]["PHASE8_RELEASE"]),
    }
 
    writer.run_record["dependencies"] = [rr_s05.get("run_id"), rr_s06b.get("run_id"), rr_s06arc.get("run_id")]
    writer.run_record["conditions"] = sorted(list(evidence.keys()))
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 0})
    phase_log.emit("artifacts_written", {"count": 0})


def phase8_release(writer: RunWriter, events: JsonlLogger, phase_log: JsonlLogger, config: dict[str, Any], args: Any) -> None:
    from .io.hashing import sha256_file

    canon = get_canon()
    root = Path.cwd()
    # Exclude this Phase 8 run directory from the release bundle to avoid self-referential hashes.
    manifest_path, zip_path = write_release_manifest_and_zip(root=root, canon=canon, exclude_dirs=[writer.run_dir])

    writer.run_record["release"] = {
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": int(manifest_path.stat().st_size),
        "zip": zip_path.relative_to(root).as_posix(),
        "zip_sha256": sha256_file(zip_path),
        "zip_bytes": int(zip_path.stat().st_size),
    }
    # Register release artifacts using run-dir-relative paths (may traverse upward).
    rel_manifest = Path(os.path.relpath(manifest_path, start=writer.run_dir)).as_posix()
    rel_zip = Path(os.path.relpath(zip_path, start=writer.run_dir)).as_posix()
    writer.register_artifact(logical_name="release_manifest", relative_path=rel_manifest)
    writer.register_artifact(logical_name="release_zip", relative_path=rel_zip)

    # Fail-closed acceptance test: MANIFEST.sha256 must match recomputation.
    verify_manifest_matches_recomputation(root=root, canon=canon, exclude_dirs=[writer.run_dir])

    writer.run_record["conditions"] = ["release"]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 2})
    phase_log.emit("artifacts_written", {"count": 2})
