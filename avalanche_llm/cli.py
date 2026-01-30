from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .canon import CanonError, get_canon
from .config import ConfigError, apply_cli_overrides, load_and_resolve_pipeline
from .errors import DependencyError
from .io.artifacts import RunError, RunWriter
from .io.jsonl import JsonlLogger
from .run_id import RunIdError, make_run_id
from .determinism import set_determinism


EXIT_SCHEMA_OR_DETERMINISM = 2
EXIT_MISSING_DEP = 3
EXIT_RUN_EXISTS = 4


@dataclass(frozen=True)
class StageSpec:
    cmd: str
    phase_id: str
    stage_tag: str
    producing: bool


def _canon_cli_defaults() -> dict[str, str]:
    canon = get_canon()
    cmd_map = canon.get("CLI", {}).get("CMD", {})
    if not isinstance(cmd_map, dict):
        return {}

    pick = None
    for k in ("PHASE1_CALIBRATE", "PHASE0_VALIDATE_ENV"):
        if k in cmd_map:
            pick = cmd_map[k]
            break
    if not isinstance(pick, str):
        return {}

    flat = re.sub(r"\\s+", " ", pick).strip()
    tokens = flat.split(" ")

    def _get(flag: str) -> str | None:
        try:
            i = tokens.index(flag)
        except ValueError:
            return None
        if i + 1 >= len(tokens):
            return None
        return tokens[i + 1]

    defaults: dict[str, str] = {}
    for flag, key in [
        ("--config", "config"),
        ("--device", "device"),
        ("--dtype", "dtype"),
        ("--run_id_mode", "run_id_mode"),
    ]:
        v = _get(flag)
        if v is not None:
            defaults[key] = v
    return defaults


def _stages() -> dict[str, StageSpec]:
    return {
        "phase0_validate_env": StageSpec(
            cmd="phase0_validate_env",
            phase_id="PHASE0_VALIDATE_ENV",
            stage_tag="S00",
            producing=False,
        ),
        "phase1_calibrate": StageSpec(
            cmd="phase1_calibrate",
            phase_id="PHASE1_CALIBRATE",
            stage_tag="S01",
            producing=True,
        ),
        "phase2_gain_grid": StageSpec(
            cmd="phase2_gain_grid",
            phase_id="PHASE2_GAIN_GRID",
            stage_tag="S02",
            producing=True,
        ),
        "phase2_select_gstar": StageSpec(
            cmd="phase2_select_gstar",
            phase_id="PHASE2_SELECT_GSTAR",
            stage_tag="S02B",
            producing=True,
        ),
        "phase3_extract_rasters": StageSpec(
            cmd="phase3_extract_rasters",
            phase_id="PHASE3_EXTRACT_RASTERS",
            stage_tag="S03",
            producing=True,
        ),
        "phase4_run_nulls": StageSpec(
            cmd="phase4_run_nulls",
            phase_id="PHASE4_RUN_NULLS",
            stage_tag="S04",
            producing=True,
        ),
        "phase5_analyze_and_export": StageSpec(
            cmd="phase5_analyze_and_export",
            phase_id="PHASE5_ANALYZE_AND_EXPORT",
            stage_tag="S05",
            producing=True,
        ),
        "phase6_generalize_b_metrics": StageSpec(
            cmd="phase6_generalize_b_metrics",
            phase_id="PHASE6_GENERALIZE_B_METRICS",
            stage_tag="S06B",
            producing=True,
        ),
        "phase6_arc_mcq_eval": StageSpec(
            cmd="phase6_arc_mcq_eval",
            phase_id="PHASE6_ARC_MCQ_EVAL",
            stage_tag="S06ARC",
            producing=True,
        ),
        "phase7_paper_export": StageSpec(
            cmd="phase7_paper_export",
            phase_id="PHASE7_PAPER_EXPORT",
            stage_tag="S07",
            producing=True,
        ),
        "phase8_release": StageSpec(
            cmd="phase8_release",
            phase_id="PHASE8_RELEASE",
            stage_tag="S08",
            producing=True,
        ),
    }


def _add_common_args(p: argparse.ArgumentParser, *, defaults: dict[str, str]) -> None:
    canon = get_canon()
    pipeline_default = str(canon.get("PATH", {}).get("CONFIG_PIPELINE", defaults.get("config", "")))
    p.add_argument("--config", default=pipeline_default)
    p.add_argument("--device", default=defaults.get("device", "cuda"))
    p.add_argument("--dtype", default=defaults.get("dtype", "bf16"))
    p.add_argument("--run_id_mode", default=defaults.get("run_id_mode", "content_hash"))
    roles = canon.get("ENUM", {}).get("MODEL_ROLE", {})
    role_instruct = str(roles.get("INSTRUCT", "INSTRUCT"))
    role_base = str(roles.get("BASE", "BASE"))
    p.add_argument("--model_role", default=role_instruct, choices=[role_instruct, role_base])
    p.add_argument("--allow_resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    defaults = _canon_cli_defaults()
    parser = argparse.ArgumentParser(prog="avalanche_llm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, spec in _stages().items():
        sp = sub.add_parser(name)
        _add_common_args(sp, defaults=defaults)
        if name == "phase2_select_gstar":
            sp.add_argument("--gstar_method", default="btot_near_one")
        if name == "phase6_arc_mcq_eval":
            sp.add_argument("--arc_split", default=None)
            sp.add_argument("--arc_config", default=None)
        # Optional explicit dependency run ids for fail-closed determinism.
        if spec.producing and name not in {"phase1_calibrate", "phase8_release"}:
            sp.add_argument("--dep", action="append", default=[], help="Upstream run_id dependency")

    return parser.parse_args(argv)


def _run_phase0_validate_env(args: argparse.Namespace) -> int:
    import importlib
    import traceback

    try:
        resolved = apply_cli_overrides(
            resolved=load_and_resolve_pipeline(Path(args.config)),
            model_role=getattr(args, "model_role", None),
        )
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.stderr.write(traceback.format_exc())
        return EXIT_SCHEMA_OR_DETERMINISM

    det_info = set_determinism(resolved.data.get("determinism", {}))

    required = ["yaml", "numpy", "pandas", "pyarrow", "torch", "transformers", "jsonschema"]
    missing: list[str] = []
    for m in required:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    if missing:
        sys.stderr.write(f"Missing required packages: {missing}\n")
        return EXIT_MISSING_DEP

    canon = get_canon()
    try:
        import transformers

        pin = str(canon["CONST"]["TRANSFORMERS_VERSION_PIN"])
        have = str(getattr(transformers, "__version__", ""))
        if have != pin:
            sys.stderr.write(f"Transformers version mismatch: have={have} pin={pin}\n")
            return EXIT_SCHEMA_OR_DETERMINISM
    except Exception as e:
        sys.stderr.write(f"Could not validate transformers version pin: {e}\n")
        return EXIT_SCHEMA_OR_DETERMINISM

    import torch

    if args.device == "cuda":
        if not torch.cuda.is_available():
            sys.stderr.write("CUDA requested but torch.cuda.is_available() is false\n")
            return EXIT_MISSING_DEP

    # Model load + hook sanity.
    try:
        from .model.hooks import DownprojInputCapture, GainIntervention, UCapture, get_mlp_layers
        from .model.loader import load_model

        model_id = str(resolved.data.get("model_selected", {}).get("hf_id", ""))
        if not model_id:
            raise RuntimeError("Resolved config missing model_selected.hf_id")

        lm = load_model(model_id, device=args.device, dtype=args.dtype)
        layers = get_mlp_layers(lm.model)
        if not layers:
            raise RuntimeError("Model has no MLP layers")

        test_seq_len = int(canon["CONST"]["CRACKLING_D_RANGE_MAX"])
        txt = "Hello world"
        ids = lm.tokenizer.encode(txt, add_special_tokens=False)
        if not ids:
            raise RuntimeError("Tokenizer produced no tokens for probe text")
        ids_rep = (ids * ((test_seq_len // len(ids)) + 1))[:test_seq_len]
        input_ids = torch.tensor([ids_rep], dtype=torch.long, device=args.device)
        attn = torch.ones_like(input_ids)

        baseline_gain = float(canon["CONST"]["GAIN_BASELINE"])

        # Gain=baseline does not change logits.
        with torch.no_grad():
            base = lm.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            base_logits = getattr(base, "logits", None)
            if base_logits is None:
                raise RuntimeError("Model forward did not return logits")

            with GainIntervention(layers, gain=baseline_gain):
                out = lm.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                g1_logits = getattr(out, "logits", None)
                if g1_logits is None:
                    raise RuntimeError("Gain=1 forward did not return logits")

        tol = float(canon["CONST"]["U_HOOK_MATCH_TOL_ABS"])
        max_abs = float((base_logits - g1_logits).abs().max().detach().cpu().item())
        if max_abs > tol:
            raise RuntimeError(f"Gain=1 output mismatch: max_abs_diff={max_abs} tol={tol}")

        # u hook matches down_proj input (layer 0) at baseline gain.
        u_cap: dict[int, torch.Tensor] = {}

        def on_u(layer_idx: int, u: torch.Tensor) -> None:
            if layer_idx == 0 and layer_idx not in u_cap:
                u_cap[layer_idx] = u.detach().clone()

        with torch.no_grad():
            with GainIntervention(layers, gain=baseline_gain):
                with DownprojInputCapture(layers[0]) as cap:
                    with UCapture(layers, on_u=on_u):
                        lm.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        if 0 not in u_cap:
            raise RuntimeError("UCapture did not capture u for layer 0")
        if cap.captured is None:
            raise RuntimeError("DownprojInputCapture did not capture down_proj input")

        diff = float((cap.captured - u_cap[0]).abs().max().detach().cpu().item())
        if diff > tol:
            raise RuntimeError(f"u hook mismatch vs down_proj input: max_abs_diff={diff} tol={tol}")

    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.stderr.write(traceback.format_exc())
        return EXIT_SCHEMA_OR_DETERMINISM

    sys.stdout.write(f"Environment validation passed; determinism={det_info}\n")
    return 0


def _next_timestamp_counter(runs_dir: Path) -> int:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"RUN_{stamp}_"
    counters: list[int] = []
    if runs_dir.is_dir():
        for p in runs_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if not name.startswith(prefix):
                continue
            try:
                counters.append(int(name.split("_")[-1]))
            except Exception:
                continue
    return (max(counters) + 1) if counters else 1


def _make_logger(run_dir: Path, run_id: str, phase_cmd: str) -> tuple[JsonlLogger, JsonlLogger]:
    canon = get_canon()
    logs_subdir = str(canon["OUTPUT"]["RUN_SUBDIR"]["LOGS"])
    logs_dir = run_dir / logs_subdir
    events = JsonlLogger(path=logs_dir / "events.jsonl", run_id=run_id)
    per_phase = JsonlLogger(path=logs_dir / f"{phase_cmd}.jsonl", run_id=run_id)
    return events, per_phase


def _run_producing(
    spec: StageSpec,
    args: argparse.Namespace,
    stage_fn: Callable[[RunWriter, JsonlLogger, JsonlLogger, dict[str, Any], argparse.Namespace], None],
) -> int:
    canon = get_canon()
    resolved = apply_cli_overrides(
        resolved=load_and_resolve_pipeline(Path(args.config)),
        model_role=getattr(args, "model_role", None),
    )
    det_info = set_determinism(resolved.data.get("determinism", {}))

    if args.dry_run:
        sys.stdout.write(
            f"Dry run OK for {spec.phase_id}; config_hash={resolved.sha256} model_role={resolved.data.get('model_role')}\n"
        )
        return 0

    runs_dir = Path(str(canon["PATH"]["RUNS_DIR"]))
    runs_dir.mkdir(parents=True, exist_ok=True)

    counter = None
    if args.run_id_mode == "timestamp_counter":
        counter = _next_timestamp_counter(runs_dir)

    rid = make_run_id(
        stage_tag=spec.stage_tag,
        mode=args.run_id_mode,
        resolved_config_bytes=resolved.resolved_bytes,
        counter=counter,
    )

    run_dir = runs_dir / rid.run_id
    writer = RunWriter(
        run_dir=run_dir,
        run_id=rid.run_id,
        phase_id=spec.phase_id,
        stage_tag=spec.stage_tag,
        created_utc=rid.created_utc,
        config_hash=resolved.sha256,
        resolved_config_bytes=resolved.resolved_bytes,
        allow_resume=bool(args.allow_resume),
    )

    try:
        writer.init_run_dir()
    except RunError as e:
        sys.stderr.write(str(e) + "\n")
        return EXIT_RUN_EXISTS if "already exists" in str(e) else EXIT_SCHEMA_OR_DETERMINISM

    events_log, phase_log = _make_logger(run_dir, rid.run_id, spec.cmd)
    events_log.emit("start", {"phase_id": spec.phase_id})
    phase_log.emit("start", {"phase_id": spec.phase_id})

    writer.write_resolved_config()
    events_log.emit("config_resolved", {"config_sha256": resolved.sha256})

    selected = resolved.data.get("model_selected", {})
    model_info = {
        "device": args.device,
        "dtype": args.dtype,
        "model_role": resolved.data.get("model_role"),
        "selected_hf_id": selected.get("hf_id") if isinstance(selected, dict) else None,
        "models": resolved.data.get("models", {}),
    }
    dataset_info = {
        "datasets": resolved.data.get("datasets", {}),
    }
    writer.start_run_record(model=model_info, dataset=dataset_info, determinism=det_info)

    try:
        stage_fn(writer, events_log, phase_log, resolved.data, args)
    except DependencyError as e:
        events_log.emit("end", {"phase_id": spec.phase_id, "error": str(e)})
        phase_log.emit("end", {"phase_id": spec.phase_id, "error": str(e)})
        sys.stderr.write(str(e) + "\n")
        msg = str(e)
        if "Multiple runs found" in msg:
            return EXIT_SCHEMA_OR_DETERMINISM
        return EXIT_MISSING_DEP
    except Exception as e:  # fail-closed
        events_log.emit("end", {"phase_id": spec.phase_id, "error": str(e)})
        phase_log.emit("end", {"phase_id": spec.phase_id, "error": str(e)})
        import traceback

        sys.stderr.write(traceback.format_exc())
        return EXIT_SCHEMA_OR_DETERMINISM

    events_log.emit("stage_complete", {"phase_id": spec.phase_id})
    phase_log.emit("stage_complete", {"phase_id": spec.phase_id})
    writer.finish()
    events_log.emit("end", {"phase_id": spec.phase_id})
    phase_log.emit("end", {"phase_id": spec.phase_id})
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parse_args(argv)
        stages = _stages()
        spec = stages[args.cmd]
        if args.cmd == "phase0_validate_env":
            return _run_phase0_validate_env(args)

        from .phases import (
            phase1_calibrate,
            phase2_gain_grid,
            phase2_select_gstar,
            phase3_extract_rasters,
            phase4_run_nulls,
            phase5_analyze_and_export,
            phase6_arc_mcq_eval,
            phase6_generalize_b_metrics,
            phase7_paper_export,
            phase8_release,
        )

        stage_fns: dict[str, Callable[..., None]] = {
            "phase1_calibrate": phase1_calibrate,
            "phase2_gain_grid": phase2_gain_grid,
            "phase2_select_gstar": phase2_select_gstar,
            "phase3_extract_rasters": phase3_extract_rasters,
            "phase4_run_nulls": phase4_run_nulls,
            "phase5_analyze_and_export": phase5_analyze_and_export,
            "phase6_generalize_b_metrics": phase6_generalize_b_metrics,
            "phase6_arc_mcq_eval": phase6_arc_mcq_eval,
            "phase7_paper_export": phase7_paper_export,
            "phase8_release": phase8_release,
        }
        return _run_producing(spec, args, stage_fns[args.cmd])
    except (CanonError, ConfigError, RunIdError) as e:
        sys.stderr.write(str(e) + "\n")
        return EXIT_SCHEMA_OR_DETERMINISM
