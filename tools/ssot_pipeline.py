from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineError(RuntimeError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PipelineError(msg)


def _load_canon(source_root: Path) -> dict:
    prev_cwd = Path.cwd()
    try:
        os.chdir(source_root)
        sys.path.insert(0, str(source_root))
        from avalanche_llm.canon import get_canon  # noqa: PLC0415

        return get_canon()
    finally:
        os.chdir(prev_cwd)


def _discover_task_files(source_root: Path, canon: dict, max_phase: int) -> list[Path]:
    tasks_dir = source_root / str(canon["PATH"]["TASKS_DIR"])
    task_index = tasks_dir / "TASK_INDEX.md"
    _require(task_index.is_file(), f"Missing task index: {task_index}")

    phase_files: list[Path] = []
    for line in task_index.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+([A-Za-z0-9_]+\.md)\s*$", line)
        if not m:
            continue
        candidate = tasks_dir / m.group(1)
        if not candidate.is_file():
            continue
        m_num = re.match(r"^PHASE_(\d+)_", candidate.name)
        if m_num and int(m_num.group(1)) <= max_phase:
            phase_files.append(candidate)

    _require(phase_files, f"No phase task files found under: {tasks_dir}")
    return phase_files


def _discover_commands_with_phase_numbers(task_files: list[Path], canon: dict) -> list[tuple[str, int]]:
    cmd_map = canon.get("CLI", {}).get("CMD", {})
    _require(isinstance(cmd_map, dict) and cmd_map, "Missing CANON.CLI.CMD mapping")

    ordered: list[tuple[str, int]] = []
    seen: set[str] = set()
    for task_file in task_files:
        phase_num = 0
        m_num = re.match(r"^PHASE_(\d+)_", task_file.name)
        if m_num:
            phase_num = int(m_num.group(1))

        text = task_file.read_text(encoding="utf-8")
        for cmd_id in re.findall(r"CANON\.CLI\.CMD\.([A-Z0-9_]+)", text):
            if cmd_id in seen:
                continue
            _require(cmd_id in cmd_map, f"Task references missing CANON.CLI.CMD.{cmd_id}")
            ordered.append((cmd_id, phase_num))
            seen.add(cmd_id)

    _require(ordered, "No CANON.CLI.CMD.* references found in phase tasks")
    return ordered


def _copy_source_to_workspace(
    source_root: Path,
    workspace_dir: Path,
    canon: dict,
) -> None:
    _require(not workspace_dir.exists(), f"Workspace already exists: {workspace_dir}")
    _require(
        source_root != workspace_dir and source_root not in workspace_dir.parents,
        f"Refusing to create workspace inside source tree: {workspace_dir}",
    )

    runs_dir_name = str(canon["PATH"]["RUNS_DIR"])
    manifest_basename = str(canon["OUTPUT"]["MANIFEST_SHA256_BASENAME"])
    zip_name = f"{canon['PROJECT']['PACK_NAME']}.zip"

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        ignore = {".git", runs_dir_name, manifest_basename, zip_name, "__pycache__", ".pytest_cache"}
        return {n for n in names if n in ignore}

    shutil.copytree(source_root, workspace_dir, ignore=_ignore, dirs_exist_ok=False)


def _nvidia_smi_sample() -> str | None:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return p.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _run_cmd(
    *,
    cmd_id: str,
    cmd: str,
    cwd: Path,
    env: dict[str, str],
    gpu_smi: bool,
) -> None:
    flat = " ".join(cmd.split())
    if gpu_smi:
        sample = _nvidia_smi_sample()
        if sample is not None:
            sys.stdout.write(f"[gpu] {sample}\n")
    sys.stdout.write(f"[run] {cmd_id}: {flat}\n")
    sys.stdout.flush()
    subprocess.run(flat, cwd=cwd, env=env, shell=True, check=True)
    if gpu_smi:
        sample = _nvidia_smi_sample()
        if sample is not None:
            sys.stdout.write(f"[gpu] {sample}\n")
            sys.stdout.flush()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SSOT pipeline (tasks/TASK_INDEX order) in a fresh workspace.")
    p.add_argument(
        "--out-root",
        required=True,
        help="Directory where a new workspace directory will be created.",
    )
    p.add_argument(
        "--source-root",
        default=str(Path.cwd()),
        help="SSOT source root to copy from (defaults to current directory).",
    )
    p.add_argument(
        "--workspace-name",
        default=None,
        help="Workspace directory name (default: derived from CANON.PROJECT.PACK_NAME + UTC timestamp).",
    )
    p.add_argument(
        "--max-phase",
        type=int,
        default=8,
        help="Maximum phase number from tasks/TASK_INDEX.md to execute (default: 8).",
    )
    p.add_argument(
        "--gpu-smi",
        action="store_true",
        help="Print nvidia-smi samples before and after each command when available.",
    )
    p.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest gate before Phase 8 release (not recommended).",
    )
    p.add_argument(
        "--sync-latex",
        action="store_true",
        help="After Phase 7 (and before Phase 8), sync latest run artifacts into paper/latex and rebuild the Overleaf zip.",
    )
    p.add_argument(
        "--latex-build",
        action="store_true",
        help="When --sync-latex is set, also build paper/latex/main.pdf via pdflatex (nonstopmode).",
    )
    return p.parse_args()


def _load_run_record(run_dir: Path, *, canon: dict[str, Any]) -> dict[str, Any]:
    rr_path = run_dir / str(canon["OUTPUT"]["RUN_RECORD_JSON"])
    _require(rr_path.is_file(), f"Missing run record: {rr_path}")
    import json

    obj = json.loads(rr_path.read_text(encoding="utf-8"))
    _require(isinstance(obj, dict), f"Invalid run record format: {rr_path}")
    return obj


def _find_unique_run_dir(
    *,
    workspace_dir: Path,
    canon: dict[str, Any],
    phase_id: str,
    model_role: str,
) -> Path:
    runs_dir = workspace_dir / str(canon["PATH"]["RUNS_DIR"])
    _require(runs_dir.is_dir(), f"Missing runs dir: {runs_dir}")
    matches: list[Path] = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        rr = _load_run_record(d, canon=canon)
        if str(rr.get("phase_id")) != phase_id:
            continue
        role = rr.get("model", {}).get("model_role")
        if str(role) != model_role:
            continue
        matches.append(d)
    _require(matches, f"Missing run for phase_id={phase_id} model_role={model_role}")
    _require(
        len(matches) == 1,
        f"Multiple runs found for phase_id={phase_id} model_role={model_role}; workspace must be clean",
    )
    return matches[0]


def _tex_tt(s: str) -> str:
    # Keep this minimal: only escape the characters we emit in \texttt{}.
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    s = s.replace("%", "\\%")
    s = s.replace("&", "\\&")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    return f"\\texttt{{{s}}}"


def _sync_latex_bundle(*, workspace_dir: Path, canon: dict[str, Any], latex_build: bool) -> None:
    """
    Copy the latest canonical exported figures/tables into paper/latex, regenerate LaTeX table snippets,
    and rebuild the Overleaf zip (and optionally main.pdf).
    """
    import json
    import zipfile

    import numpy as np
    import pandas as pd

    paper_dir = workspace_dir / str(canon["PATH"]["PAPER_DIR"])
    latex_dir = paper_dir / "latex"
    _require(latex_dir.is_dir(), f"Missing LaTeX dir: {latex_dir}")

    fig_dst_root = latex_dir / "figures"
    tbl_dst_root = latex_dir / "tables"
    fig_dst_root.mkdir(parents=True, exist_ok=True)
    tbl_dst_root.mkdir(parents=True, exist_ok=True)
    (fig_dst_root / "appendix").mkdir(parents=True, exist_ok=True)
    (tbl_dst_root / "appendix").mkdir(parents=True, exist_ok=True)

    role_instruct = str(canon["ENUM"]["MODEL_ROLE"]["INSTRUCT"])
    s05 = _find_unique_run_dir(workspace_dir=workspace_dir, canon=canon, phase_id="PHASE5_ANALYZE_AND_EXPORT", model_role=role_instruct)
    s06b = _find_unique_run_dir(workspace_dir=workspace_dir, canon=canon, phase_id="PHASE6_GENERALIZE_B_METRICS", model_role=role_instruct)
    s06arc = _find_unique_run_dir(workspace_dir=workspace_dir, canon=canon, phase_id="PHASE6_ARC_MCQ_EVAL", model_role=role_instruct)
    s07 = _find_unique_run_dir(workspace_dir=workspace_dir, canon=canon, phase_id="PHASE7_PAPER_EXPORT", model_role=role_instruct)
    s02b = _find_unique_run_dir(workspace_dir=workspace_dir, canon=canon, phase_id="PHASE2_SELECT_GSTAR", model_role=role_instruct)

    def _copy(src: Path, dst: Path) -> None:
        _require(src.is_file(), f"Missing artifact file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Copy canonical figures (PDF+PNG) into paper/latex/figures[/appendix].
    fig_pdf = canon["OUTPUT"]["FIG_FILE_PDF"]
    fig_png = canon["OUTPUT"]["FIG_FILE_PNG"]
    for fig_id, rel_pdf in fig_pdf.items():
        rel_png = fig_png.get(fig_id)
        _require(isinstance(rel_pdf, str) and isinstance(rel_png, str), f"Invalid FIG_FILE mapping for {fig_id}")

        if fig_id == str(canon["ID"]["FIG"]["F06_GENERALIZATION_B"]):
            src_run = s06b
        elif fig_id == str(canon["ID"]["FIG"]["F07_ARC_MCQ"]):
            src_run = s06arc
        else:
            src_run = s05

        pdf_src = src_run / Path(rel_pdf)
        png_src = src_run / Path(rel_png)
        is_appendix = "appendix" in Path(rel_pdf).parts
        pdf_dst = fig_dst_root / ("appendix" if is_appendix else "") / pdf_src.name
        png_dst = fig_dst_root / ("appendix" if is_appendix else "") / png_src.name
        _copy(pdf_src, pdf_dst)
        _copy(png_src, png_dst)

    # Copy canonical tables (CSV+Parquet) into paper/latex/tables[/appendix].
    tbl_csv = canon["OUTPUT"]["TABLE_FILE_CSV"]
    tbl_parq = canon["OUTPUT"]["TABLE_FILE_PARQUET"]
    for table_id, rel_csv in tbl_csv.items():
        rel_parq = tbl_parq.get(table_id)
        _require(isinstance(rel_csv, str) and isinstance(rel_parq, str), f"Invalid TABLE_FILE mapping for {table_id}")

        if table_id == str(canon["ID"]["TABLE"]["T02_GENERALIZATION"]):
            src_run = s06b
        elif table_id == str(canon["ID"]["TABLE"]["T03_ARC"]):
            src_run = s06arc
        elif table_id == str(canon["ID"]["TABLE"].get("T07_REPLICATION_SUMMARY", "")):
            src_run = s07
        else:
            src_run = s05

        csv_src = src_run / Path(rel_csv)
        parq_src = src_run / Path(rel_parq)
        is_appendix = "appendix" in Path(rel_csv).parts
        csv_dst = tbl_dst_root / ("appendix" if is_appendix else "") / csv_src.name
        parq_dst = tbl_dst_root / ("appendix" if is_appendix else "") / parq_src.name
        _copy(csv_src, csv_dst)
        _copy(parq_src, parq_dst)

    # Load key tables for snippet generation.
    t01 = pd.read_parquet(s05 / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T01_SUMMARY"])))
    t02 = pd.read_parquet(s06b / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T02_GENERALIZATION"])))
    t03 = pd.read_parquet(s06arc / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T03_ARC"])))
    t04 = pd.read_parquet(s05 / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T04_TAIL_FITS"])))
    t05 = pd.read_parquet(s05 / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T05_CRACKLING_DIAGNOSTICS"])))
    t06 = pd.read_parquet(s05 / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T06_ABLATIONS"])))
    t07_path = s07 / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T07_REPLICATION_SUMMARY"]))
    _require(t07_path.is_file(), f"Missing replication summary parquet (T07): {t07_path}")
    t07 = pd.read_parquet(t07_path)

    gstar_path = s02b / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    _require(gstar_path.is_file(), f"Missing gstar.json: {gstar_path}")
    gstar = json.loads(gstar_path.read_text(encoding="utf-8"))
    gstar_by = gstar.get("by_condition", {})
    _require(isinstance(gstar_by, dict) and gstar_by, "Invalid gstar.json format")

    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rates = [float(x) for x in canon["CONST"]["TARGET_LAYER_RATES"]]
    g_baseline = float(canon["CONST"]["GAIN_BASELINE"])

    def _spike_label(sd: str) -> str:
        if sd == str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_ONE_SIDED_POS"]):
            return "one-sided (+)"
        if sd == str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_TWO_SIDED_ABS"]):
            return "two-sided ($|\\cdot|$)"
        return sd

    def _fmt_rate(tr: float) -> str:
        return f"{float(tr):.0e}"

    def _pick_row(df: pd.DataFrame, *, sd: str, tr: float, g: float) -> pd.Series:
        sub = df[(df["spike_def_id"].astype(str) == sd) & (df["target_rate"].astype(float) == float(tr))]
        _require(len(sub) > 0, f"Missing rows for spike_def_id={sd} target_rate={tr}")
        sub = sub[np.isclose(sub["g"].astype(float).to_numpy(), float(g), atol=1e-12, rtol=0.0)]
        _require(len(sub) == 1, f"Expected exactly 1 row for spike_def_id={sd} target_rate={tr} g={g}")
        return sub.iloc[0]

    # tab_gstar_summary.tex
    g_rows: list[list[str]] = []
    for sd in spike_defs:
        for tr in target_rates:
            key = f"{sd}|{float(tr)}"
            _require(key in gstar_by, f"gstar missing for {key}")
            g_star = float(gstar_by[key]["gstar"])
            r_g1 = _pick_row(t01, sd=sd, tr=tr, g=g_baseline)
            r_gs = _pick_row(t01, sd=sd, tr=tr, g=g_star)
            g_rows.append(
                [
                    _spike_label(sd),
                    _fmt_rate(tr),
                    f"{g_star:.3f}",
                    f"{float(r_g1['b_tot']):.3f}",
                    f"{float(r_gs['b_tot']):.3f}",
                    f"{float(r_g1['delta_b_tot']):.3f}",
                    f"{float(r_gs['delta_b_tot']):.3f}",
                ]
            )
    gstar_tex = """\\begin{table}[t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{2.5pt}
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{llrrrrr}
\\toprule
Spike def & Target rate & $g^{\\star}$ & $b_{\\mathrm{tot}}(g{=}1)$ & $b_{\\mathrm{tot}}(g^{\\star})$ & $\\Delta b_{\\mathrm{tot}}(g{=}1)$ & $\\Delta b_{\\mathrm{tot}}(g^{\\star})$ \\\\
\\midrule
"""
    for row in g_rows:
        gstar_tex += " " + " & ".join(row) + " \\\\\n"
    gstar_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Mechanistic gain calibration on Dataset A: selected $g^{\\star}$ by minimizing $|b_{\\mathrm{tot}}(g)-1|$ under rate-matched thresholds, and corresponding branching statistics at $g=1$ and $g^{\\star}$. Values from Table~T01 and \\texttt{gstar.json}.}
\\label{tab:gstar}
\\end{table}
"""
    (tbl_dst_root / "tab_gstar_summary.tex").write_text(gstar_tex, encoding="utf-8", newline="\n")

    # tab_generalization_b.tex
    gen_rows: list[list[str]] = []
    _require("opt_ppl_ci_low" in t02.columns and "opt_ppl_ci_high" in t02.columns, "T02 missing PPL CI columns")
    for sd in spike_defs:
        for tr in target_rates:
            sub = t02[(t02["spike_def_id"].astype(str) == sd) & (t02["target_rate"].astype(float) == float(tr))]
            _require(len(sub) > 0, f"Missing T02 rows for spike_def_id={sd} target_rate={tr}")
            r1 = sub[sub["g_condition"].astype(str) == "g1"]
            rs = sub[sub["g_condition"].astype(str) == "gstar"]
            _require(len(r1) == 1 and len(rs) == 1, "Expected exactly one g1 and one gstar row in T02")
            r1 = r1.iloc[0]
            rs = rs.iloc[0]
            dppl = float(rs["ppl"]) - float(r1["ppl"])
            ppl1 = float(r1["ppl"])
            ppl1_lo = float(r1["opt_ppl_ci_low"])
            ppl1_hi = float(r1["opt_ppl_ci_high"])
            ppls = float(rs["ppl"])
            ppls_lo = float(rs["opt_ppl_ci_low"])
            ppls_hi = float(rs["opt_ppl_ci_high"])
            gen_rows.append(
                [
                    _spike_label(sd),
                    _fmt_rate(tr),
                    f"{float(rs['g']):.3f}",
                    f"{ppl1:.3f} [{ppl1_lo:.3f}, {ppl1_hi:.3f}]",
                    f"{ppls:.3f} [{ppls_lo:.3f}, {ppls_hi:.3f}]",
                    f"{dppl:.3f}",
                ]
            )
    gen_tex = """\\begin{table}[t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{3pt}
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{llrrrr}
\\toprule
 Spike def & Target rate & $g^{\\star}$ & PPL($g{=}1$) [95\\% CI] & PPL($g^{\\star}$) [95\\% CI] & $\\Delta$PPL \\\\
\\midrule
"""
    for row in gen_rows:
        gen_tex += " " + " & ".join(row) + " \\\\\n"
    gen_tex += """\\bottomrule
\\end{tabular}%
}
 \\caption{Dataset B evaluation: perplexity at $g=1$ and at the mechanistically calibrated $g^{\\star}$ (selected on Dataset A), with bootstrap 95\\% confidence intervals from sequence resampling.}
\\label{tab:genB}
\\end{table}
"""
    (tbl_dst_root / "tab_generalization_b.tex").write_text(gen_tex, encoding="utf-8", newline="\n")

    # tab_arc.tex
    arc_rows: list[list[str]] = []
    _require("accuracy_ci_low" in t03.columns and "accuracy_ci_high" in t03.columns, "T03 missing accuracy CI columns")
    for sd in spike_defs:
        for tr in target_rates:
            sub = t03[(t03["opt_spike_def_id"].astype(str) == sd) & (t03["opt_target_rate"].astype(float) == float(tr))]
            _require(len(sub) > 0, f"Missing T03 rows for spike_def_id={sd} target_rate={tr}")
            r1 = sub[sub["g_condition"].astype(str) == "g1"]
            rs = sub[sub["g_condition"].astype(str) == "gstar"]
            _require(len(r1) == 1 and len(rs) == 1, "Expected exactly one g1 and one gstar row in T03")
            r1 = r1.iloc[0]
            rs = rs.iloc[0]
            dacc = float(rs["accuracy"]) - float(r1["accuracy"])
            a1 = float(r1["accuracy"])
            a1_lo = float(r1["accuracy_ci_low"])
            a1_hi = float(r1["accuracy_ci_high"])
            aS = float(rs["accuracy"])
            aS_lo = float(rs["accuracy_ci_low"])
            aS_hi = float(rs["accuracy_ci_high"])
            arc_rows.append(
                [
                    _spike_label(sd),
                    _fmt_rate(tr),
                    f"{float(rs['g']):.3f}",
                    f"{a1:.3f} [{a1_lo:.3f}, {a1_hi:.3f}]",
                    f"{aS:.3f} [{aS_lo:.3f}, {aS_hi:.3f}]",
                    f"{dacc:.3f}",
                ]
            )
    arc_tex = """\\begin{table}[t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{3pt}
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{llrrrr}
\\toprule
 Spike def & Target rate & $g^{\\star}$ & Acc($g{=}1$) [95\\% CI] & Acc($g^{\\star}$) [95\\% CI] & $\\Delta$Acc \\\\
\\midrule
"""
    for row in arc_rows:
        arc_tex += " " + " & ".join(row) + " \\\\\n"
    arc_tex += """\\bottomrule
\\end{tabular}%
}
 \\caption{ARC-Challenge multiple-choice evaluation: accuracy at $g=1$ vs $g^{\\star}$ (selected on Dataset A), with bootstrap 95\\% confidence intervals from question resampling.}
\\label{tab:arc}
\\end{table}
"""
    (tbl_dst_root / "tab_arc.tex").write_text(arc_tex, encoding="utf-8", newline="\n")

    # tab_t01_selected.tex (selected columns from T01 across full gain grid)
    needed = [
        "spike_def_id",
        "target_rate",
        "g",
        "achieved_rate_max_abs_err",
        "b_tot",
        "delta_b_tot",
        "chi",
        "crackling_gamma",
        "n_avalanches",
        "avalanche_size_mean",
        "avalanche_span_tokens_mean",
        "avalanche_span_layers_mean",
    ]
    for c in needed:
        _require(c in t01.columns, f"T01 missing required column for LaTeX table: {c}")
    pick = t01[needed].copy()
    pick = pick.sort_values(["spike_def_id", "target_rate", "g"], kind="mergesort")
    t01_rows: list[list[str]] = []
    for _, r in pick.iterrows():
        t01_rows.append(
            [
                _spike_label(str(r["spike_def_id"])),
                _fmt_rate(float(r["target_rate"])),
                f"{float(r['g']):.3f}",
                f"{float(r['achieved_rate_max_abs_err']):.2e}",
                f"{float(r['b_tot']):.3f}",
                f"{float(r['delta_b_tot']):.3f}",
                f"{float(r['chi']):.3f}",
                f"{float(r['crackling_gamma']):.3f}",
                f"{int(r['n_avalanches'])}",
                f"{float(r['avalanche_size_mean']):.3f}",
                f"{float(r['avalanche_span_tokens_mean']):.3f}",
                f"{float(r['avalanche_span_layers_mean']):.3f}",
            ]
        )
    t01_tex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrrrrrrrr}
\\toprule
Spike def & Target rate & $g$ & max|rate err| & $b_{\\mathrm{tot}}$ & $\\Delta b_{\\mathrm{tot}}$ & $\\chi$ & $\\gamma$ & \\#avals & mean size & mean span (tok) & mean span (layers) \\\\
\\midrule
"""
    for row in t01_rows:
        t01_tex += " " + " & ".join(row) + " \\\\\n"
    t01_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Selected columns from Table~T01 (Dataset A): rate-matching error, branching, null-controlled residual, susceptibility proxy, crackling exponent estimate, and component summary statistics across all conditions.}
\\label{tab:t01_selected}
\\end{table*}
"""
    (tbl_dst_root / "tab_t01_selected.tex").write_text(t01_tex, encoding="utf-8", newline="\n")

    # tab_tail_fits.tex (appendix)
    tail_cols = [
        "spike_def_id",
        "target_rate",
        "g",
        "xmin",
        "n_tail",
        "alpha_powerlaw",
        "llr_powerlaw_vs_lognormal",
        "llr_powerlaw_vs_exponential",
    ]
    for c in tail_cols:
        _require(c in t04.columns, f"T04 missing required column for LaTeX table: {c}")
    tail = t04[tail_cols].copy().sort_values(["spike_def_id", "target_rate", "g"], kind="mergesort")
    tail_rows: list[list[str]] = []
    for _, r in tail.iterrows():
        tail_rows.append(
            [
                _spike_label(str(r["spike_def_id"])),
                _fmt_rate(float(r["target_rate"])),
                f"{float(r['g']):.3f}",
                f"{float(r['xmin']):.3f}",
                f"{int(r['n_tail'])}",
                f"{float(r['alpha_powerlaw']):.3f}",
                f"{float(r['llr_powerlaw_vs_lognormal']):.3f}",
                f"{float(r['llr_powerlaw_vs_exponential']):.3f}",
            ]
        )
    tail_tex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrrrr}
\\toprule
Spike def & Target rate & $g$ & $x_{\\min}$ & $n_{\\mathrm{tail}}$ & $\\alpha$ (PL) & LLR(PL--LN) & LLR(PL--EXP) \\\\
\\midrule
"""
    for row in tail_rows:
        tail_tex += " " + " & ".join(row) + " \\\\\n"
    tail_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Tail-fit diagnostics on avalanche size (descriptive only). Continuous-approximation fits on the upper tail defined by a fixed percentile.}
\\label{tab:tail_fits}
\\end{table*}
"""
    (tbl_dst_root / "appendix" / "tab_tail_fits.tex").write_text(tail_tex, encoding="utf-8", newline="\n")

    # tab_crackling_diagnostics.tex (appendix)
    crack_cols = [
        "spike_def_id",
        "target_rate",
        "g",
        "crackling_gamma",
        "crackling_ci_low",
        "crackling_ci_high",
        "crackling_ci_width",
        "n_avalanches_used",
        "r2",
        "pass",
    ]
    for c in crack_cols:
        _require(c in t05.columns, f"T05 missing required column for LaTeX table: {c}")
    crack = t05[crack_cols].copy().sort_values(["spike_def_id", "target_rate", "g"], kind="mergesort")
    crack_rows: list[list[str]] = []
    for _, r in crack.iterrows():
        crack_rows.append(
            [
                _spike_label(str(r["spike_def_id"])),
                _fmt_rate(float(r["target_rate"])),
                f"{float(r['g']):.3f}",
                f"{float(r['crackling_gamma']):.3f}",
                f"{float(r['crackling_ci_low']):.3f}",
                f"{float(r['crackling_ci_high']):.3f}",
                f"{float(r['crackling_ci_width']):.3f}",
                f"{int(r['n_avalanches_used'])}",
                f"{float(r['r2']):.3f}",
                "pass" if bool(r["pass"]) else "fail",
            ]
        )
    crack_tex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrrrrrr}
\\toprule
Spike def & Target rate & $g$ & $\\gamma$ & CI low & CI high & CI width & $n_{\\mathrm{avals}}$ & $R^2$ & gate \\\\
\\midrule
"""
    for row in crack_rows:
        crack_tex += " " + " & ".join(row) + " \\\\\n"
    crack_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Crackling fit diagnostics (descriptive) with a fail-closed gate.}
\\label{tab:crackling}
\\end{table*}
"""
    (tbl_dst_root / "appendix" / "tab_crackling_diagnostics.tex").write_text(crack_tex, encoding="utf-8", newline="\n")

    # tab_ablations.tex (appendix)
    ab_cols = ["intervention_id", "spike_def_id", "target_rate", "g", "delta_b_tot", "achieved_rate_max_abs_err"]
    for c in ab_cols:
        _require(c in t06.columns, f"T06 missing required column for LaTeX table: {c}")
    ab = t06[ab_cols].copy().sort_values(["intervention_id", "spike_def_id", "target_rate"], kind="mergesort")
    ab_rows: list[list[str]] = []
    for _, r in ab.iterrows():
        ab_rows.append(
            [
                _tex_tt(str(r["intervention_id"])),
                _spike_label(str(r["spike_def_id"])),
                _fmt_rate(float(r["target_rate"])),
                f"{float(r['g']):.3f}",
                f"{float(r['delta_b_tot']):.3f}",
                f"{float(r['achieved_rate_max_abs_err']):.2e}",
            ]
        )
    ab_tex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrr}
\\toprule
Intervention & Spike def & Target rate & $g$ & $\\Delta b_{\\mathrm{tot}}$ & max|rate err| \\\\
\\midrule
"""
    for row in ab_rows:
        ab_tex += " " + " & ".join(row) + " \\\\\n"
    ab_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Ablation comparison of gain interventions at matched rates (appendix).}
\\label{tab:ablations}
\\end{table*}
"""
    (tbl_dst_root / "appendix" / "tab_ablations.tex").write_text(ab_tex, encoding="utf-8", newline="\n")

    # tab_replication_summary.tex (appendix)
    rep_cols = [
        "spike_def_id",
        "target_rate",
        "gstar_base",
        "gstar_instruct",
        "delta_b_tot_at_gstar_base",
        "delta_b_tot_at_gstar_instruct",
        "chi_at_gstar_base",
        "chi_at_gstar_instruct",
    ]
    for c in rep_cols:
        _require(c in t07.columns, f"T07 missing required column for LaTeX table: {c}")
    rep = t07[rep_cols].copy().sort_values(["spike_def_id", "target_rate"], kind="mergesort")
    rep_rows: list[list[str]] = []
    for _, r in rep.iterrows():
        rep_rows.append(
            [
                _spike_label(str(r["spike_def_id"])),
                _fmt_rate(float(r["target_rate"])),
                f"{float(r['gstar_base']):.3f}",
                f"{float(r['gstar_instruct']):.3f}",
                f"{float(r['delta_b_tot_at_gstar_base']):.3f}",
                f"{float(r['delta_b_tot_at_gstar_instruct']):.3f}",
                f"{float(r['chi_at_gstar_base']):.3f}",
                f"{float(r['chi_at_gstar_instruct']):.3f}",
            ]
        )
    rep_tex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrrrr}
\\toprule
Spike def & Target rate & $g^{\\star}_{\\mathrm{base}}$ & $g^{\\star}_{\\mathrm{inst}}$ & $\\Delta b_{\\mathrm{tot}}$ (base) & $\\Delta b_{\\mathrm{tot}}$ (inst) & $\\chi$ (base) & $\\chi$ (inst) \\\\
\\midrule
"""
    for row in rep_rows:
        rep_tex += " " + " & ".join(row) + " \\\\\n"
    rep_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Replication summary comparing base vs instruction-tuned checkpoints at their respective $g^{\\star}$ (appendix).}
\\label{tab:replication}
\\end{table*}
"""
    (tbl_dst_root / "appendix" / "tab_replication_summary.tex").write_text(rep_tex, encoding="utf-8", newline="\n")

    # tab_provenance.tex (include all canonical artifacts).
    rr_s05 = _load_run_record(s05, canon=canon)
    rr_s06b = _load_run_record(s06b, canon=canon)
    rr_s06arc = _load_run_record(s06arc, canon=canon)
    rr_s07 = _load_run_record(s07, canon=canon)

    def _hash_prefix(rr: dict[str, Any]) -> str:
        h = rr.get("hashes", {}).get("config_sha256")
        return (str(h)[:12] + "...") if isinstance(h, str) else "unknown"

    prov_rows: list[list[str]] = []
    for fid in canon.get("ID", {}).get("FIG", {}).values():
        if fid == str(canon["ID"]["FIG"]["F06_GENERALIZATION_B"]):
            rr = rr_s06b
        elif fid == str(canon["ID"]["FIG"]["F07_ARC_MCQ"]):
            rr = rr_s06arc
        else:
            rr = rr_s05
        prov_rows.append([_tex_tt(str(fid)), _tex_tt(str(rr.get("run_id"))), _tex_tt(_hash_prefix(rr))])
    for tid in canon.get("ID", {}).get("TABLE", {}).values():
        if tid == str(canon["ID"]["TABLE"]["T02_GENERALIZATION"]):
            rr = rr_s06b
        elif tid == str(canon["ID"]["TABLE"]["T03_ARC"]):
            rr = rr_s06arc
        elif tid == str(canon["ID"]["TABLE"].get("T07_REPLICATION_SUMMARY")):
            rr = rr_s07
        else:
            rr = rr_s05
        prov_rows.append([_tex_tt(str(tid)), _tex_tt(str(rr.get("run_id"))), _tex_tt(_hash_prefix(rr))])

    prov_tex = """\\begin{table*}[t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{3pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lll}
\\toprule
Artifact & Run & Config hash (prefix)\\\\
\\midrule
"""
    for row in prov_rows:
        prov_tex += " " + " & ".join(row) + " \\\\\n"
    prov_tex += """\\bottomrule
\\end{tabular}%
}
\\caption{Artifact provenance (run identifiers and resolved config hashes) for exported figures and tables. Full hashes and checksums are recorded in each run's \\texttt{run\\_record.json} and in \\texttt{MANIFEST.sha256}.}
\\label{tab:provenance}
\\end{table*}
"""
    (tbl_dst_root / "tab_provenance.tex").write_text(prov_tex, encoding="utf-8", newline="\n")

    # Optional LaTeX build.
    if latex_build:
        main_tex = latex_dir / "main.tex"
        _require(main_tex.is_file(), f"Missing LaTeX entrypoint: {main_tex}")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=latex_dir,
            check=True,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=latex_dir,
            check=True,
        )

    # Rebuild Overleaf zip (replace existing zip if present).
    zips = sorted([p for p in latex_dir.glob("*.zip") if p.is_file()], key=lambda p: p.name)
    _require(len(zips) == 1, f"Expected exactly one Overleaf zip in {latex_dir}, found {len(zips)}")
    zip_path = zips[0]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(latex_dir.rglob("*"), key=lambda x: x.as_posix()):
            if p.is_dir():
                continue
            if p == zip_path:
                continue
            rel = p.relative_to(latex_dir).as_posix()
            zf.write(p, rel)


def main() -> int:
    try:
        args = _parse_args()
        source_root = Path(args.source_root).expanduser().resolve()
        out_root = Path(args.out_root).expanduser().resolve()
        _require(source_root.is_dir(), f"Missing source root: {source_root}")
        out_root.mkdir(parents=True, exist_ok=True)

        canon = _load_canon(source_root)
        ws_name = args.workspace_name
        if ws_name is None:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            ws_name = f"{canon['PROJECT']['PACK_NAME']}_run_{ts}"
        workspace_dir = out_root / ws_name

        task_files = _discover_task_files(source_root, canon, max_phase=int(args.max_phase))
        cmd_ids_with_phases = _discover_commands_with_phase_numbers(task_files, canon)

        _copy_source_to_workspace(source_root, workspace_dir, canon)
        sys.stdout.write(f"[workspace] {workspace_dir}\n")
        sys.stdout.flush()

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        cmd_map = canon["CLI"]["CMD"]
        tests_ran = False
        latex_synced = False
        for cmd_id, phase_num in cmd_ids_with_phases:
            if phase_num >= 8 and int(args.max_phase) >= 8 and not args.skip_tests and not tests_ran:
                subprocess.run(
                    ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                    cwd=workspace_dir,
                    env=env,
                    check=True,
                )
                tests_ran = True
            if bool(args.sync_latex) and not latex_synced and int(args.max_phase) >= 8 and phase_num >= 8:
                _sync_latex_bundle(workspace_dir=workspace_dir, canon=canon, latex_build=bool(args.latex_build))
                latex_synced = True
            _run_cmd(
                cmd_id=cmd_id,
                cmd=str(cmd_map[cmd_id]),
                cwd=workspace_dir,
                env=env,
                gpu_smi=bool(args.gpu_smi),
            )

        sys.stdout.write("[done] pipeline complete\n")
        return 0
    except PipelineError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ERROR: command failed (exit={e.returncode}): {e.cmd}\n")
        return e.returncode if isinstance(e.returncode, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
