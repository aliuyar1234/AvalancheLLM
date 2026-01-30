from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..canon import get_canon
from ..errors import DependencyError
from ..io.artifacts import RunWriter
from ..io.jsonl import JsonlLogger
from ..model.hooks import GainIntervention, get_mlp_layers
from ..model.loader import load_model
from ..plotting.tables import write_table
from ..plotting.savefig import save_figure


def _load_run_record(run_dir: Path) -> dict[str, Any]:
    canon = get_canon()
    rr_path = run_dir / str(canon["OUTPUT"]["RUN_RECORD_JSON"])
    if not rr_path.is_file():
        raise DependencyError(f"Missing run record file: {rr_path}")
    return json.loads(rr_path.read_text(encoding="utf-8"))


def _seed_int(*parts: str) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest()[:8], 16)


def _logprob_continuation(
    *,
    model: torch.nn.Module,
    layers,
    context_ids: list[int],
    continuation_ids: list[int],
    gain: float,
    device: str,
) -> float:
    if not context_ids:
        raise RuntimeError("Empty ARC context_ids")
    if not continuation_ids:
        raise RuntimeError("Empty ARC continuation_ids")

    ids = context_ids + continuation_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        with GainIntervention(layers, gain=float(gain)):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = getattr(out, "logits", None)
    if logits is None:
        raise RuntimeError("Model did not return logits")
    logits = logits[0]  # [T, V]
    c = len(context_ids)
    k = len(continuation_ids)
    if c + k != int(logits.shape[0]):
        raise RuntimeError("Logits length mismatch")
    # Score continuation token(s): logits positions [c-1 .. c+k-2] predict tokens [c .. c+k-1]
    scores = logits[c - 1 : c + k - 1]
    target = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    lp = F.log_softmax(scores, dim=-1).gather(-1, target.unsqueeze(-1)).sum()
    return float(lp.detach().cpu().item())


def run_phase6_arc(
    *,
    writer: RunWriter,
    events: JsonlLogger,
    phase_log: JsonlLogger,
    config: dict[str, Any],
    args: Any,
    dep_gstar: Path,
    dep_s05: Path,
) -> None:
    canon = get_canon()
    rr_gstar = _load_run_record(dep_gstar)
    rr_s05 = _load_run_record(dep_s05)

    ds_cfg = config.get("datasets", {}).get("ARC_MCQ", {})
    hf_id = str(ds_cfg.get("hf_id", ""))
    hf_config = ds_cfg.get("config")
    split = str(ds_cfg.get("split", ""))
    if not hf_id or not split:
        raise RuntimeError("Missing datasets.ARC_MCQ hf_id/split in resolved config")

    template_id = str(canon["DATASET"]["ARC_MCQ"]["PROMPT_TEMPLATE_ID"])
    templates = canon["DATASET"]["ARC_MCQ"].get("PROMPT_TEMPLATES", {})
    if not isinstance(templates, dict) or template_id not in templates:
        raise RuntimeError("Missing ARC prompt template in CANON")
    template = str(templates[template_id])

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

    # gstar per condition (spike_def_id, target_rate)
    gstar_path = dep_gstar / str(canon["OUTPUT"]["RUN_SUBDIR"]["RESULTS"]) / str(canon["OUTPUT"]["GSTAR_JSON_BASENAME"])
    if not gstar_path.is_file():
        raise DependencyError("Missing gstar.json dependency")
    gstar_obj = json.loads(gstar_path.read_text(encoding="utf-8"))
    by_cond = gstar_obj.get("by_condition", {})
    if not isinstance(by_cond, dict):
        raise DependencyError("Invalid gstar.json format")

    baseline = float(canon["CONST"]["GAIN_BASELINE"])
    spike_defs = [str(v) for v in canon["ENUM"]["SPIKE_DEF_ID"].values()]
    target_rates = [str(float(x)) for x in config.get("pipeline", {}).get("target_rates", [])]

    combos: list[tuple[str, str, float]] = []
    gains_needed: set[float] = {baseline}
    for sd in spike_defs:
        for tr in target_rates:
            key = f"{sd}|{float(tr)}"
            if key not in by_cond:
                raise DependencyError(f"gstar missing for {key}")
            g = float(by_cond[key]["gstar"])
            combos.append((sd, tr, g))
            gains_needed.add(float(g))

    # Load ARC dataset and pre-tokenize prompts and option labels deterministically.
    ds_kwargs: dict[str, Any] = {"split": split}
    if hf_config:
        ds_kwargs["name"] = str(hf_config)
    ds = load_dataset(hf_id, **ds_kwargs)
    n_questions = int(len(ds))
    if n_questions <= 0:
        raise RuntimeError("ARC dataset split is empty")

    prepared = []
    hasher = hashlib.sha256()
    for idx, ex in enumerate(ds):
        q = ex.get("question")
        stem = None
        choices = None
        if isinstance(q, str):
            stem = q
            choices = ex.get("choices")
        elif isinstance(q, dict):
            stem = q.get("stem")
            choices = q.get("choices")
        else:
            raise RuntimeError("ARC example missing question")

        if not isinstance(stem, str) or not stem.strip():
            raise RuntimeError("ARC question stem missing")

        answer = ex.get("answerKey")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError("ARC example missing answerKey")

        def _add_choice(*, lab: Any, txt: Any, labels: list[str], lines: list[str]) -> None:
            if not isinstance(lab, str) or not lab:
                return
            if not isinstance(txt, str) or not txt:
                return
            labels.append(lab)
            lines.append(f"{lab}) {txt}")

        lines: list[str] = []
        labels: list[str] = []
        if isinstance(choices, dict):
            labs = choices.get("label")
            txts = choices.get("text")
            if not isinstance(labs, list) or not isinstance(txts, list) or len(labs) != len(txts) or not labs:
                raise RuntimeError("ARC choices dict missing label/text lists")
            for lab, txt in zip(labs, txts):
                _add_choice(lab=lab, txt=txt, labels=labels, lines=lines)
        elif isinstance(choices, list):
            if not choices:
                raise RuntimeError("ARC choices list empty")
            for c in choices:
                if not isinstance(c, dict):
                    continue
                _add_choice(lab=c.get("label"), txt=c.get("text"), labels=labels, lines=lines)
        else:
            raise RuntimeError("ARC choices missing")

        if not labels:
            raise RuntimeError("ARC choices empty after parsing")
        if answer not in labels:
            raise RuntimeError("ARC answerKey not found in labels")

        prompt = template.format(question=stem, choices="\n".join(lines))
        prompt_ids = lm.tokenizer.encode(prompt, add_special_tokens=False)
        # Score labels as continuations with a leading space.
        option_ids = {lab: lm.tokenizer.encode(f" {lab}", add_special_tokens=False) for lab in labels}
        if any(len(v) == 0 for v in option_ids.values()):
            raise RuntimeError("ARC option tokenization produced empty continuation")

        prepared.append(
            {
                "index": int(idx),
                "prompt_ids": [int(i) for i in prompt_ids],
                "option_ids": {k: [int(i) for i in v] for k, v in option_ids.items()},
                "answer": str(answer),
            }
        )
        # Dataset slice hash over sample ids and token ids (spec/11)
        slice_obj = {"dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["ARC_MCQ"]), "index": int(idx), "prompt_ids": prompt_ids, "option_ids": option_ids}
        hasher.update(json.dumps(slice_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\n")

    slice_sha = hasher.hexdigest()
    writer.run_record.setdefault("dataset", {}).setdefault("slices", {})["ARC_MCQ"] = {
        "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["ARC_MCQ"]),
        "hf_id": hf_id,
        "hf_config": str(hf_config) if hf_config else None,
        "split": split,
        "n_questions": int(n_questions),
        "dataset_slice_sha256": str(slice_sha),
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "prompt_template_id": template_id,
    }
    writer.run_record.setdefault("hashes", {})["dataset_slice_sha256_ARC_MCQ"] = str(slice_sha)
    writer.flush_run_record()

    reps = int(canon["CONST"]["BOOTSTRAP_REPS"])
    seed_base = int(canon["CONST"]["BOOTSTRAP_SEED"])

    def _eval_gain(g: float) -> dict[str, Any]:
        correct = np.zeros((n_questions,), dtype=np.uint8)
        margins = np.zeros((n_questions,), dtype=np.float64)
        for i, item in enumerate(prepared):
            prompt_ids = item["prompt_ids"]
            opts = item["option_ids"]
            answer = item["answer"]
            lps = {}
            for lab, cont_ids in opts.items():
                lp = _logprob_continuation(
                    model=lm.model,
                    layers=layers,
                    context_ids=prompt_ids,
                    continuation_ids=cont_ids,
                    gain=float(g),
                    device=args.device,
                )
                lps[lab] = float(lp)
            pred = max(lps.items(), key=lambda kv: kv[1])[0]
            correct[i] = 1 if pred == answer else 0
            incorrect = [v for k, v in lps.items() if k != answer]
            if not incorrect:
                raise RuntimeError("ARC has no incorrect options to compute margin")
            margins[i] = float(lps[answer] - float(np.mean(np.asarray(incorrect, dtype=np.float64))))

        acc = float(np.mean(correct))
        margin_mean = float(np.mean(margins))
        seed = _seed_int(str(seed_base), str(writer.run_id), str(float(g)))
        rng = np.random.default_rng(seed)
        acc_bs = np.zeros((reps,), dtype=np.float64)
        for r in range(reps):
            idx = rng.integers(0, n_questions, size=n_questions)
            acc_bs[r] = float(np.mean(correct[idx]))
        lo = float(np.quantile(acc_bs, 0.025))
        hi = float(np.quantile(acc_bs, 0.975))
        return {
            "n_questions": int(n_questions),
            "accuracy": float(acc),
            "accuracy_ci_low": float(lo),
            "accuracy_ci_high": float(hi),
            "mean_logprob_correct_minus_incorrect": float(margin_mean),
        }

    gain_results: dict[float, dict[str, Any]] = {}
    for g in sorted(gains_needed):
        gain_results[float(g)] = _eval_gain(float(g))

    # Build T03 table rows per (spike_def_id, target_rate) with opt_ columns to disambiguate.
    rows: list[dict[str, Any]] = []
    for sd, tr, gstar_val in combos:
        for gcond, g in [("g1", baseline), ("gstar", float(gstar_val))]:
            res = gain_results[float(g)]
            rows.append(
                {
                    "run_id": writer.run_id,
                    "stage_id": writer.phase_id,
                    "model_id": model_id,
                    "model_role": model_role,
                    "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["ARC_MCQ"]),
                    "g_condition": str(gcond),
                    "g": float(g),
                    "n_questions": int(res["n_questions"]),
                    "accuracy": float(res["accuracy"]),
                    "accuracy_ci_low": float(res["accuracy_ci_low"]),
                    "accuracy_ci_high": float(res["accuracy_ci_high"]),
                    "mean_logprob_correct_minus_incorrect": float(res["mean_logprob_correct_minus_incorrect"]),
                    "config_hash": writer.config_hash,
                    "code_version": writer.run_record.get("hashes", {}).get("code_version", "no_git_metadata"),
                    "opt_spike_def_id": str(sd),
                    "opt_target_rate": float(tr),
                    "opt_prompt_template_id": template_id,
                }
            )

    df = pd.DataFrame(rows)

    out_csv = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_CSV"]["T03_ARC"]))
    out_parq = writer.run_dir / Path(str(canon["OUTPUT"]["TABLE_FILE_PARQUET"]["T03_ARC"]))
    write_table(df=df, out_csv=out_csv, out_parquet=out_parq)
    writer.register_artifact(logical_name="table_T03_csv", relative_path=out_csv.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="table_T03_parquet", relative_path=out_parq.relative_to(writer.run_dir).as_posix())

    # Figure F07 (representative condition: first combo)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sd0, tr0, gstar0 = combos[0]
    base_res = gain_results[baseline]
    gstar_res = gain_results[float(gstar0)]
    fig_pdf = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PDF"]["F07_ARC_MCQ"]))
    fig_png = writer.run_dir / Path(str(canon["OUTPUT"]["FIG_FILE_PNG"]["F07_ARC_MCQ"]))
    xs = [0, 1]
    acc = [float(base_res["accuracy"]), float(gstar_res["accuracy"])]
    lo = [float(base_res["accuracy_ci_low"]), float(gstar_res["accuracy_ci_low"])]
    hi = [float(base_res["accuracy_ci_high"]), float(gstar_res["accuracy_ci_high"])]
    yerr = [[a - l for a, l in zip(acc, lo)], [h - a for a, h in zip(acc, hi)]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(xs, acc, yerr=yerr, fmt="o-", capsize=4)
    ax.set_xticks(xs, ["g1", "gstar"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("ARC MCQ accuracy (with CI)")
    ax.text(
        0.01,
        0.01,
        f"run_id={writer.run_id}, opt_spike_def_id={sd0}, opt_target_rate={tr0}, gstar={float(gstar0)}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    fig.tight_layout()
    fig_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig=fig,
        out_pdf=fig_pdf,
        out_png=fig_png,
        provenance={
            "run_id": writer.run_id,
            "model_id": model_id,
            "dataset_role": str(canon["ENUM"]["DATASET_ROLE"]["ARC_MCQ"]),
            "spike_def_id": str(sd0),
            "target_rate": float(tr0),
            "g_condition": "g1|gstar",
            "config_hash": writer.config_hash,
        },
        dpi=200,
    )
    plt.close(fig)
    writer.register_artifact(logical_name="F07_ARC_MCQ_pdf", relative_path=fig_pdf.relative_to(writer.run_dir).as_posix())
    writer.register_artifact(logical_name="F07_ARC_MCQ_png", relative_path=fig_png.relative_to(writer.run_dir).as_posix())

    writer.run_record["dependencies"] = [str(rr_gstar.get("run_id")), str(rr_s05.get("run_id"))]
    writer.run_record["conditions"] = [f"{sd}|{tr}|g1" for sd, tr, _ in combos] + [f"{sd}|{tr}|gstar" for sd, tr, _ in combos]
    writer.flush_run_record()
    events.emit("artifacts_written", {"count": 4})
    phase_log.emit("artifacts_written", {"count": 4})
