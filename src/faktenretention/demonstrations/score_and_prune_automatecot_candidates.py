#!/usr/bin/env python3
"""Score and prune summarization-adapted Automate-CoT candidates without GLM QA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer


SUMMAC_NLI_REVISION = "3082ba54344bd9ddada2be1c5e9b4131721d2a5d"
SUMMAC_HEAD_SHA256 = "cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric_tokens(text: str) -> set[str]:
    normalized: set[str] = set()
    for token in re.findall(r"(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?%?", text or ""):
        value = re.sub(r"[$£€,%\s]", "", token.lower())
        if value:
            normalized.add(value)
    return normalized


def hard_filter(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not str(row.get("input_text") or "").strip():
        reasons.append("empty_input")
    if not str(row.get("thinking") or "").strip():
        reasons.append("empty_rationale")
    if int(row.get("substantive_thinking_word_count") or 0) < 25:
        reasons.append("rationale_has_fewer_than_25_substantive_words")
    if not str(row.get("response") or "").strip():
        reasons.append("empty_response")
    if not bool(row.get("word_window_success")):
        reasons.append("outside_word_window")
    if not bool(row.get("ends_with_complete_sentence")):
        reasons.append("incomplete_final_sentence")
    expected_input = {500: 800, 300: 500, 150: 300, 75: 150}[int(row["target_words"])]
    input_low, input_high = (
        (795, 805) if int(row["target_words"]) == 500
        else (round(expected_input * 0.95), round(expected_input * 1.05))
    )
    input_words = int(row.get("input_word_count") or 0)
    if not input_low <= input_words <= input_high:
        reasons.append(f"input_outside_expected_stage_window:{input_words}_not_{input_low}-{input_high}")
    return reasons


def build_summac(model_name: str, granularity: str, device: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from faktenretention.metric_resources import summac_resources
    nli_path, head = summac_resources()
    try:
        import torch

        if torch.cuda.is_available():
            try:
                from torch._native.registry import deregister_op_overrides

                deregister_op_overrides(disable_op_symbols=["bmm", "_foreach_mm"])
            except (ImportError, AttributeError):
                pass
        from summac.model_summac import SummaCConv
        from faktenretention.metric_resources import configure_summac
        configure_summac(nli_path, head)
        from transformers import PreTrainedTokenizerBase
    except ImportError as exc:
        raise RuntimeError("SummaC environment is incomplete") from exc

    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        def batch_encode_plus_compat(self, batch_text_or_text_pairs, **kwargs):
            kwargs.pop("truncation_strategy", None)
            return self(batch_text_or_text_pairs, **kwargs)

        PreTrainedTokenizerBase.batch_encode_plus = batch_encode_plus_compat
    return SummaCConv(
        models=[model_name],
        granularity=granularity,
        device=device,
        bins="percentile",
        start_file=str(head),
        imager_load_cache=False,
    )


def score_summac_batches(scorer, rows: list[dict[str, Any]], batch_size: int) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        result = scorer.score(
            [str(row["input_text"]) for row in batch],
            [str(row["response"]) for row in batch],
        )
        values = result.get("scores", result)
        scores.extend(float(value) for value in values)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--summac-model", default="vitc")
    parser.add_argument("--summac-granularity", default="sentence")
    parser.add_argument("--summac-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(args.candidates)
    from faktenretention.input_checks import validate_candidate_keys
    validate_candidate_keys(candidates)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scorer = build_summac(args.summac_model, args.summac_granularity, args.summac_device)
    selected_pools: dict[str, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    stage_report: dict[str, Any] = {}

    for target in (500, 300, 150, 75):
        stage = [row for row in candidates if int(row["target_words"]) == target]
        eligible: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row in stage:
            item = dict(row)
            reasons = hard_filter(item)
            item["novel_numeric_tokens"] = sorted(
                numeric_tokens(str(item.get("response") or ""))
                - numeric_tokens(str(item.get("input_text") or ""))
            )
            item["hard_filter_reasons"] = reasons
            if reasons:
                rejected.append(item)
            else:
                eligible.append(item)
        if len(eligible) < args.pool_size:
            raise RuntimeError(
                f"target={target}: only {len(eligible)} hard-eligible candidates; "
                f"need {args.pool_size}. Generate more candidates before changing filters."
            )

        summac_scores = score_summac_batches(scorer, eligible, args.batch_size)
        for item, summac_score in zip(eligible, summac_scores):
            reference_scores = rouge.score(str(item["highlights"]), str(item["response"]))
            item["summac_input_consistency"] = summac_score
            item["highlight_rouge1_f1"] = float(reference_scores["rouge1"].fmeasure)
            item["highlight_rouge2_f1"] = float(reference_scores["rouge2"].fmeasure)
            item["highlight_rougeL_f1"] = float(reference_scores["rougeL"].fmeasure)

        eligible.sort(
            key=lambda item: (
                -float(item["summac_input_consistency"]),
                -float(item["highlight_rougeL_f1"]),
                str(item["candidate_id"]),
            )
        )
        selected = eligible[: args.pool_size]
        selected_ids = {str(item["candidate_id"]) for item in selected}
        for rank, item in enumerate(eligible, 1):
            item["summac_rank_within_stage"] = rank
            item["selected_into_pool"] = str(item["candidate_id"]) in selected_ids
        selected_pools[str(target)] = selected
        stage_report[str(target)] = {
            "generated_candidates": len(stage),
            "hard_eligible": len(eligible),
            "hard_rejected": len(rejected),
            "retained_pool_size": len(selected),
            "retention_rule": "top K hard-eligible candidates by SummaC input consistency; highlight ROUGE-L then candidate_id break exact ties",
            "minimum_retained_summac": min(float(item["summac_input_consistency"]) for item in selected),
            "maximum_rejected_eligible_summac": (
                max(float(item["summac_input_consistency"]) for item in eligible[args.pool_size:])
                if len(eligible) > args.pool_size
                else None
            ),
            "hard_rejection_reason_counts": {
                reason: sum(reason in item["hard_filter_reasons"] for item in rejected)
                for reason in sorted({reason for item in rejected for reason in item["hard_filter_reasons"]})
            },
        }
        audit_rows.extend(eligible)
        audit_rows.extend(rejected)

    (args.output_dir / "automate_cot_pruned_pools.json").write_text(
        json.dumps(selected_pools, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "automate_cot_pruning_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "created_at_utc": now_utc(),
        "method": "summarization-adapted Automate-CoT augmentation and pruning",
        "glm_qa_used": False,
        "pool_size_per_stage": args.pool_size,
        "summac": {
            "model": args.summac_model,
            "granularity": args.summac_granularity,
            "checkpoint": "authors' trained default SummaCConv checkpoint",
            "nli_revision": SUMMAC_NLI_REVISION,
            "convolution_head_sha256": SUMMAC_HEAD_SHA256,
            "offline_loading": True,
        },
        "hard_filters": [
            "non-empty input, rationale, and response",
            "at least 25 substantive rationale words after Seed budget-marker and markup removal",
            "stage input inside its expected source-length window (795-805, then prior target +/-5%)",
            "response inside preregistered +/-5% word window",
            "complete final sentence",
        ],
        "numeric_novelty_role": "recorded diagnostic and final metric; not a hard filter or pruning objective",
        "reference_relevance_signal": "stemmed ROUGE against human CNN/DailyMail highlights; recorded for all eligible candidates and used only as exact-tie breaker during pruning",
        "stages": stage_report,
    }
    (args.output_dir / "automate_cot_pruning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
