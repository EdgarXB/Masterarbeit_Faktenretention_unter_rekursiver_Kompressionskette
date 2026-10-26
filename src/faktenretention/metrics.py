#!/usr/bin/env python3
"""Evaluate final Auto-CoT/Automate-CoT summaries with paired uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from faktenretention.protocol import STAGES as TARGETS
from faktenretention.text_rules import bounds_for_target, count_words
from faktenretention.metric_scoring import compute_bertscore_f1, compute_summac_scores


METRICS = (
    "highlight_rouge1_f1",
    "highlight_rouge2_f1",
    "highlight_rougeL_f1",
    "highlight_bertscore_f1",
    "source_summac",
    "stage_input_summac",
    "word_window_success",
    "initial_word_window_success",
    "absolute_word_error",
    "absolute_relative_word_error",
    "word_window_violation_words",
    "word_window_violation_rate",
    "complete_sentence_success",
    "novel_numeric_rate_vs_article",
    "novel_numeric_rate_vs_stage_input",
    "empty_output",
)
PINNED_BERTSCORE_MODEL = None

PRIMARY_CONTRASTS = (
    ("plain", "lead_k"),
    ("zero_shot_cot", "plain"),
    ("few_shot_cot", "plain"),
    ("auto_cot", "plain"),
    ("automate_cot", "plain"),
    ("few_shot_cot", "zero_shot_cot"),
    ("automate_cot", "auto_cot"),
)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_tokens(text: str) -> set[str]:
    values: set[str] = set()
    for token in re.findall(r"(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?%?", text or ""):
        value = re.sub(r"[$£€,%\s]", "", token.lower())
        if value:
            values.add(value)
    return values


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = rng.choice(array, size=len(array), replace=True).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def parse_bool(value: str) -> float:
    return 1.0 if str(value).lower() == "true" else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bertscore-model", default=PINNED_BERTSCORE_MODEL)
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--summac-model", default="vitc")
    parser.add_argument("--summac-granularity", default="sentence")
    parser.add_argument("--summac-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--stages", type=int, nargs="+", default=list(TARGETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.results)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    details: list[dict[str, Any]] = []
    candidates: list[str] = []
    highlights: list[str] = []
    sources: list[str] = []
    stage_inputs: list[str] = []
    semantic_indices: list[int] = []
    for row in rows:
        for target in args.stages:
            summary = str(row.get(f"summary_{target}") or "").strip()
            highlight = str(row["highlights"])
            source = str(row["article"])
            scores = rouge.score(highlight, summary)
            low, high = bounds_for_target(target, 5.0)
            words = int(row.get(f"summary_{target}_word_count") or count_words(summary))
            absolute_word_error = abs(words - target)
            word_window_violation_words = max(low - words, words - high, 0)
            prior_target = {500: None, 300: 500, 150: 300, 75: 150}[target]
            stage_input = source if prior_target is None else str(row.get(f"summary_{prior_target}") or "")
            novel_vs_article = sorted(numeric_tokens(summary) - numeric_tokens(source))
            novel_vs_stage_input = sorted(numeric_tokens(summary) - numeric_tokens(stage_input))
            detail = {
                "article_id": row["article_id"],
                "dataset_index": row["dataset_index"],
                "method": row["method"],
                "stage_target_words": target,
                "stage_actual_words": words,
                "accepted_min_words": low,
                "accepted_max_words": high,
                "word_window_success": float(low <= words <= high),
                "initial_word_window_success": parse_bool(row.get(f"summary_{target}_initial_window_success", "")),
                "absolute_word_error": absolute_word_error,
                "absolute_relative_word_error": absolute_word_error / target,
                "word_window_violation_words": word_window_violation_words,
                "word_window_violation_rate": word_window_violation_words / target,
                "complete_sentence_success": parse_bool(row.get(f"summary_{target}_complete_sentence_success", "")),
                "attempts": int(row.get(f"summary_{target}_attempts") or 1),
                "highlight_rouge1_f1": float(scores["rouge1"].fmeasure),
                "highlight_rouge2_f1": float(scores["rouge2"].fmeasure),
                "highlight_rougeL_f1": float(scores["rougeL"].fmeasure),
                "highlight_bertscore_f1": None,
                "source_summac": None,
                "stage_input_summac": None,
                "empty_output": float(not bool(summary)),
                "novel_numeric_tokens_vs_article": "|".join(novel_vs_article),
                "novel_numeric_rate_vs_article": float(bool(novel_vs_article)),
                "novel_numeric_tokens_vs_stage_input": "|".join(novel_vs_stage_input),
                "novel_numeric_rate_vs_stage_input": float(bool(novel_vs_stage_input)),
            }
            details.append(detail)
            if summary:
                semantic_indices.append(len(details) - 1)
                candidates.append(summary)
                highlights.append(highlight)
                sources.append(source)
                stage_inputs.append(stage_input)
            else:
                detail["highlight_bertscore_f1"] = 0.0
                detail["source_summac"] = 0.0
                detail["stage_input_summac"] = 0.0

    bert = compute_bertscore_f1(
        candidates, highlights, model_type=args.bertscore_model, lang=args.bertscore_lang
    )
    summac = compute_summac_scores(
        candidates, sources, model_name=args.summac_model,
        granularity=args.summac_granularity, device=args.summac_device,
    )
    stage_summac = compute_summac_scores(
        candidates, stage_inputs, model_name=args.summac_model,
        granularity=args.summac_granularity, device=args.summac_device,
    )
    for detail_index, bert_value, summac_value, stage_summac_value in zip(
        semantic_indices, bert, summac, stage_summac
    ):
        detail = details[detail_index]
        detail["highlight_bertscore_f1"] = bert_value
        detail["source_summac"] = summac_value
        detail["stage_input_summac"] = stage_summac_value

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        groups[(str(detail["method"]), int(detail["stage_target_words"]))].append(detail)
    aggregates: list[dict[str, Any]] = []
    for (method, target), group in sorted(groups.items()):
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            low, high = bootstrap_mean_ci(values, args.bootstrap_samples, args.seed + target)
            aggregates.append({
                "method": method,
                "stage_target_words": target,
                "metric": metric,
                "n_articles": len(values),
                "mean": mean(values),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
            })

    by_key = {
        (str(row["article_id"]), int(row["stage_target_words"]), str(row["method"])): row
        for row in details
    }
    paired: list[dict[str, Any]] = []
    present_methods = {str(row["method"]) for row in details}
    for contrast_index, (left_method, right_method) in enumerate(PRIMARY_CONTRASTS):
        if not {left_method, right_method}.issubset(present_methods):
            continue
        for target in args.stages:
            article_ids = sorted({str(row["article_id"]) for row in details if int(row["stage_target_words"]) == target})
            for metric in METRICS:
                differences = [
                    float(by_key[(article_id, target, left_method)][metric])
                    - float(by_key[(article_id, target, right_method)][metric])
                    for article_id in article_ids
                    if (article_id, target, left_method) in by_key
                    and (article_id, target, right_method) in by_key
                ]
                if not differences:
                    continue
                low, high = bootstrap_mean_ci(
                    differences, args.bootstrap_samples,
                    args.seed + 1000 + target + contrast_index * 10000,
                )
                paired.append({
                    "contrast": f"{left_method}_minus_{right_method}",
                    "stage_target_words": target,
                    "metric": metric,
                    "n_paired_articles": len(differences),
                    "mean_paired_difference": mean(differences),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                })

    detail_fields = list(details[0].keys()) if details else []
    write_csv(args.output_dir / "metrics_detailed.csv", details, detail_fields)
    write_csv(
        args.output_dir / "metrics_by_method_stage.csv", aggregates,
        ["method", "stage_target_words", "metric", "n_articles", "mean", "bootstrap_95_ci_low", "bootstrap_95_ci_high"],
    )
    write_csv(
        args.output_dir / "paired_method_differences.csv", paired,
        ["contrast", "stage_target_words", "metric", "n_paired_articles", "mean_paired_difference", "bootstrap_95_ci_low", "bootstrap_95_ci_high"],
    )
    report = {
        "created_at_utc": now_utc(),
        "rows": len(details),
        "articles": len({row["article_id"] for row in details}),
        "methods": sorted({row["method"] for row in details}),
        "metrics": {
            "rouge": "stemmed ROUGE against human CNN/DailyMail highlights",
            "bertscore": {"reference": "human highlights", "model": args.bertscore_model},
            "summac": {"reference": "original article", "model": args.summac_model, "granularity": args.summac_granularity},
            "stage_input_summac": {"reference": "immediate recursive input", "model": args.summac_model, "granularity": args.summac_granularity},
            "numeric_vs_article": "normalized digit/currency/percentage values absent from original article",
            "numeric_vs_stage_input": "normalized digit/currency/percentage values absent from the immediate recursive input",
            "complete_sentence_success": "selected output ends in declared terminal punctuation/closer pattern",
            "absolute_word_error": "absolute distance between observed and target word count; lower is better",
            "absolute_relative_word_error": "absolute word error divided by stage target; lower is better",
            "word_window_violation_words": (
                "distance in words to the nearest accepted plus/minus-5-percent window boundary; "
                "zero for every in-window output"
            ),
            "word_window_violation_rate": (
                "word_window_violation_words divided by stage target; primary continuous length-adherence "
                "measure for comparisons across target sizes; zero is optimal and lower is better"
            ),
            "empty_output_policy": (
                "Empty Single-pass outputs remain in every denominator with zero word count, zero "
                "overlap/semantic/consistency scores, and empty_output=1."
            ),
        },
        "uncertainty": {
            "method": "article-level nonparametric bootstrap percentile interval",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "paired_contrasts": [f"{left}_minus_{right}" for left, right in PRIMARY_CONTRASTS],
        },
        "interpretation_warning": (
            "The metrics are complementary proxies; no metric establishes human-equivalent overall "
            "quality or factuality. CNN/DailyMail highlights are much shorter than the 500/300-word "
            "outputs, so highlight ROUGE/BERTScore are interpreted as within-stage paired method "
            "comparisons, not absolute quality comparisons across target lengths."
        ),
    }
    (args.output_dir / "metrics_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
