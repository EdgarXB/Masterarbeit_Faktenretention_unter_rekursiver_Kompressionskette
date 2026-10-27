from faktenretention.io import read_csv, write_csv
from faktenretention.scoring import (token_f1, parse_number_tokens, retention_token_f1, normalize_textual_quantities, parse_longest_number, normalize, retained, quantity_token, canonicalize_for_retention, fraction_token, normalize_symbolic_quantities, NUMBER_WORDS, NUMBER_SCALES, FRACTION_DENOMINATORS)
#!/usr/bin/env python3
import argparse
import csv
import glob
import hashlib
import os
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from faktenretention.protocol import STAGES as TARGETS
from faktenretention.text_rules import bounds_for_target, count_words


def rouge_scores(reference: str, candidate: str) -> Tuple[float, float, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: rouge-score. Install with `python3 -m pip install rouge-score`.") from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return (
        float(scores["rouge1"].fmeasure),
        float(scores["rouge2"].fmeasure),
        float(scores["rougeL"].fmeasure),
    )


def avg(values: Iterable[float]) -> float:
    collected = list(values)
    return mean(collected) if collected else 0.0


def compute_bertscore_f1(
    candidates: List[str],
    references: List[str],
    *,
    model_type: str,
    lang: str,
) -> List[Optional[float]]:
    if not candidates:
        return []
    # Torch 2.13 may route CUDA batched matrix multiplication through an
    # optional Triton override that JIT-compiles against Python.h.  Minimal
    # benchmark hosts do not necessarily ship Python development headers;
    # falling back to Torch's native CUDA kernels keeps the metric identical
    # while avoiding that unrelated build-time dependency.
    try:
        import torch
        if torch.cuda.is_available():
            from torch._native.registry import deregister_op_overrides
            deregister_op_overrides(disable_op_symbols=["bmm", "_foreach_mm"])
    except (ImportError, AttributeError):
        pass
    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise RuntimeError("Missing dependency: bert-score. Install with `python3 -m pip install bert-score`.") from exc

    kwargs = {
        "model_type": model_type, "lang": lang, "verbose": False,
        "rescale_with_baseline": False,
    }
    if Path(model_type).is_dir():
        kwargs["num_layers"] = 6
    _, _, f1 = bert_score_fn(candidates, references, **kwargs)
    return [float(value) for value in f1]


def compute_summac_scores(
    candidates: List[str],
    references: List[str],
    *,
    model_name: str,
    granularity: str,
    device: str,
) -> List[Optional[float]]:
    if not candidates:
        return []
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    expected_revision = "3082ba54344bd9ddada2be1c5e9b4131721d2a5d"
    from faktenretention.metric_resources import summac_resources
    nli_path, head = summac_resources()
    try:
        from summac.model_summac import SummaCConv
        from faktenretention.metric_resources import configure_summac
        configure_summac(nli_path, head)
        from transformers import PreTrainedTokenizerBase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: summac. Install with `python3 -m pip install summac`.") from exc

    # SummaC 0.0.4 calls the legacy Transformers batch_encode_plus API,
    # which was removed in Transformers 5. Preserve its behavior through
    # the modern tokenizer call interface.
    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        def batch_encode_plus_compat(self, batch_text_or_text_pairs, **kwargs):
            kwargs.pop("truncation_strategy", None)
            return self(batch_text_or_text_pairs, **kwargs)

        PreTrainedTokenizerBase.batch_encode_plus = batch_encode_plus_compat

    # SummaCConv's library default leaves the final convolution head randomly
    # initialized.  Use the authors' published trained checkpoint so the
    # resulting factual-consistency scores are meaningful and reproducible.
    scorer = SummaCConv(
        models=[model_name],
        granularity=granularity,
        device=device,
        bins="percentile",
        start_file=str(head),
        imager_load_cache=False,
    )
    result = scorer.score(references, candidates)
    scores = result.get("scores", result)
    return [float(value) for value in scores]


def score_summaries(
    articles_path: Path,
    summary_files: List[Path],
    output_dir: Path,
    tolerance_pct: float,
    *,
    with_bertscore: bool,
    bertscore_model: str,
    bertscore_lang: str,
    with_summac: bool,
    summac_model: str,
    summac_granularity: str,
    summac_device: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    articles = {row["article_id"]: row for row in read_csv(articles_path, missing_ok=True)}
    detailed: List[Dict[str, str]] = []
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    bert_candidates: List[str] = []
    bert_references: List[str] = []
    bert_detail_indices: List[int] = []
    summac_candidates: List[str] = []
    summac_references: List[str] = []
    summac_detail_indices: List[int] = []

    for summary_file in summary_files:
        for row in read_csv(summary_file, missing_ok=True):
            article_id = row.get("article_id", "")
            reference = (articles.get(article_id, {}).get("article") or "").strip()
            if not reference:
                continue
            for stage in TARGETS:
                summary = (row.get(f"summary_{stage}") or "").strip()
                wc = int(row.get(f"summary_{stage}_word_count") or count_words(summary))
                min_words, max_words = bounds_for_target(stage, tolerance_pct)
                rouge1, rouge2, rouge_l = rouge_scores(reference, summary)
                reference_wc = int(articles.get(article_id, {}).get("word_count") or count_words(reference))
                compression_rate = wc / reference_wc if reference_wc else 0.0
                detail = {
                    "summary_file": str(summary_file),
                    "batch_run_id": row.get("batch_run_id", summary_file.stem),
                    "model_run_id": row.get("model_run_id", ""),
                    "model_name": row.get("model_name", ""),
                    "prompt_variant": row.get("prompt_variant", ""),
                    "article_id": article_id,
                    "reference_word_count": str(reference_wc),
                    "stage_target_words": str(stage),
                    "stage_actual_words": str(wc),
                    "stage_min_words": str(min_words),
                    "stage_max_words": str(max_words),
                    "word_window_success": str(min_words <= wc <= max_words),
                    "compression_rate": f"{compression_rate:.6f}",
                    "rouge1_fmeasure": f"{rouge1:.6f}",
                    "rouge2_fmeasure": f"{rouge2:.6f}",
                    "rougeL_fmeasure": f"{rouge_l:.6f}",
                    "bertscore_f1": "0.000000" if not summary else "",
                    "summac_score": "0.000000" if not summary else "",
                    "empty_output": str(not bool(summary)),
                }
                detailed.append(detail)
                detail_index = len(detailed) - 1
                if with_bertscore and summary:
                    bert_candidates.append(summary)
                    bert_references.append(reference)
                    bert_detail_indices.append(detail_index)
                if with_summac and summary:
                    summac_candidates.append(summary)
                    summac_references.append(reference)
                    summac_detail_indices.append(detail_index)
                groups[(detail["batch_run_id"], detail["model_name"], detail["prompt_variant"], str(stage))].append(
                    {
                        "word_window_success": 1.0 if min_words <= wc <= max_words else 0.0,
                        "compression_rate": compression_rate,
                        "rouge1": rouge1,
                        "rouge2": rouge2,
                        "rougeL": rouge_l,
                        "bertscore_f1": 0.0,
                        "has_bertscore": 0.0,
                        "summac_score": 0.0,
                        "has_summac": 0.0,
                    }
                )

    if with_bertscore:
        bert_scores = compute_bertscore_f1(
            bert_candidates,
            bert_references,
            model_type=bertscore_model,
            lang=bertscore_lang,
        )
        for detail_index, score in zip(bert_detail_indices, bert_scores):
            detailed[detail_index]["bertscore_f1"] = f"{score:.6f}" if score is not None else ""

    if with_summac:
        summac_scores = compute_summac_scores(
            summac_candidates,
            summac_references,
            model_name=summac_model,
            granularity=summac_granularity,
            device=summac_device,
        )
        for detail_index, score in zip(summac_detail_indices, summac_scores):
            detailed[detail_index]["summac_score"] = f"{score:.6f}" if score is not None else ""

    groups.clear()
    for detail in detailed:
        key = (detail["batch_run_id"], detail["model_name"], detail["prompt_variant"], detail["stage_target_words"])
        groups[key].append(
            {
                "word_window_success": 1.0 if detail["word_window_success"] == "True" else 0.0,
                "compression_rate": float(detail["compression_rate"] or 0),
                "rouge1": float(detail["rouge1_fmeasure"] or 0),
                "rouge2": float(detail["rouge2_fmeasure"] or 0),
                "rougeL": float(detail["rougeL_fmeasure"] or 0),
                "bertscore_f1": float(detail["bertscore_f1"] or 0),
                "has_bertscore": 1.0 if detail["bertscore_f1"] else 0.0,
                "summac_score": float(detail["summac_score"] or 0),
                "has_summac": 1.0 if detail["summac_score"] else 0.0,
            }
        )

    summary_rows: List[Dict[str, str]] = []
    for (batch_run_id, model_name, prompt_variant, stage), rows in groups.items():
        summary_rows.append(
            {
                "batch_run_id": batch_run_id,
                "model_name": model_name,
                "prompt_variant": prompt_variant,
                "stage_target_words": stage,
                "num_rows": str(len(rows)),
                "word_window_success_rate": f"{avg(row['word_window_success'] for row in rows):.6f}",
                "avg_compression_rate": f"{avg(row['compression_rate'] for row in rows):.6f}",
                "avg_rouge1_fmeasure": f"{avg(row['rouge1'] for row in rows):.6f}",
                "avg_rouge2_fmeasure": f"{avg(row['rouge2'] for row in rows):.6f}",
                "avg_rougeL_fmeasure": f"{avg(row['rougeL'] for row in rows):.6f}",
                "avg_bertscore_f1": f"{avg(row['bertscore_f1'] for row in rows if row['has_bertscore']):.6f}",
                "avg_summac_score": f"{avg(row['summac_score'] for row in rows if row['has_summac']):.6f}",
            }
        )

    write_csv(
        output_dir / "summary_quality_detailed.csv",
        detailed,
        [
            "summary_file",
            "batch_run_id",
            "model_run_id",
            "model_name",
            "prompt_variant",
            "article_id",
            "reference_word_count",
            "stage_target_words",
            "stage_actual_words",
            "stage_min_words",
            "stage_max_words",
            "word_window_success",
            "compression_rate",
            "rouge1_fmeasure",
            "rouge2_fmeasure",
            "rougeL_fmeasure",
            "bertscore_f1",
            "summac_score",
            "empty_output",
        ],
    )
    write_csv(
        output_dir / "summary_quality_by_stage.csv",
        summary_rows,
        [
            "batch_run_id",
            "model_name",
            "prompt_variant",
            "stage_target_words",
            "num_rows",
            "word_window_success_rate",
            "avg_compression_rate",
            "avg_rouge1_fmeasure",
            "avg_rouge2_fmeasure",
            "avg_rougeL_fmeasure",
            "avg_bertscore_f1",
            "avg_summac_score",
        ],
    )
    return detailed, summary_rows


def score_facts(fact_summary_path: Path, output_dir: Path) -> List[Dict[str, str]]:
    rows = read_csv(fact_summary_path, missing_ok=True)
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("model_name", "")].append(row)
    summary_rows: List[Dict[str, str]] = []
    for model_name, model_rows in groups.items():
        summary_rows.append(
            {
                "model_name": model_name,
                "num_article_runs": str(len(model_rows)),
                "exactly_20_rate": f"{avg(1.0 if row['exactly_20'] == 'True' else 0.0 for row in model_rows):.6f}",
                "avg_valid_schema_count": f"{avg(float(row['valid_schema_count'] or 0) for row in model_rows):.3f}",
                "avg_source_supported_count": f"{avg(float(row['source_supported_count'] or 0) for row in model_rows):.3f}",
                "avg_reference_answer_supported_count": f"{avg(float(row['reference_answer_supported_count'] or 0) for row in model_rows):.3f}",
                "avg_duplicate_count": f"{avg(float(row['duplicate_count'] or 0) for row in model_rows):.3f}",
                "avg_answerable_count": f"{avg(float(row['answerable_count'] or 0) for row in model_rows):.3f}",
                "avg_entity_type_count": f"{avg(float(row['entity_type_count'] or 0) for row in model_rows):.3f}",
            }
        )
    write_csv(
        output_dir / "fact_extraction_kpis.csv",
        summary_rows,
        [
            "model_name",
            "num_article_runs",
            "exactly_20_rate",
            "avg_valid_schema_count",
            "avg_source_supported_count",
            "avg_reference_answer_supported_count",
            "avg_duplicate_count",
            "avg_answerable_count",
            "avg_entity_type_count",
        ],
    )
    return summary_rows


def score_qa(answers_path: Path, output_dir: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows = read_csv(answers_path, missing_ok=True)
    detailed: List[Dict[str, str]] = []
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    entity_groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    group_totals: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    entity_totals: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)

    for row in rows:
        group_key = (
            row.get("answer_model_name", ""),
            row.get("summary_model_name", ""),
            row.get("summary_prompt_variant", ""),
            row.get("stage_target_words", ""),
            row.get("summary_batch_run_id", ""),
        )
        entity_key = (
            row.get("entity_type", ""),
            row.get("answer_model_name", ""),
            row.get("summary_model_name", ""),
            row.get("summary_prompt_variant", ""),
            row.get("stage_target_words", ""),
        )
        group_totals[group_key] += 1
        entity_totals[entity_key] += 1
        is_missing = bool(row.get("error"))
        if is_missing:
            detailed.append({
                **row,
                "analysis_eligible": "False",
                "technical_missing": "True",
                "normalized_exact_or_containment": "",
                "token_f1": "",
                "retention": "",
                "loss": "",
                "reference_discordant_answer": "",
                "unsupported_or_hallucinated": "",
            })
            continue
        answer = row.get("answer", "")
        reference = row.get("reference_answer", "")
        is_retained = retained(reference, answer)
        is_loss = answer.upper() == "NOT_FOUND"
        is_reference_discordant = bool(answer and answer.upper() != "NOT_FOUND" and not is_retained)
        f1 = retention_token_f1(reference, answer)
        detail = {
            **row,
            "analysis_eligible": "True",
            "technical_missing": "False",
            "normalized_exact_or_containment": str(is_retained),
            "token_f1": f"{f1:.6f}",
            "retention": str(is_retained),
            "loss": str(is_loss),
            "reference_discordant_answer": str(is_reference_discordant),
            # Backward-compatible alias for older analysis consumers. It must not be
            # interpreted as a direct hallucination estimate in this experiment.
            "unsupported_or_hallucinated": str(is_reference_discordant),
        }
        detailed.append(detail)
        metric = {
            "retention": 1.0 if is_retained else 0.0,
            "loss": 1.0 if is_loss else 0.0,
            "reference_discordant": 1.0 if is_reference_discordant else 0.0,
            "token_f1": f1,
            "latency": float(row.get("latency_seconds") or 0),
        }
        groups[group_key].append(metric)
        entity_groups[entity_key].append(metric)

    summary_rows: List[Dict[str, str]] = []
    sensitivity_rows: List[Dict[str, str]] = []
    for (answer_model, summary_model, prompt_variant, stage, summary_batch), total in group_totals.items():
        metrics = groups[(answer_model, summary_model, prompt_variant, stage, summary_batch)]
        missing = total - len(metrics)
        retained_count = sum(item["retention"] for item in metrics)
        loss_count = sum(item["loss"] for item in metrics)
        discordant_count = sum(item["reference_discordant"] for item in metrics)
        f1_sum = sum(item["token_f1"] for item in metrics)
        summary_rows.append(
            {
                "answer_model_name": answer_model,
                "summary_model_name": summary_model,
                "summary_prompt_variant": prompt_variant,
                "stage_target_words": stage,
                "summary_batch_run_id": summary_batch,
                "num_expected": str(total),
                "num_answers": str(len(metrics)),
                "num_technical_missing": str(missing),
                "technical_missing_rate": f"{missing / total:.6f}",
                "retention_rate": f"{avg(item['retention'] for item in metrics):.6f}",
                "loss_rate": f"{avg(item['loss'] for item in metrics):.6f}",
                "reference_discordant_answer_rate": f"{avg(item['reference_discordant'] for item in metrics):.6f}",
                "unsupported_or_hallucination_rate": f"{avg(item['reference_discordant'] for item in metrics):.6f}",
                "avg_token_f1": f"{avg(item['token_f1'] for item in metrics):.6f}",
                "answers_per_minute": f"{(60.0 / avg(item['latency'] for item in metrics)) if avg(item['latency'] for item in metrics) else 0.0:.3f}",
            }
        )
        sensitivity_rows.append({
            "answer_model_name": answer_model,
            "summary_model_name": summary_model,
            "summary_prompt_variant": prompt_variant,
            "stage_target_words": stage,
            "summary_batch_run_id": summary_batch,
            "num_expected": str(total),
            "num_analyzed": str(len(metrics)),
            "num_technical_missing": str(missing),
            "retention_complete_case": f"{retained_count / len(metrics):.6f}",
            "retention_all_missing_zero": f"{retained_count / total:.6f}",
            "retention_all_missing_one": f"{(retained_count + missing) / total:.6f}",
            "loss_complete_case": f"{loss_count / len(metrics):.6f}",
            "loss_all_missing_zero": f"{loss_count / total:.6f}",
            "loss_all_missing_one": f"{(loss_count + missing) / total:.6f}",
            "reference_discordant_complete_case": f"{discordant_count / len(metrics):.6f}",
            "reference_discordant_all_missing_zero": f"{discordant_count / total:.6f}",
            "reference_discordant_all_missing_one": f"{(discordant_count + missing) / total:.6f}",
            "token_f1_complete_case": f"{f1_sum / len(metrics):.6f}",
            "token_f1_all_missing_zero": f"{f1_sum / total:.6f}",
            "token_f1_all_missing_one": f"{(f1_sum + missing) / total:.6f}",
        })

    entity_rows = [
        {
            "entity_type": entity_type,
            "answer_model_name": answer_model,
            "summary_model_name": summary_model,
            "summary_prompt_variant": prompt_variant,
            "stage_target_words": stage,
            "num_answers": str(len(metrics)),
            "retention_rate": f"{avg(item['retention'] for item in metrics):.6f}",
            "loss_rate": f"{avg(item['loss'] for item in metrics):.6f}",
            "reference_discordant_answer_rate": f"{avg(item['reference_discordant'] for item in metrics):.6f}",
            "unsupported_or_hallucination_rate": f"{avg(item['reference_discordant'] for item in metrics):.6f}",
        }
        for (entity_type, answer_model, summary_model, prompt_variant, stage), metrics in entity_groups.items()
    ]
    for row in entity_rows:
        key = (
            row["entity_type"], row["answer_model_name"], row["summary_model_name"],
            row["summary_prompt_variant"], row["stage_target_words"],
        )
        total = entity_totals[key]
        row["num_expected"] = total
        row["num_technical_missing"] = total - int(row["num_answers"])

    detailed_fieldnames = list(detailed[0].keys()) if detailed else []
    if detailed_fieldnames:
        write_csv(output_dir / "qa_retention_detailed.csv", detailed, detailed_fieldnames)
    write_csv(
        output_dir / "qa_retention_kpis.csv",
        summary_rows,
        [
            "answer_model_name",
            "summary_model_name",
            "summary_prompt_variant",
            "stage_target_words",
            "summary_batch_run_id",
            "num_expected",
            "num_answers",
            "num_technical_missing",
            "technical_missing_rate",
            "retention_rate",
            "loss_rate",
            "reference_discordant_answer_rate",
            "unsupported_or_hallucination_rate",
            "avg_token_f1",
            "answers_per_minute",
        ],
    )
    write_csv(
        output_dir / "entity_retention_kpis.csv",
        entity_rows,
        [
            "entity_type",
            "answer_model_name",
            "summary_model_name",
            "summary_prompt_variant",
            "stage_target_words",
            "num_expected",
            "num_answers",
            "num_technical_missing",
            "retention_rate",
            "loss_rate",
            "reference_discordant_answer_rate",
            "unsupported_or_hallucination_rate",
        ],
    )
    write_csv(
        output_dir / "qa_missingness_sensitivity.csv",
        sensitivity_rows,
        list(sensitivity_rows[0].keys()),
    )
    return summary_rows, entity_rows


def write_report(output_dir: Path, summary_kpis: List[Dict[str, str]], fact_kpis: List[Dict[str, str]], qa_kpis: List[Dict[str, str]]) -> None:
    lines = ["# Model Selection KPI Report", ""]
    lines.append("## Recommended Discussion Points")
    lines.append("")
    if summary_kpis:
        best_summary = max(summary_kpis, key=lambda row: (float(row["avg_rougeL_fmeasure"]), float(row["word_window_success_rate"])))
        lines.append(
            f"- Best summary ROUGE-L row: {best_summary['model_name']} / {best_summary['prompt_variant']} / "
            f"{best_summary['stage_target_words']} words with ROUGE-L {best_summary['avg_rougeL_fmeasure']}."
        )
    if fact_kpis:
        best_fact = max(
            fact_kpis,
            key=lambda row: (
                float(row["exactly_20_rate"]),
                float(row["avg_source_supported_count"]),
                float(row["avg_reference_answer_supported_count"]),
                float(row["avg_answerable_count"]),
            ),
        )
        lines.append(
            f"- Best fact extractor by schema/support heuristic: {best_fact['model_name']} "
            f"with exactly-20 rate {best_fact['exactly_20_rate']}."
        )
    if qa_kpis:
        best_qa = max(
            qa_kpis,
            key=lambda row: (
                float(row["retention_rate"]),
                -float(row["reference_discordant_answer_rate"]),
                float(row["answers_per_minute"]),
            ),
        )
        lines.append(
            f"- Best answer-from-summary row: answerer {best_qa['answer_model_name']} on "
            f"{best_qa['summary_model_name']} / {best_qa['summary_prompt_variant']} / "
            f"{best_qa['stage_target_words']} words with retention {best_qa['retention_rate']}."
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary_quality_by_stage.csv`",
            "- `summary_quality_detailed.csv`",
            "- `fact_extraction_kpis.csv`",
            "- `qa_retention_kpis.csv`",
            "- `entity_retention_kpis.csv`",
            "- `qa_retention_detailed.csv`",
            "",
            "Use the KPI files for the final professor-facing model-choice table. The automated report is a first pass, not a substitute for the manual spot check of canonical facts.",
        ]
    )
    (output_dir / "professor_model_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_summary_files(patterns: List[str]) -> List[Path]:
    files = []
    for pattern in patterns:
        files.extend(Path(match) for match in sorted(glob.glob(pattern, recursive=True)) if Path(match).name.endswith("_summaries.csv"))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an end-to-end model-selection batch.")
    parser.add_argument("--articles", default="selected_articles.csv")
    parser.add_argument("--summary-file", action="append", dest="summary_files", default=[])
    parser.add_argument("--summary-glob", action="append", dest="summary_globs", default=[])
    parser.add_argument("--fact-summary")
    parser.add_argument("--answers")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance-pct", type=float, default=5.0)
    parser.add_argument("--with-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="distilroberta-base")
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--with-summac", action="store_true")
    parser.add_argument("--summac-model", default="vitc")
    parser.add_argument("--summac-granularity", default="sentence")
    parser.add_argument("--summac-device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_files = [Path(path) for path in args.summary_files]
    for pattern in args.summary_globs:
        summary_files.extend(Path(match) for match in sorted(glob.glob(pattern, recursive=True)) if Path(match).name.endswith("_summaries.csv"))

    _, summary_kpis = (
        score_summaries(
            Path(args.articles),
            summary_files,
            output_dir,
            args.tolerance_pct,
            with_bertscore=args.with_bertscore,
            bertscore_model=args.bertscore_model,
            bertscore_lang=args.bertscore_lang,
            with_summac=args.with_summac,
            summac_model=args.summac_model,
            summac_granularity=args.summac_granularity,
            summac_device=args.summac_device,
        )
        if summary_files
        else ([], [])
    )
    fact_kpis = score_facts(Path(args.fact_summary), output_dir) if args.fact_summary else []
    qa_kpis, _ = score_qa(Path(args.answers), output_dir) if args.answers else ([], [])
    write_report(output_dir, summary_kpis, fact_kpis, qa_kpis)
    print(f"Saved model selection report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
