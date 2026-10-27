#!/usr/bin/env python3
import argparse
import csv
import glob
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from faktenretention.io import read_csv, append_csv, append_jsonl
from faktenretention.protocol import STAGES as TARGETS
from faktenretention.model_transport import ollama_generate


ANSWER_FIELDNAMES = [
    "experiment_run_id",
    "answer_batch_run_id",
    "answer_model_run_id",
    "answer_model_name",
    "summary_batch_run_id",
    "summary_model_run_id",
    "summary_model_name",
    "summary_prompt_variant",
    "article_id",
    "dataset_index",
    "stage_target_words",
    "canonical_fact_id",
    "entity_type",
    "question",
    "reference_answer",
    "answer",
    "support",
    "confidence",
    "answer_status",
    "latency_seconds",
    "total_duration_ns",
    "load_duration_ns",
    "prompt_eval_count",
    "eval_count",
    "eval_duration_ns",
    "api_attempts",
    "error",
]

ANSWER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "support": {"type": "string"},
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    },
    "required": ["answer", "support", "confidence"],
}

TERMINAL_MISSING_ERRORS = {"support_not_exactly_grounded_in_summary"}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "run"


def parse_model_args(values: List[str]) -> List[Dict[str, str]]:
    models = []
    for value in values:
        if "=" in value:
            run_id, model_name = value.split("=", 1)
        else:
            model_name = value
            run_id = slugify(model_name)
        models.append({"run_id": slugify(run_id), "model_name": model_name.strip()})
    return models


def compact_json_text(text: str) -> str:
    stripped = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    return stripped


def parse_answer(text: str) -> Tuple[str, str, str, str, str]:
    raw = compact_json_text(text)
    parse_error = ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                data = {}
                parse_error = "malformed_json"
        else:
            data = {}
            parse_error = "malformed_json"
    if isinstance(data, dict):
        missing_keys = {"answer", "support", "confidence"} - set(data)
        if missing_keys:
            parse_error = "missing_keys:" + ",".join(sorted(missing_keys))
        answer_value = data.get("answer", "")
        support_value = data.get("support", "")
        confidence_value = data.get("confidence", "")
        answer = "" if answer_value is None else str(answer_value).strip()
        support = "" if support_value is None else str(support_value).strip()
        confidence = "" if confidence_value is None else str(confidence_value).strip().upper()
    else:
        answer = raw.strip()
        support = ""
        confidence = ""
        parse_error = "response_is_not_json_object"
    if not answer:
        # Some schema-constrained local models encode an unavailable answer as
        # an empty string even though the prompt requests the literal
        # NOT_FOUND.  Empty answer + empty support is semantically identical
        # and can be normalized deterministically without another model call.
        answer = "NOT_FOUND"
        if support:
            parse_error = parse_error or "missing_answer"
    if answer.upper() == "NOT_FOUND":
        answer = "NOT_FOUND"
        status = "not_found"
    else:
        status = "answered"
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"
        parse_error = parse_error or "invalid_confidence"
    return answer, support, confidence, status, parse_error


def build_prompt(summary: str, question: str) -> str:
    return (
        "You answer factual questions about a generated summary for an evaluation pipeline.\n"
        "Use only the SUMMARY. Do not use outside facts.\n\n"
        "Decision rule:\n"
        "- If the summary gives the requested fact, answer it directly.\n"
        "- Equivalent wording counts as present.\n"
        "- If the summary only gives a vague category where the question asks for a named person, "
        "organization, location, date, number, or event, return NOT_FOUND.\n"
        "- If the summary contradicts the requested fact or gives a different specific value, "
        "return the value stated in the summary, not NOT_FOUND.\n"
        "- Return NOT_FOUND only when the requested fact is absent from the summary.\n\n"
        "Return valid JSON only with exactly these keys: answer, support, confidence.\n"
        "answer must be one short phrase or sentence, never an empty string.\n"
        "For an answered question, support must be the shortest decisive exact contiguous passage "
        "copied character-for-character from the summary. The answer text itself is sufficient "
        "support when it occurs verbatim. Never use ellipses, paraphrase, commentary, Markdown, or "
        "extra quotation-mark characters inside the support value.\n"
        "For NOT_FOUND, answer must be the literal string NOT_FOUND and support must be empty.\n"
        "confidence must be HIGH, MEDIUM, or LOW.\n\n"
        "Examples:\n"
        "SUMMARY: Alice won the race on Tuesday. QUESTION: Who won the race?\n"
        "{\"answer\":\"Alice\",\"support\":\"Alice\",\"confidence\":\"HIGH\"}\n"
        "SUMMARY: Alice won the race. QUESTION: When was the race won?\n"
        "{\"answer\":\"NOT_FOUND\",\"support\":\"\",\"confidence\":\"HIGH\"}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"SUMMARY:\n{summary}"
    )


from faktenretention.scoring import support_is_grounded


def canonicalize_support(summary: str, answer: str, support: str) -> str:
    """Remove model-added quote wrappers only when the interior is exactly grounded."""
    candidate = (support or "").strip()
    if support_is_grounded(summary, answer, candidate):
        return candidate
    quote_pairs = {("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")}
    while len(candidate) >= 2 and (candidate[0], candidate[-1]) in quote_pairs:
        candidate = candidate[1:-1].strip()
        if support_is_grounded(summary, answer, candidate):
            return candidate
    return (support or "").strip()


def recover_support_from_verbatim_answer(summary: str, answer: str, support: str) -> str:
    """Use an answer's own exact summary span when model-supplied support is malformed."""
    canonical = canonicalize_support(summary, answer, support)
    if support_is_grounded(summary, answer, canonical):
        return canonical
    if (answer or "").strip().upper() == "NOT_FOUND":
        return canonical
    answer_tokens = (answer or "").strip().split()
    if not answer_tokens:
        return canonical
    pattern = r"\s+".join(re.escape(token) for token in answer_tokens)
    match = re.search(pattern, summary or "", flags=re.IGNORECASE)
    return match.group(0) if match else canonical


def empty_summary_answer() -> Tuple[str, str, str, str]:
    """Keep empty summaries in the QA denominator without making a model call."""
    return "NOT_FOUND", "", "HIGH", "not_found"


def discover_summary_files(patterns: List[str], files: Optional[List[str]]) -> List[Path]:
    paths = {Path(item) for item in files or []}
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            if path.name.endswith("_summaries.csv"):
                paths.add(path)
    return sorted(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer canonical questions from generated summaries.")
    parser.add_argument("--facts", required=True)
    parser.add_argument("--summary-file", action="append", dest="summary_files")
    parser.add_argument("--summary-glob", action="append", dest="summary_globs", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-run-id", required=True)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--host", default="127.0.0.1:11435")
    parser.add_argument("--limit-articles", type=int)
    parser.add_argument("--limit-facts", type=int)
    parser.add_argument("--limit-summary-files", type=int)
    parser.add_argument("--prompt-variant", action="append", dest="prompt_variants")
    parser.add_argument("--stage", action="append", dest="stages", type=int, choices=TARGETS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--gpt-oss-think-level", default=None)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--accept-terminal-grounding-missing",
        action="store_true",
        help=(
            "Retain exact-support grounding failures after all configured attempts as explicit "
            "missing analysis units instead of requiring an endless resume."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_csv = output_dir / "summary_question_answers.csv"
    output_jsonl = output_dir / "summary_question_answers.jsonl"
    if not args.resume:
        for path in [output_csv, output_jsonl]:
            if path.exists():
                path.unlink()
    existing_rows = read_csv(output_csv) if args.resume and output_csv.exists() else []
    successful_rows = [row for row in existing_rows if not row.get("error")]
    terminal_missing_rows = [
        row for row in existing_rows
        if args.accept_terminal_grounding_missing and row.get("error") in TERMINAL_MISSING_ERRORS
    ]
    retained_rows = successful_rows + terminal_missing_rows
    completed_keys = {
        (
            row["answer_model_name"], row["summary_prompt_variant"], row["article_id"],
            row["stage_target_words"], row["canonical_fact_id"],
        )
        for row in retained_rows
    }
    if args.resume and output_csv.exists():
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ANSWER_FIELDNAMES)
            writer.writeheader()
            writer.writerows(retained_rows)

    facts = read_csv(Path(args.facts))
    if args.limit_articles:
        allowed_articles = []
        for fact in facts:
            article_id = fact.get("article_id", "")
            if article_id not in allowed_articles:
                allowed_articles.append(article_id)
            if len(allowed_articles) >= args.limit_articles:
                break
        facts = [fact for fact in facts if fact.get("article_id", "") in set(allowed_articles)]
    if args.limit_facts:
        by_article: Dict[str, int] = {}
        limited = []
        for fact in facts:
            article_id = fact.get("article_id", "")
            by_article[article_id] = by_article.get(article_id, 0) + 1
            if by_article[article_id] <= args.limit_facts:
                limited.append(fact)
        facts = limited

    facts_by_article: Dict[str, List[Dict[str, str]]] = {}
    for fact in facts:
        facts_by_article.setdefault(fact.get("article_id", ""), []).append(fact)

    summary_files = discover_summary_files(args.summary_globs, args.summary_files)
    if args.limit_summary_files:
        summary_files = summary_files[: args.limit_summary_files]
    if not summary_files:
        print("No summary files found.")
        return 1

    models = parse_model_args(args.models)
    for model in models:
        model_name = model["model_name"]
        model_run_id = model["run_id"]
        answer_batch_run_id = f"{args.experiment_run_id}__{model_run_id}__summary_qa"
        for summary_file in summary_files:
            summary_rows = read_csv(summary_file)
            if args.prompt_variants:
                summary_rows = [
                    row for row in summary_rows
                    if row.get("prompt_variant", "") in set(args.prompt_variants)
                ]
            for summary_row in summary_rows:
                article_id = (summary_row.get("article_id") or "").strip()
                article_facts = facts_by_article.get(article_id, [])
                if not article_facts:
                    continue
                for stage in (args.stages or TARGETS):
                    summary = (summary_row.get(f"summary_{stage}") or "").strip()
                    for fact in article_facts:
                        unit_key = (
                            model_name,
                            summary_row.get("prompt_variant", ""),
                            article_id,
                            str(stage),
                            fact["canonical_fact_id"],
                        )
                        if unit_key in completed_keys:
                            continue
                        if not summary:
                            answer, support, confidence, status = empty_summary_answer()
                            row = {
                                "experiment_run_id": args.experiment_run_id,
                                "answer_batch_run_id": answer_batch_run_id,
                                "answer_model_run_id": model_run_id,
                                "answer_model_name": model_name,
                                "summary_batch_run_id": summary_row.get("batch_run_id", summary_file.stem),
                                "summary_model_run_id": summary_row.get("model_run_id", ""),
                                "summary_model_name": summary_row.get("model_name", ""),
                                "summary_prompt_variant": summary_row.get("prompt_variant", ""),
                                "article_id": article_id,
                                "dataset_index": summary_row.get("dataset_index", ""),
                                "stage_target_words": str(stage),
                                "canonical_fact_id": fact["canonical_fact_id"],
                                "entity_type": fact["entity_type"],
                                "question": fact["question"],
                                "reference_answer": fact["reference_answer"],
                                "answer": answer,
                                "support": support,
                                "confidence": confidence,
                                "answer_status": status,
                                "latency_seconds": "0.000",
                                "total_duration_ns": "",
                                "load_duration_ns": "",
                                "prompt_eval_count": "",
                                "eval_count": "",
                                "eval_duration_ns": "",
                                "api_attempts": "0",
                                "error": "",
                            }
                            append_csv(output_csv, row, ANSWER_FIELDNAMES)
                            append_jsonl(output_jsonl, row)
                            completed_keys.add(unit_key)
                            continue
                        response_data: Dict[str, Any] = {}
                        error = ""
                        api_attempts = 0
                        started = time.time()
                        for api_attempt in range(1, args.api_retries + 1):
                            api_attempts = api_attempt
                            try:
                                think_setting: Any = True if model_name.startswith("qwen3:") else False
                                if args.gpt_oss_think_level is not None and model_name.startswith("gpt-oss:"):
                                    think_setting = args.gpt_oss_think_level
                                response_data = ollama_generate(
                                    model=model_name,
                                    prompt=build_prompt(summary, fact["question"]),
                                    host=args.host,
                                    system_prompt="You are a strict JSON API. Output only valid JSON.",
                                    temperature=args.temperature,
                                    seed=args.seed + api_attempt - 1,
                                    num_ctx=args.num_ctx,
                                    keep_alive=args.keep_alive,
                                    think=think_setting,
                                    num_predict=args.num_predict,
                                    response_format=ANSWER_RESPONSE_SCHEMA,
                                    timeout=args.timeout,
                                )
                                answer, support, confidence, status, parse_error = parse_answer(
                                    response_data.get("response") or ""
                                )
                                support = recover_support_from_verbatim_answer(
                                    summary, answer, support
                                )
                                error = parse_error
                                if not error and not support_is_grounded(summary, answer, support):
                                    error = "support_not_exactly_grounded_in_summary"
                                if not error:
                                    break
                            except Exception as exc:
                                answer, support, confidence, status = "NOT_FOUND", "", "LOW", "error"
                                error = str(exc)
                            if api_attempt < args.api_retries:
                                time.sleep(min(30, 2 ** (api_attempt - 1)))
                        latency = time.time() - started
                        row = {
                            "experiment_run_id": args.experiment_run_id,
                            "answer_batch_run_id": answer_batch_run_id,
                            "answer_model_run_id": model_run_id,
                            "answer_model_name": model_name,
                            "summary_batch_run_id": summary_row.get("batch_run_id", summary_file.stem),
                            "summary_model_run_id": summary_row.get("model_run_id", ""),
                            "summary_model_name": summary_row.get("model_name", ""),
                            "summary_prompt_variant": summary_row.get("prompt_variant", ""),
                            "article_id": article_id,
                            "dataset_index": summary_row.get("dataset_index", ""),
                            "stage_target_words": str(stage),
                            "canonical_fact_id": fact["canonical_fact_id"],
                            "entity_type": fact["entity_type"],
                            "question": fact["question"],
                            "reference_answer": fact["reference_answer"],
                            "answer": answer,
                            "support": support,
                            "confidence": confidence,
                            "answer_status": status,
                            "latency_seconds": f"{latency:.3f}",
                            "total_duration_ns": str(response_data.get("total_duration", "")),
                            "load_duration_ns": str(response_data.get("load_duration", "")),
                            "prompt_eval_count": str(response_data.get("prompt_eval_count", "")),
                            "eval_count": str(response_data.get("eval_count", "")),
                            "eval_duration_ns": str(response_data.get("eval_duration", "")),
                            "api_attempts": str(api_attempts),
                            "error": error,
                        }
                        append_csv(output_csv, row, ANSWER_FIELDNAMES)
                        append_jsonl(output_jsonl, row)
                        if not error:
                            completed_keys.add(unit_key)

    expected_units = 0
    for summary_file in summary_files:
        for summary_row in read_csv(summary_file):
            article_facts = facts_by_article.get((summary_row.get("article_id") or "").strip(), [])
            if args.prompt_variants and summary_row.get("prompt_variant", "") not in set(args.prompt_variants):
                continue
            expected_units += len(article_facts) * len(args.stages or TARGETS) * len(models)
    final_rows = read_csv(output_csv) if output_csv.exists() else []
    successful_final = [row for row in final_rows if not row.get("error")]
    successful_keys_final = {
        (
            row["answer_model_name"], row["summary_prompt_variant"], row["article_id"],
            row["stage_target_words"], row["canonical_fact_id"],
        )
        for row in successful_final
    }
    error_rows = sum(bool(row.get("error")) for row in final_rows)
    terminal_missing_final = [
        row for row in final_rows if row.get("error") in TERMINAL_MISSING_ERRORS
    ]
    unexpected_error_rows = [
        row for row in final_rows
        if row.get("error") and row.get("error") not in TERMINAL_MISSING_ERRORS
    ]
    all_keys_final = {
        (
            row["answer_model_name"], row["summary_prompt_variant"], row["article_id"],
            row["stage_target_words"], row["canonical_fact_id"],
        )
        for row in final_rows
    }
    complete = (
        len(all_keys_final) == expected_units
        and len(final_rows) == expected_units
        and not unexpected_error_rows
        and (
            error_rows == 0
            or (
                args.accept_terminal_grounding_missing
                and error_rows == len(terminal_missing_final)
            )
        )
    )
    manifest = {
        "models": models,
        "num_ctx": args.num_ctx,
        "temperature": args.temperature,
        "base_seed": args.seed,
        "retry_seeds": [args.seed + offset for offset in range(args.api_retries)],
        "api_retries": args.api_retries,
        "expected_units": expected_units,
        "successful_unique_units": len(successful_keys_final),
        "error_rows": error_rows,
        "terminal_missing_units": len(terminal_missing_final),
        "unexpected_error_rows": len(unexpected_error_rows),
        "all_expected_units_processed": len(all_keys_final) == expected_units,
        "fully_observed": error_rows == 0,
        "missing_data_policy": (
            "exact-support failures after all configured attempts remain technical missing values; "
            "they are not relabeled as retained, lost, or reference-discordant"
            if terminal_missing_final else "no missing QA units"
        ),
        "missing_data_amendment_timing": (
            "post-hoc after the complete QA collection and before method-outcome scoring"
            if terminal_missing_final else "not applicable"
        ),
        "prompt_variants": args.prompt_variants or "all",
        "stages": args.stages or TARGETS,
        "answer_prompt_protocol": "human-validation-v2 plus exact contiguous support gate",
        "answer_prompt_sha256": hashlib.sha256(
            build_prompt("__SUMMARY__", "__QUESTION__").encode("utf-8")
        ).hexdigest(),
        "verbatim_answer_support_recovery": True,
        "complete": complete,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved summary Q/A answers to {output_csv}")
    if not complete:
        raise RuntimeError(
            "Summary QA completion gate failed. "
            f"Expected {expected_units}, got {len(successful_keys_final)} successful unique units "
            f"and {len(terminal_missing_final)} eligible terminal missing units."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
