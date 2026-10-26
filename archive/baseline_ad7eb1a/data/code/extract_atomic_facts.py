#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from iterative_summarize import apply_model_prompt_controls, ollama_generate


ENTITY_TYPES = {"PERSON", "ORG", "LOCATION", "DATE_TIME", "NUMBER", "EVENT", "OBJECT", "OTHER"}

FACT_FIELDNAMES = [
    "experiment_run_id",
    "batch_run_id",
    "model_run_id",
    "model_name",
    "article_id",
    "dataset_index",
    "fact_id",
    "fact",
    "question",
    "reference_answer",
    "entity_type",
    "source_evidence",
    "is_valid_schema",
    "is_source_supported",
    "is_reference_answer_supported",
    "is_duplicate",
    "is_answerable",
    "latency_seconds",
    "total_duration_ns",
    "load_duration_ns",
    "prompt_eval_count",
    "eval_count",
    "eval_duration_ns",
    "error",
]

SUMMARY_FIELDNAMES = [
    "experiment_run_id",
    "batch_run_id",
    "model_run_id",
    "model_name",
    "article_id",
    "dataset_index",
    "raw_fact_count",
    "valid_schema_count",
    "source_supported_count",
    "reference_answer_supported_count",
    "duplicate_count",
    "answerable_count",
    "entity_type_count",
    "generation_attempts",
    "exactly_20",
    "selected_for_canonical",
    "error",
]

CANONICAL_FIELDNAMES = [
    "experiment_run_id",
    "canonical_source_model_run_id",
    "canonical_source_model_name",
    "article_id",
    "dataset_index",
    "canonical_fact_id",
    "source_fact_id",
    "fact",
    "question",
    "reference_answer",
    "entity_type",
    "source_evidence",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "run"


def load_articles(path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    from input_checks import validate_article_keys
    validate_article_keys(rows)
    return rows[:limit] if limit else rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Dict[str, str], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\W_]+", " ", text.casefold(), flags=re.UNICODE)).strip()


def compact_json_text(text: str) -> str:
    stripped = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    return stripped


def parse_json_payload(text: str) -> Tuple[List[Dict[str, Any]], str]:
    raw = compact_json_text(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.JSONDecoder(strict=False).decode(raw)
        except json.JSONDecodeError:
            data = None
    if data is None:
        start_candidates = [idx for idx in [raw.find("["), raw.find("{")] if idx >= 0]
        if not start_candidates:
            return [], "no_json_start"
        start = min(start_candidates)
        end = max(raw.rfind("]"), raw.rfind("}"))
        if end <= start:
            return [], "no_json_end"
        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError as exc:
            try:
                data = json.JSONDecoder(strict=False).decode(snippet)
            except json.JSONDecodeError:
                repaired = re.sub(r",(\s*[\]}])", r"\1", snippet)
                try:
                    data = json.JSONDecoder(strict=False).decode(repaired)
                except json.JSONDecodeError:
                    return [], f"json_decode_error: {exc}"

    if isinstance(data, dict):
        for key in ("facts", "atomic_facts", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return [], "json_not_list"
    return [item for item in data if isinstance(item, dict)], ""


def build_prompt(article: str, facts_per_article: int) -> str:
    return (
        "Extract exactly "
        f"{facts_per_article} important atomic facts from the news article.\n"
        "For each fact, create a question and a short reference answer that can be answered "
        "from that one fact. Use only information explicitly stated in the article.\n\n"
        "Choose the most important facts: major actors, actions, dates, outcomes, causes, numbers, "
        "locations, and named events. Keep each fact atomic; do not combine multiple claims.\n"
        "Entity type must be one of: PERSON, ORG, LOCATION, DATE_TIME, NUMBER, EVENT, OBJECT, OTHER.\n"
        "Return JSON only, as an array of objects. Do not write notes, analysis, markdown, or a preface.\n"
        "The first character of your response must be [ and the last character must be ].\n"
        "Each object must have exactly these keys:\n"
        "fact_id, fact, question, reference_answer, entity_type, source_evidence.\n"
        "source_evidence must be one exact contiguous quote from the article that supports the fact. "
        "reference_answer must be a short exact contiguous span copied from source_evidence.\n\n"
        "ARTICLE:\n"
        f"{article}"
    )


def validate_fact(item: Dict[str, Any], article: str, seen: Counter) -> Dict[str, str]:
    fact = str(item.get("fact", "")).strip()
    question = str(item.get("question", "")).strip()
    answer = str(item.get("reference_answer", "")).strip()
    entity_type = str(item.get("entity_type", "")).strip().upper()
    evidence = str(item.get("source_evidence", "")).strip()
    fact_id = str(item.get("fact_id", "")).strip()

    norm_fact = normalize_text(fact)
    seen[norm_fact] += 1
    is_duplicate = seen[norm_fact] > 1 if norm_fact else True
    article_norm = normalize_text(article)
    evidence_norm = normalize_text(evidence)
    is_source_supported = bool(evidence_norm and evidence_norm in article_norm)
    answer_norm = normalize_text(answer)
    is_reference_answer_supported = bool(
        is_source_supported and answer_norm and answer_norm in evidence_norm
    )
    is_answerable = bool(question.endswith("?") and answer and answer.upper() != "NOT_FOUND")
    is_valid_schema = bool(
        fact
        and question
        and answer
        and evidence
        and entity_type in ENTITY_TYPES
        and fact_id
    )

    return {
        "fact_id": fact_id,
        "fact": fact,
        "question": question,
        "reference_answer": answer,
        "entity_type": entity_type if entity_type in ENTITY_TYPES else "OTHER",
        "source_evidence": evidence,
        "is_valid_schema": str(is_valid_schema),
        "is_source_supported": str(is_source_supported),
        "is_reference_answer_supported": str(is_reference_answer_supported),
        "is_duplicate": str(is_duplicate),
        "is_answerable": str(is_answerable),
    }


def model_score(
    rows: Iterable[Dict[str, str]], raw_count: int, preferred: bool, required_count: int
) -> Tuple[int, int, int, int, int]:
    row_list = list(rows)
    valid = sum(row["is_valid_schema"] == "True" for row in row_list)
    supported = sum(row["is_source_supported"] == "True" for row in row_list)
    answer_supported = sum(row["is_reference_answer_supported"] == "True" for row in row_list)
    answerable = sum(row["is_answerable"] == "True" for row in row_list)
    duplicates = sum(row["is_duplicate"] == "True" for row in row_list)
    exact = int(raw_count >= required_count)
    preferred_bonus = 1 if preferred else 0
    return exact, preferred_bonus, valid + supported + answer_supported + answerable, -duplicates, len({row["entity_type"] for row in row_list})


def select_canonical(
    fact_rows: List[Dict[str, str]],
    summaries: List[Dict[str, str]],
    preferred_model: str,
    experiment_run_id: str,
    facts_per_article: int,
) -> List[Dict[str, str]]:
    by_article_model: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    raw_counts: Dict[Tuple[str, str], int] = {}
    for row in fact_rows:
        key = (row["article_id"], row["model_name"])
        by_article_model[key].append(row)
    for row in summaries:
        raw_counts[(row["article_id"], row["model_name"])] = int(row["raw_fact_count"] or "0")

    article_ids = sorted({row["article_id"] for row in fact_rows}, key=lambda value: int(value) if value.isdigit() else value)
    canonical_rows: List[Dict[str, str]] = []
    selected_keys = set()
    for article_id in article_ids:
        candidates = [(key, rows) for key, rows in by_article_model.items() if key[0] == article_id]
        if not candidates:
            continue
        best_key, best_rows = max(
            candidates,
            key=lambda item: model_score(
                item[1],
                raw_counts.get(item[0], len(item[1])),
                item[0][1] == preferred_model,
                facts_per_article,
            ),
        )
        selected_keys.add(best_key)
        usable = [
            row
            for row in best_rows
            if row["is_valid_schema"] == "True"
            and row["is_source_supported"] == "True"
            and row["is_reference_answer_supported"] == "True"
            and row["is_duplicate"] != "True"
            and row["is_answerable"] == "True"
        ]
        for index, row in enumerate(usable[:facts_per_article], start=1):
            canonical_rows.append(
                {
                    "experiment_run_id": experiment_run_id,
                    "canonical_source_model_run_id": row["model_run_id"],
                    "canonical_source_model_name": row["model_name"],
                    "article_id": row["article_id"],
                    "dataset_index": row["dataset_index"],
                    "canonical_fact_id": f"{row['article_id']}_F{index:02d}",
                    "source_fact_id": row["fact_id"],
                    "fact": row["fact"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "entity_type": row["entity_type"],
                    "source_evidence": row["source_evidence"],
                }
            )

    for row in summaries:
        key = (row["article_id"], row["model_name"])
        row["selected_for_canonical"] = str(key in selected_keys)
    return canonical_rows


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate atomic fact and reference Q/A candidates.")
    parser.add_argument("--articles", default="selected_articles.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-run-id", required=True)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--preferred-canonical-model", default="gemma4:31b")
    parser.add_argument("--host", default="127.0.0.1:11435")
    parser.add_argument("--facts-per-article", type=int, default=20)
    parser.add_argument(
        "--candidate-pool-size", type=int,
        help="Request this many candidates, then retain exactly facts-per-article strict-valid facts",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--qwen3-native-think", action="store_true")
    parser.add_argument("--gpt-oss-think-level", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_pool_size = args.candidate_pool_size or args.facts_per_article
    if candidate_pool_size < args.facts_per_article:
        raise ValueError("candidate-pool-size must be at least facts-per-article")
    output_dir = Path(args.output_dir)
    articles = load_articles(Path(args.articles), args.limit)
    models = parse_model_args(args.models)

    facts_csv = output_dir / "candidate_facts.csv"
    facts_jsonl = output_dir / "candidate_facts.jsonl"
    summary_csv = output_dir / "fact_extraction_summary.csv"
    canonical_csv = output_dir / "canonical_facts.csv"
    canonical_jsonl = output_dir / "canonical_facts.jsonl"
    errors_jsonl = output_dir / "fact_extraction_errors.jsonl"
    if not args.resume:
        for path in [facts_csv, facts_jsonl, summary_csv, canonical_csv, canonical_jsonl, errors_jsonl]:
            if path.exists():
                path.unlink()

    fact_rows: List[Dict[str, str]] = read_csv(facts_csv) if args.resume and facts_csv.exists() else []
    summary_rows: List[Dict[str, str]] = read_csv(summary_csv) if args.resume and summary_csv.exists() else []
    strict_counts = Counter(
        (row["model_name"], row["article_id"])
        for row in fact_rows
        if row.get("is_valid_schema") == "True"
        and row.get("is_source_supported") == "True"
        and row.get("is_reference_answer_supported") == "True"
        and row.get("is_duplicate") != "True"
        and row.get("is_answerable") == "True"
    )
    successful_keys = {
        (row["model_name"], row["article_id"])
        for row in summary_rows
        if not row.get("error") and int(row.get("raw_fact_count") or 0) >= args.facts_per_article
        and strict_counts[(row["model_name"], row["article_id"])] >= args.facts_per_article
    }
    # Failed units are retried. Remove their tabular partial rows first; JSONL remains an
    # append-only attempt audit and is intentionally not rewritten.
    fact_rows = [row for row in fact_rows if (row["model_name"], row["article_id"]) in successful_keys]
    summary_rows = [row for row in summary_rows if (row["model_name"], row["article_id"]) in successful_keys]
    if args.resume:
        write_csv(facts_csv, fact_rows, FACT_FIELDNAMES)
        write_csv(summary_csv, summary_rows, SUMMARY_FIELDNAMES)

    for model in models:
        model_name = model["model_name"]
        model_run_id = model["run_id"]
        for article_row in articles:
            article = (article_row.get("article") or "").strip()
            article_id = (article_row.get("article_id") or "").strip()
            dataset_index = (article_row.get("dataset_index") or "").strip()
            if (model_name, article_id) in successful_keys:
                print(f"SKIP completed facts model={model_name} article={article_id}", flush=True)
                continue
            batch_run_id = f"{args.experiment_run_id}__{model_run_id}__facts"
            error = ""
            raw_items: List[Dict[str, Any]] = []
            response_data: Dict[str, Any] = {}
            started = time.time()
            attempts_used = 0
            for attempt_index in range(1, args.max_attempts + 1):
                attempts_used = attempt_index
                response_data = {}
                try:
                    prompt = apply_model_prompt_controls(model_name, build_prompt(article, candidate_pool_size), False)
                    think_setting: Any = False
                    if args.qwen3_native_think and model_name.startswith("qwen3:"):
                        think_setting = True
                    if args.gpt_oss_think_level is not None and model_name.startswith("gpt-oss:"):
                        think_setting = args.gpt_oss_think_level
                    response_data = ollama_generate(
                        model=model_name,
                        prompt=prompt,
                        host=args.host,
                        system_prompt=(
                            "You are a strict JSON API. Output only valid JSON. "
                            "Do not reveal reasoning or planning."
                        ),
                        temperature=args.temperature,
                        seed=args.seed + attempt_index - 1,
                        num_ctx=args.num_ctx,
                        keep_alive=args.keep_alive,
                        think=think_setting,
                        num_predict=args.num_predict,
                        timeout=args.timeout,
                    )
                    raw_items, parse_error = parse_json_payload(response_data.get("response") or "")
                    error = parse_error
                    preview_seen: Counter = Counter()
                    preview_rows = []
                    for preview_index, raw_item in enumerate(raw_items, 1):
                        prepared = dict(raw_item)
                        if not str(prepared.get("fact_id") or "").strip():
                            prepared["fact_id"] = f"preview_{preview_index:02d}"
                        preview_rows.append(validate_fact(prepared, article, preview_seen))
                    strict_preview_count = sum(
                        row["is_valid_schema"] == "True"
                        and row["is_source_supported"] == "True"
                        and row["is_reference_answer_supported"] == "True"
                        and row["is_duplicate"] != "True"
                        and row["is_answerable"] == "True"
                        for row in preview_rows
                    )
                    if (
                        not error
                        and len(raw_items) >= args.facts_per_article
                        and strict_preview_count >= args.facts_per_article
                    ):
                        break
                    if not error:
                        error = (
                            f"incomplete_fact_gate:raw={len(raw_items)},"
                            f"strict_supported={strict_preview_count},"
                            f"required={args.facts_per_article}"
                        )
                except Exception as exc:
                    raw_items = []
                    error = str(exc)
                append_jsonl(
                    errors_jsonl,
                    {
                        "experiment_run_id": args.experiment_run_id,
                        "batch_run_id": batch_run_id,
                        "model_run_id": model_run_id,
                        "model_name": model_name,
                        "article_id": article_id,
                        "dataset_index": dataset_index,
                        "attempt_index": attempt_index,
                        "error": error,
                        "response_preview": (response_data.get("response") or "")[:4000],
                        "thinking_preview": (response_data.get("thinking") or "")[:2000],
                    },
                )
            latency = time.time() - started

            seen: Counter = Counter()
            article_rows: List[Dict[str, str]] = []
            for index, item in enumerate(raw_items, start=1):
                prepared = dict(item)
                if not str(prepared.get("fact_id") or "").strip():
                    prepared["fact_id"] = f"{article_id}_{model_run_id}_{index:02d}"
                validated = validate_fact(prepared, article, seen)
                row = {
                    "experiment_run_id": args.experiment_run_id,
                    "batch_run_id": batch_run_id,
                    "model_run_id": model_run_id,
                    "model_name": model_name,
                    "article_id": article_id,
                    "dataset_index": dataset_index,
                    **validated,
                    "latency_seconds": f"{latency:.3f}",
                    "total_duration_ns": str(response_data.get("total_duration", "")),
                    "load_duration_ns": str(response_data.get("load_duration", "")),
                    "prompt_eval_count": str(response_data.get("prompt_eval_count", "")),
                    "eval_count": str(response_data.get("eval_count", "")),
                    "eval_duration_ns": str(response_data.get("eval_duration", "")),
                    "error": error,
                }
                fact_rows.append(row)
                article_rows.append(row)
                append_csv(facts_csv, row, FACT_FIELDNAMES)
                append_jsonl(facts_jsonl, row)

            summary_row = {
                "experiment_run_id": args.experiment_run_id,
                "batch_run_id": batch_run_id,
                "model_run_id": model_run_id,
                "model_name": model_name,
                "article_id": article_id,
                "dataset_index": dataset_index,
                "raw_fact_count": str(len(raw_items)),
                "valid_schema_count": str(sum(row["is_valid_schema"] == "True" for row in article_rows)),
                "source_supported_count": str(sum(row["is_source_supported"] == "True" for row in article_rows)),
                "reference_answer_supported_count": str(sum(row["is_reference_answer_supported"] == "True" for row in article_rows)),
                "duplicate_count": str(sum(row["is_duplicate"] == "True" for row in article_rows)),
                "answerable_count": str(sum(row["is_answerable"] == "True" for row in article_rows)),
                "entity_type_count": str(len({row["entity_type"] for row in article_rows})),
                "generation_attempts": str(attempts_used),
                # Backward-compatible field name: True now means the candidate pool
                # contains enough strict-valid items to select the required 20.
                "exactly_20": str(sum(
                    row["is_valid_schema"] == "True"
                    and row["is_source_supported"] == "True"
                    and row["is_reference_answer_supported"] == "True"
                    and row["is_duplicate"] != "True"
                    and row["is_answerable"] == "True"
                    for row in article_rows
                ) >= args.facts_per_article),
                "selected_for_canonical": "False",
                "error": error,
            }
            summary_rows.append(summary_row)
            append_csv(summary_csv, summary_row, SUMMARY_FIELDNAMES)

    canonical_rows = select_canonical(
        fact_rows=fact_rows,
        summaries=summary_rows,
        preferred_model=args.preferred_canonical_model,
        experiment_run_id=args.experiment_run_id,
        facts_per_article=args.facts_per_article,
    )
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(canonical_csv, canonical_rows, CANONICAL_FIELDNAMES)
    with canonical_jsonl.open("w", encoding="utf-8") as handle:
        for row in canonical_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    canonical_counts = Counter(row["article_id"] for row in canonical_rows)
    incomplete = {
        row["article_id"]: canonical_counts.get(row["article_id"], 0)
        for row in articles
        if canonical_counts.get(row["article_id"], 0) != args.facts_per_article
    }
    manifest = {
        "articles": len(articles),
        "models": models,
        "num_ctx": args.num_ctx,
        "temperature": args.temperature,
        "base_seed": args.seed,
        "facts_per_article": args.facts_per_article,
        "candidate_pool_size": candidate_pool_size,
        "max_attempts": args.max_attempts,
        "canonical_fact_count": len(canonical_rows),
        "canonical_counts_by_article": dict(sorted(canonical_counts.items())),
        "complete": not incomplete,
        "incomplete_articles": incomplete,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if incomplete:
        raise RuntimeError(
            "Canonical fact completion gate failed; rerun with --resume. "
            f"Expected {args.facts_per_article} per article, observed {incomplete}"
        )

    print(f"Saved candidate facts to {facts_csv}")
    print(f"Saved canonical facts to {canonical_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
