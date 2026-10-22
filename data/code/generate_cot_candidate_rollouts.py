#!/usr/bin/env python3
"""Generate restart-safe recursive Seed zero-shot-CoT candidate rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summary_rules import seed_raw
from iterative_summarize import (
    TARGETS,
    bounds_for_target,
    build_prompt,
    count_words,
    extract_text,
    extract_thinking,
    ollama_generate,
)


MODEL = "thesis-seed:v1"
MODEL_DIGEST = "7a66a2f466bf48fdafa7004a7975a7f5fac6e667a6de7d01751aacb98b3f387c"
TEMPERATURE = 1.1
TOP_P = 0.95
REPEAT_PENALTY = 1.0
NUM_PREDICT = 2400
SEED = 32
TOLERANCE_PCT = 5.0
PROMPT_TEMPLATE = "summarize in {target_word_length}, think step by step"
MIN_SUBSTANTIVE_RATIONALE_WORDS = 25


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_sentence(text: str) -> bool:
    return bool(re.search(r'[.!?]["\'\)\]”’»›]*\s*$', (text or "").strip()))


def substantive_rationale_text(text: str) -> str:
    """Remove Seed's bookkeeping-only token-budget markers before CoT validation."""
    cleaned = re.sub(
        r"<seed:cot_budget_reflect>.*?</seed:cot_budget_reflect>",
        " ",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def api_call(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    rendered = seed_raw(prompt, True, reflection_interval=args.reflection_interval)
    last_error: Exception | None = None
    for retry in range(args.api_retries):
        try:
            return ollama_generate(
                model=MODEL,
                prompt=rendered,
                host=args.host,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repeat_penalty=REPEAT_PENALTY,
                seed=SEED,
                num_ctx=args.num_ctx,
                keep_alive=args.keep_alive,
                num_predict=NUM_PREDICT,
                raw=True,
                timeout=args.timeout,
            )
        except Exception as exc:  # recorded and retried; final exception is propagated
            last_error = exc
            if retry + 1 < args.api_retries:
                time.sleep(min(30, 2 ** retry))
    assert last_error is not None
    raise last_error


def initial_prompt(source: str, target: int) -> str:
    return build_prompt(
        source,
        target,
        TOLERANCE_PCT,
        instruction_template=PROMPT_TEMPLATE,
    )


def revision_prompt(
    source: str,
    desired_target: int,
    control_target: int,
    prior_words: int,
) -> str:
    low, high = bounds_for_target(control_target, TOLERANCE_PCT)
    return (
        f"Task: Generate a new summary of SOURCE in {control_target} words, thinking step by step before the final answer.\n"
        f"A previous independent attempt contained {prior_words} words. Do not copy that attempt; "
        f"write a fresh summary from SOURCE and aim for the operational interval {low}-{high}.\n"
        "Preserve the most important supported facts, remove lower-priority details as needed, and do "
        "not add facts. Internally verify the length before answering.\n"
        "Output plain prose only after the exact label FINAL_SUMMARY:. Do not put text after it.\n\n"
        f"SOURCE:\n{source}"
    )


def generate_stage(
    source: str,
    target: int,
    args: argparse.Namespace,
    attempts_log: Path | None = None,
    article_key: str = "",
    run_instance: str = "",
) -> dict[str, Any]:
    low, high = bounds_for_target(target, TOLERANCE_PCT)
    attempts: list[dict[str, Any]] = []
    previous = ""
    control_target = target
    for attempt_index in range(1, args.max_attempts + 1):
        prompt = (
            initial_prompt(source, target)
            if attempt_index == 1
            else revision_prompt(source, target, control_target, count_words(previous))
        )
        started = time.monotonic()
        response_data = api_call(prompt, args)
        response = extract_text(response_data).strip()
        thinking = extract_thinking(response_data).strip()
        substantive_thinking = substantive_rationale_text(thinking)
        words = count_words(response)
        attempt = {
            "attempt_index": attempt_index,
            "phase": "initial_zero_shot_cot" if attempt_index == 1 else "length_revision_zero_shot_cot",
            "desired_target_words": target,
            "operational_control_target_words": control_target,
            "prompt": prompt,
            "response": response,
            "thinking": thinking,
            "response_word_count": words,
            "thinking_word_count": count_words(thinking),
            "substantive_thinking_word_count": count_words(substantive_thinking),
            "substantive_rationale_success": (
                count_words(substantive_thinking) >= args.min_substantive_rationale_words
            ),
            "word_window_success": low <= words <= high,
            "ends_with_complete_sentence": complete_sentence(response),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "done_reason": response_data.get("done_reason"),
            "usage": {
                key: response_data.get(key)
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
            },
        }
        attempts.append(attempt)
        if attempts_log is not None:
            append_jsonl_durable(
                attempts_log,
                {
                    "run_instance": run_instance,
                    "recorded_at_utc": now_utc(),
                    "article_key": article_key,
                    "model": MODEL,
                    "model_digest": MODEL_DIGEST,
                    "num_ctx": args.num_ctx,
                    "target_words": target,
                    "input_word_count": count_words(source),
                    **attempt,
                },
            )
        previous = response
        rationale_ok = (
            args.allow_nonsubstantive_rationale
            or attempt["substantive_rationale_success"]
        )
        if response and low <= words <= high and complete_sentence(response) and rationale_ok:
            break
        if words and attempt_index >= 2:
            control_target = round(control_target * target / words)
            control_target = max(round(target * 0.35), min(round(target * 1.75), control_target))

    eligible = [
        item
        for item in attempts
        if item["response"]
        and item["word_window_success"]
        and item["ends_with_complete_sentence"]
        and (
            args.allow_nonsubstantive_rationale
            or item["substantive_rationale_success"]
        )
    ]
    if eligible:
        selected = eligible[0]
        selection_reason = (
            "first attempt satisfying word window, complete-sentence, and substantive-rationale constraints"
            if not args.allow_nonsubstantive_rationale
            else "first attempt satisfying word window and complete-sentence constraints"
        )
    else:
        nonempty = [item for item in attempts if item["response"]]
        if not nonempty:
            raise RuntimeError(f"All generation attempts were empty for target={target}")
        selected = min(
            nonempty,
            key=lambda item: (
                abs(int(item["response_word_count"]) - target),
                int(item["attempt_index"]),
            ),
        )
        selection_reason = "closest non-empty attempt; no attempt passed all hard constraints"
    return {
        "target_words": target,
        "accepted_min_words": low,
        "accepted_max_words": high,
        "selected_attempt_index": selected["attempt_index"],
        "selection_reason": selection_reason,
        "selected": selected,
        "attempts": attempts,
    }


def aggregate(input_rows: list[dict[str, str]], article_dir: Path, output_dir: Path) -> None:
    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for source_row in input_rows:
        key = f"{source_row['split']}__{source_row['dataset_index']}"
        path = article_dir / f"{key}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        summary_row: dict[str, Any] = {
            "article_id": source_row["article_id"],
            "dataset": source_row["dataset"],
            "config": source_row["config"],
            "split": source_row["split"],
            "dataset_index": source_row["dataset_index"],
            "article_sha256": source_row["article_sha256"],
            "original_word_count": count_words(source_row["article"]),
            "highlights": source_row["highlights"],
        }
        for stage in record["stages"]:
            target = int(stage["target_words"])
            selected = stage["selected"]
            summary_row[f"summary_{target}"] = selected["response"]
            summary_row[f"summary_{target}_word_count"] = selected["response_word_count"]
            candidates.append(
                {
                    "candidate_id": f"{source_row['split']}:{source_row['dataset_index']}:{target}",
                    "split": source_row["split"],
                    "dataset_index": source_row["dataset_index"],
                    "article_id": source_row["article_id"],
                    "article_sha256": source_row["article_sha256"],
                    "target_words": target,
                    "input_word_count": count_words(stage["input_text"]),
                    "input_text": stage["input_text"],
                    "thinking": selected["thinking"],
                    "thinking_word_count": selected["thinking_word_count"],
                    "substantive_thinking_word_count": selected["substantive_thinking_word_count"],
                    "substantive_rationale_success": selected["substantive_rationale_success"],
                    "response": selected["response"],
                    "response_word_count": selected["response_word_count"],
                    "word_window_success": selected["word_window_success"],
                    "ends_with_complete_sentence": selected["ends_with_complete_sentence"],
                    "selected_attempt_index": stage["selected_attempt_index"],
                    "selected_operational_control_target_words": selected["operational_control_target_words"],
                    "selection_reason": stage["selection_reason"],
                    "highlights": source_row["highlights"],
                }
            )
        summaries.append(summary_row)

    summary_fields = [
        "article_id", "dataset", "config", "split", "dataset_index", "article_sha256",
        "original_word_count", "highlights",
    ]
    for target in TARGETS:
        summary_fields.extend([f"summary_{target}", f"summary_{target}_word_count"])
    with (output_dir / "recursive_rollouts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "stage_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def automate_eligible_counts(stage_candidates_path: Path) -> dict[int, int]:
    """Count structural Automate-CoT eligibility without reference or model-based scores."""
    from score_and_prune_automatecot_candidates import hard_filter

    counts = {target: 0 for target in TARGETS}
    if not stage_candidates_path.exists():
        return counts
    with stage_candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = int(row["target_words"])
            if not hard_filter(row):
                counts[target] += 1
    return counts


def auto_eligible_counts(stage_candidates_path: Path) -> dict[int, int]:
    """Count Auto-CoT eligibility without embeddings or reference-quality signals."""
    from select_autocot_demonstrations import quality_eligible

    counts = {target: 0 for target in TARGETS}
    if not stage_candidates_path.exists():
        return counts
    with stage_candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = int(row["target_words"])
            if quality_eligible(row)[0]:
                counts[target] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1:11435")
    parser.add_argument("--num-ctx", type=int, default=524288)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--reflection-interval", type=int, default=512)
    parser.add_argument("--min-substantive-rationale-words", type=int, default=MIN_SUBSTANTIVE_RATIONALE_WORDS)
    parser.add_argument(
        "--allow-nonsubstantive-rationale",
        action="store_true",
        help="Do not require substantive CoT when rollouts are used only as validation inputs.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-articles-before-eligibility-stop", type=int)
    parser.add_argument("--minimum-eligible-per-stage", type=int)
    parser.add_argument("--minimum-auto-eligible-per-stage", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    article_dir = args.output_dir / "articles"
    article_dir.mkdir(exist_ok=True)
    with args.input.open(encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    if not input_rows:
        raise RuntimeError("No input rows")

    started_at = now_utc()
    attempts_log = args.output_dir / "attempts.jsonl"
    for position, row in enumerate(input_rows, 1):
        key = f"{row['split']}__{row['dataset_index']}"
        article_path = article_dir / f"{key}.json"
        if article_path.exists():
            print(f"SKIP completed {position}/{len(input_rows)} {key}", flush=True)
            continue
        current = row["article"].strip()
        stages: list[dict[str, Any]] = []
        for target in TARGETS:
            print(
                f"START {position}/{len(input_rows)} key={key} "
                f"input_words={count_words(current)} target={target}",
                flush=True,
            )
            stage = generate_stage(
                current,
                target,
                args,
                attempts_log=attempts_log,
                article_key=key,
                run_instance=started_at,
            )
            stage["input_text"] = current
            stages.append(stage)
            current = stage["selected"]["response"]
            print(
                f"END key={key} target={target} output_words={count_words(current)} "
                f"attempt={stage['selected_attempt_index']} "
                f"window={stage['selected']['word_window_success']}",
                flush=True,
            )
        write_json_atomic(
            article_path,
            {
                "created_at_utc": now_utc(),
                "source": row,
                "model": MODEL,
                "model_digest": MODEL_DIGEST,
                "num_ctx": args.num_ctx,
                "generation": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repeat_penalty": REPEAT_PENALTY,
                    "num_predict": NUM_PREDICT,
                    "seed": SEED,
                    "tolerance_pct": TOLERANCE_PCT,
                    "prompt_template": PROMPT_TEMPLATE,
                    "max_attempts": args.max_attempts,
                    "reflection_interval": args.reflection_interval,
                    "min_substantive_rationale_words": args.min_substantive_rationale_words,
                    "allow_nonsubstantive_rationale": args.allow_nonsubstantive_rationale,
                },
                "stages": stages,
            },
        )
        aggregate(input_rows, article_dir, args.output_dir)
        if (
            args.minimum_articles_before_eligibility_stop is not None
            and args.minimum_eligible_per_stage is not None
            and len(list(article_dir.glob("*.json")))
            >= args.minimum_articles_before_eligibility_stop
        ):
            eligible_counts = automate_eligible_counts(args.output_dir / "stage_candidates.jsonl")
            auto_counts = auto_eligible_counts(args.output_dir / "stage_candidates.jsonl")
            print(f"ELIGIBILITY automate={eligible_counts} auto={auto_counts}", flush=True)
            if all(
                eligible_counts[target] >= args.minimum_eligible_per_stage
                for target in TARGETS
            ) and (
                args.minimum_auto_eligible_per_stage is None
                or all(
                    auto_counts[target] >= args.minimum_auto_eligible_per_stage
                    for target in TARGETS
                )
            ):
                print(
                    "STOP eligibility target reached after "
                    f"{len(list(article_dir.glob('*.json')))} articles",
                    flush=True,
                )
                break

    aggregate(input_rows, article_dir, args.output_dir)
    final_automate_counts = automate_eligible_counts(args.output_dir / "stage_candidates.jsonl")
    final_auto_counts = auto_eligible_counts(args.output_dir / "stage_candidates.jsonl")
    adaptive_gate_requested = (
        args.minimum_articles_before_eligibility_stop is not None
        and args.minimum_eligible_per_stage is not None
    )
    adaptive_gate_passed = (
        not adaptive_gate_requested
        or (
            len(list(article_dir.glob("*.json"))) >= args.minimum_articles_before_eligibility_stop
            and all(final_automate_counts[target] >= args.minimum_eligible_per_stage for target in TARGETS)
            and (
                args.minimum_auto_eligible_per_stage is None
                or all(final_auto_counts[target] >= args.minimum_auto_eligible_per_stage for target in TARGETS)
            )
        )
    )
    manifest = {
        "started_at_utc": started_at,
        "completed_at_utc": now_utc(),
        "status": "complete" if adaptive_gate_passed else "incomplete_eligibility_gate",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "articles_available": len(input_rows),
        "articles_completed": len(list(article_dir.glob("*.json"))),
        "minimum_articles_before_eligibility_stop": args.minimum_articles_before_eligibility_stop,
        "minimum_eligible_per_stage": args.minimum_eligible_per_stage,
        "minimum_auto_eligible_per_stage": args.minimum_auto_eligible_per_stage,
        "automate_eligible_counts": final_automate_counts,
        "auto_eligible_counts": final_auto_counts,
        "adaptive_eligibility_gate_passed": adaptive_gate_passed,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "num_ctx": args.num_ctx,
        "max_attempts": args.max_attempts,
        "reflection_interval": args.reflection_interval,
        "min_substantive_rationale_words": args.min_substantive_rationale_words,
        "allow_nonsubstantive_rationale": args.allow_nonsubstantive_rationale,
        "stage_candidates_sha256": sha256_file(args.output_dir / "stage_candidates.jsonl"),
        "recursive_rollouts_sha256": sha256_file(args.output_dir / "recursive_rollouts.csv"),
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    if not adaptive_gate_passed:
        raise RuntimeError(
            "Candidate reserve exhausted without satisfying the declared adaptive eligibility gate: "
            f"Automate={final_automate_counts}, Auto={final_auto_counts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
