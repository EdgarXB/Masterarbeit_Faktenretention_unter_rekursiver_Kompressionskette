#!/usr/bin/env python3
"""Build stage-specific Few-Shot-CoT prompts from the shared eligible CoT pool.

The shared pool is deliberately conservative: a candidate must be retained in
the real Automate-CoT pruning pool and must also pass the Auto-CoT quality gate.
Five unique candidates are then sampled uniformly without replacement for each
compression stage with a recorded deterministic seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "experiments/autocot_automatecot_100"
AUTOMATE_POOLS = PROJECT / "automate_cot_pruning/automate_cot_pruned_pools.json"
AUTOMATE_REPORT = PROJECT / "automate_cot_pruning/automate_cot_pruning_report.json"
AUTO_REPORT = PROJECT / "auto_cot_selection/auto_cot_selection_report.json"
OUT = ROOT / "experiments/few_shot_cot_shared_pool_20260930"
TARGETS = (500, 300, 150, 75)
INPUT_TARGETS = {500: 800, 300: 500, 150: 300, 75: 150}
WINDOWS = {500: (475, 525), 300: (285, 315), 150: (142, 158), 75: (71, 79)}
SEED = 32
SAMPLE_SIZE = 5
PLACEHOLDER = "{{NEW_INPUT}}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def expected_input_bounds(target: int) -> tuple[int, int]:
    expected = INPUT_TARGETS[target]
    if target == 500:
        return 795, 805
    return round(expected * 0.95), round(expected * 1.05)


def common_eligibility_reasons(row: dict[str, Any], target: int) -> list[str]:
    """Return failures from the union of Auto-CoT and Automate-CoT hard gates."""
    reasons: list[str] = []
    if int(row.get("target_words") or 0) != target:
        reasons.append("wrong_target")
    if not str(row.get("input_text") or "").strip():
        reasons.append("empty_input")
    if not str(row.get("thinking") or "").strip():
        reasons.append("empty_rationale")
    substantive_words = int(row.get("substantive_thinking_word_count") or 0)
    if substantive_words < 25:
        reasons.append("rationale_has_fewer_than_25_substantive_words")
    thinking_words = int(row.get("thinking_word_count") or count_words(str(row.get("thinking") or "")))
    if thinking_words > 512:
        reasons.append("rationale_exceeds_512_word_auto_cot_limit")
    if not str(row.get("response") or "").strip():
        reasons.append("empty_response")
    if not bool(row.get("word_window_success")):
        reasons.append("outside_word_window")
    if not bool(row.get("ends_with_complete_sentence")):
        reasons.append("incomplete_final_sentence")
    low, high = expected_input_bounds(target)
    input_words = int(row.get("input_word_count") or count_words(str(row.get("input_text") or "")))
    if not low <= input_words <= high:
        reasons.append(f"input_outside_expected_stage_window:{input_words}_not_{low}-{high}")
    if row.get("selected_into_pool") is not True:
        reasons.append("not_retained_in_automate_cot_pool")
    if list(row.get("hard_filter_reasons") or []):
        reasons.append("automate_cot_hard_filter_reasons_not_empty")
    return reasons


def demonstration_block(position: int, target: int, row: dict[str, Any]) -> str:
    return (
        f"DEMONSTRATION {position} — compress to approximately {target} words\n"
        f"INPUT:\n{str(row['input_text']).strip()}\n\n"
        f"STEP-BY-STEP REASONING:\n{str(row['thinking']).strip()}\n\n"
        f"FINAL_SUMMARY:\n{str(row['response']).strip()}"
    )


def build_prompt(target: int, rows: list[dict[str, Any]]) -> str:
    low, high = WINDOWS[target]
    blocks = "\n\n".join(
        demonstration_block(position, target, row)
        for position, row in enumerate(rows, 1)
    )
    return (
        f"Task: Learn the compression behavior from the five demonstrations, then summarize "
        f"the new input in approximately {target} words. Think step by step before the final answer.\n"
        f"The accepted length interval is {low}-{high} words. Preserve the most important facts "
        "supported by the new input, do not invent facts, and end with a complete sentence.\n"
        "Output the reasoning after the exact label STEP-BY-STEP REASONING:. Then output plain "
        "summary prose after the exact label FINAL_SUMMARY:. Do not put text after the summary.\n\n"
        f"{blocks}\n\n"
        f"NEW INPUT:\n{PLACEHOLDER}\n\n"
        "STEP-BY-STEP REASONING:\n"
    )


def source_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT)),
        "sha256": sha256_bytes(path.read_bytes()),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    global PROJECT, AUTOMATE_POOLS, AUTOMATE_REPORT, AUTO_REPORT, OUT
    PROJECT = args.project
    AUTOMATE_POOLS = PROJECT / "automate_cot_pruning/automate_cot_pruned_pools.json"
    AUTOMATE_REPORT = PROJECT / "automate_cot_pruning/automate_cot_pruning_report.json"
    AUTO_REPORT = PROJECT / "auto_cot_selection/auto_cot_selection_report.json"
    OUT = args.output_dir
    pools = json.loads(AUTOMATE_POOLS.read_text(encoding="utf-8"))
    # Require the original selection reports to be present so the generated
    # package remains tied to the audited Auto-CoT and Automate-CoT procedures.
    json.loads(AUTO_REPORT.read_text(encoding="utf-8"))
    json.loads(AUTOMATE_REPORT.read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "method": "few_shot_cot",
        "construction": "uniform random sampling without replacement from the common Auto-CoT/Automate-CoT eligible pool",
        "common_pool_definition": (
            "Candidates retained in automate_cot_pruned_pools.json that also pass the Auto-CoT "
            "quality gate, including its 512-word rationale upper bound."
        ),
        "selection_seed": SEED,
        "rng": "Python random.Random; one seeded stream; stages processed in order 500, 300, 150, 75",
        "sample_size_per_stage": SAMPLE_SIZE,
        "source_artifacts": {
            "automate_cot_retained_pools": source_receipt(AUTOMATE_POOLS),
            "automate_cot_selection_report": source_receipt(AUTOMATE_REPORT),
            "auto_cot_selection_report": source_receipt(AUTO_REPORT),
        },
        "eligibility_rules": [
            "candidate retained in the 100-item Automate-CoT stage pool",
            "non-empty stage input, rationale, and final summary",
            "at least 25 substantive rationale words",
            "at most 512 rationale words (Auto-CoT upper bound)",
            "stage input in the expected source-length window",
            "final summary in the preregistered +/-5% target window",
            "final summary ends with a complete sentence",
        ],
        "word_count_rule_for_runtime_targets": "regex \\b\\w+\\b, matching the final experiment",
        "stages": {},
        "prompts": {},
    }

    all_selected_ids: list[str] = []
    for target in TARGETS:
        stage_pool = pools[str(target)]
        eligible: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for row in stage_pool:
            reasons = common_eligibility_reasons(row, target)
            if reasons:
                excluded.append({"candidate_id": row.get("candidate_id"), "reasons": reasons})
            else:
                eligible.append(row)
        eligible.sort(key=lambda row: str(row["candidate_id"]))
        if len(eligible) < SAMPLE_SIZE:
            raise RuntimeError(f"target={target}: only {len(eligible)} commonly eligible candidates")
        selected = rng.sample(eligible, SAMPLE_SIZE)
        selected_ids = [str(row["candidate_id"]) for row in selected]
        if len(set(selected_ids)) != SAMPLE_SIZE:
            raise RuntimeError(f"target={target}: duplicate sampled candidate")
        all_selected_ids.extend(selected_ids)

        prompt = build_prompt(target, selected)
        if prompt.count(PLACEHOLDER) != 1 or prompt.count("DEMONSTRATION ") != SAMPLE_SIZE:
            raise RuntimeError(f"target={target}: malformed prompt")
        prompt_path = OUT / f"few_shot_cot_{target}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        manifest["stages"][str(target)] = {
            "automate_retained_pool_size": len(stage_pool),
            "common_eligible_pool_size": len(eligible),
            "excluded_by_auto_cot_gate": excluded,
            "ordered_selected_candidate_ids": selected_ids,
            "selected_demonstrations": [
                {
                    "position": position,
                    "candidate_id": row["candidate_id"],
                    "dataset_index": row["dataset_index"],
                    "article_id": row["article_id"],
                    "input_word_count": row["input_word_count"],
                    "thinking_word_count": row["thinking_word_count"],
                    "substantive_thinking_word_count": row["substantive_thinking_word_count"],
                    "response_word_count": row["response_word_count"],
                    "summac_rank_within_stage": row["summac_rank_within_stage"],
                    "input_sha256": sha256_text(str(row["input_text"])),
                    "thinking_sha256": sha256_text(str(row["thinking"])),
                    "response_sha256": sha256_text(str(row["response"])),
                }
                for position, row in enumerate(selected, 1)
            ],
        }
        manifest["prompts"][str(target)] = {
            "path": prompt_path.name,
            "sha256": sha256_bytes(prompt_path.read_bytes()),
            "demonstration_count": SAMPLE_SIZE,
        }

    if len(all_selected_ids) != len(set(all_selected_ids)):
        # Candidate IDs include the stage, so this protects against accidental
        # duplicate stage candidates while still allowing one article to occur
        # at different compression stages if independently sampled.
        raise RuntimeError("duplicate stage candidate across prompts")

    manifest_path = OUT / "few_shot_cot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(TARGETS)} prompts, {SAMPLE_SIZE} demonstrations per stage")
    for target in TARGETS:
        stage = manifest["stages"][str(target)]
        print(
            f"{target}: pool={stage['common_eligible_pool_size']} "
            f"selected={','.join(stage['ordered_selected_candidate_ids'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
