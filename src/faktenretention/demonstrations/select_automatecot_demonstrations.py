#!/usr/bin/env python3
"""Select ordered stage-specific demonstrations with adapted Automate-CoT VR-PGE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from faktenretention.text_rules import count_words
from faktenretention.demonstrations.rules import (
    sha256_text, cache_key, stage_input_valid, MODEL_DIGEST, SEED,
    TEMPERATURE, TOP_P, REPEAT_PENALTY, NUM_PREDICT,
)

import torch
from rouge_score import rouge_scorer

from faktenretention.text_rules import seed_raw
from faktenretention.text_rules import bounds_for_target, extract_text
from faktenretention.model_transport import ollama_generate


MODEL = "thesis-seed:v1"
TARGETS = (500, 300, 150, 75)
TOLERANCE_PCT = 5.0
CHECKPOINT_PROTOCOL = "v13_signed_structural_numeric_diagnostic_atomic_scheduler"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def torch_save_atomic(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)




def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric_tokens(text: str) -> set[str]:
    values: set[str] = set()
    for token in re.findall(r"(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?%?", text or ""):
        value = re.sub(r"[$£€,%\s]", "", token.lower())
        if value:
            values.add(value)
    return values


def complete_sentence(text: str) -> bool:
    return bool(re.search(r'[.!?]["\'\)\]”’»›]*\s*$', (text or "").strip()))




def demonstration_block(demo: dict[str, Any], position: int, target: int) -> str:
    return (
        f"DEMONSTRATION {position} — compress to approximately {target} words\n"
        f"INPUT:\n{str(demo['input_text']).strip()}\n\n"
        f"STEP-BY-STEP REASONING:\n{str(demo['thinking']).strip()}\n\n"
        f"FINAL_SUMMARY:\n{str(demo['response']).strip()}\n"
    )


def build_fewshot_prompt(input_text: str, target: int, demos: list[dict[str, Any]]) -> str:
    low, high = bounds_for_target(target, TOLERANCE_PCT)
    number_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(
        len(demos), str(len(demos))
    )
    blocks = "\n\n".join(
        demonstration_block(demo, position, target)
        for position, demo in enumerate(demos, 1)
    )
    return (
        f"Task: Learn the compression behavior from the {number_word} ordered demonstrations, then summarize "
        f"the new input in approximately {target} words. Think step by step before the final answer.\n"
        f"The accepted length interval is {low}-{high} words. Preserve the most important facts "
        "supported by the new input, do not invent facts, and end with a complete sentence.\n"
        "Output plain prose only after the exact label FINAL_SUMMARY:. Do not put text after it.\n\n"
        f"{blocks}\n\n"
        f"NEW INPUT:\n{input_text.strip()}"
    )


def generate(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    last_error: Exception | None = None
    for retry in range(args.api_retries):
        try:
            return ollama_generate(
                model=MODEL,
                prompt=seed_raw(prompt, True, reflection_interval=512),
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
        except Exception as exc:
            last_error = exc
            if retry + 1 < args.api_retries:
                time.sleep(min(30, 2**retry))
    assert last_error is not None
    raise last_error


def sample_ordered_unique(probs: torch.Tensor, generator: torch.Generator) -> list[int]:
    chosen: list[int] = []
    available = torch.ones(probs.shape[1], dtype=torch.bool)
    for position in range(probs.shape[0]):
        masked = probs[position].detach().clone()
        masked[~available] = 0
        masked /= masked.sum()
        index = int(torch.multinomial(masked, 1, generator=generator).item())
        chosen.append(index)
        available[index] = False
    return chosen


def ordered_log_probability(probs: torch.Tensor, indices: list[int]) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    available = torch.ones(probs.shape[1], dtype=torch.bool)
    for position, index in enumerate(indices):
        masked = probs[position] * available.to(probs.dtype)
        normalized = masked / masked.sum()
        terms.append(torch.log(normalized[index].clamp_min(1e-12)))
        available[index] = False
    return torch.stack(terms).sum()


def greedy_ordered_unique(probs: torch.Tensor) -> list[int]:
    chosen: list[int] = []
    available = torch.ones(probs.shape[1], dtype=torch.bool)
    for position in range(probs.shape[0]):
        scores = probs[position].detach().clone()
        scores[~available] = -1
        index = int(torch.argmax(scores).item())
        chosen.append(index)
        available[index] = False
    return chosen




def structural_output_valid(validity: dict[str, Any]) -> bool:
    return bool(
        validity["nonempty"]
        and validity["word_window_success"]
        and validity["ends_with_complete_sentence"]
    )


def policy_signature(
    target: int,
    pool: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    payload = {
        "checkpoint_protocol": CHECKPOINT_PROTOCOL,
        "target": target,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "generation": {
            "seed": SEED, "num_ctx": args.num_ctx, "temperature": TEMPERATURE,
            "top_p": TOP_P, "repeat_penalty": REPEAT_PENALTY,
            "num_predict": NUM_PREDICT, "reflection_interval": 512,
        },
        "optimization": {
            "sample_size": args.sample_size, "pge_avg_samples": args.pge_avg_samples,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "gradient_clip": args.gradient_clip,
            "probability_floor": args.probability_floor, "policy_seed": args.policy_seed,
        },
        "pool": [
            {
                "candidate_id": row["candidate_id"], "input_text": row["input_text"],
                "thinking": row["thinking"], "response": row["response"],
            }
            for row in pool
        ],
        "validation": [
            {
                "candidate_id": row["candidate_id"], "input_text": row["input_text"],
                "highlights": row["highlights"],
            }
            for row in validation
        ],
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def evaluate_one(
    target: int,
    demos: list[dict[str, Any]],
    validation: dict[str, Any],
    cache_dir: Path,
    rouge: rouge_scorer.RougeScorer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    key = cache_key(target, demos, validation, args.num_ctx)
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prompt = build_fewshot_prompt(str(validation["input_text"]), target, demos)
    started = time.monotonic()
    raw = generate(prompt, args)
    response = extract_text(raw).strip()
    words = count_words(response)
    low, high = bounds_for_target(target, TOLERANCE_PCT)
    novel_numbers = sorted(numeric_tokens(response) - numeric_tokens(str(validation["input_text"])))
    validity = {
        "nonempty": bool(response),
        "word_window_success": low <= words <= high,
        "ends_with_complete_sentence": complete_sentence(response),
        "novel_numeric_tokens": novel_numbers,
    }
    hard_valid = structural_output_valid(validity)
    rouge_l = float(
        rouge.score(str(validation["highlights"]), response)["rougeL"].fmeasure
    ) if response else 0.0
    utility = rouge_l if hard_valid else 0.0
    result = {
        "cache_key": key,
        "created_at_utc": now_utc(),
        "target_words": target,
        "ordered_demo_ids": [demo["candidate_id"] for demo in demos],
        "validation_candidate_id": validation["candidate_id"],
        "prompt_sha256": sha256_text(prompt),
        "prompt_word_count": count_words(prompt),
        "response": response,
        "response_word_count": words,
        "validity": validity,
        "hard_valid": hard_valid,
        "highlight_rougeL_f1": rouge_l,
        "utility": utility,
        "loss": 1.0 - utility,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "done_reason": raw.get("done_reason"),
        "usage": {
            name: raw.get(name)
            for name in (
                "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration",
            )
        },
    }
    write_json_atomic(path, result)
    return result


def mean_loss_for_set(
    target: int,
    indices: list[int],
    pool: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    cache_dir: Path,
    rouge: rouge_scorer.RougeScorer,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, Any]]]:
    demos = [pool[index] for index in indices]
    outcomes = [evaluate_one(target, demos, row, cache_dir, rouge, args) for row in batch]
    return sum(float(row["loss"]) for row in outcomes) / len(outcomes), outcomes


def project_probabilities(probs: torch.Tensor, floor: float) -> None:
    with torch.no_grad():
        probs.clamp_(min=floor)
        probs.div_(probs.sum(dim=1, keepdim=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", required=True, type=Path)
    parser.add_argument("--validation-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1:11435")
    parser.add_argument("--num-ctx", type=int, default=524288)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--pge-avg-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=3.0)
    parser.add_argument("--probability-floor", type=float, default=1e-4)
    parser.add_argument("--policy-seed", type=int, default=32)
    parser.add_argument("--minimum-validation-size", type=int, default=20)
    parser.add_argument("--targets", type=int, nargs="+", default=list(TARGETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "generation_cache"
    cache_dir.mkdir(exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    pools = json.loads(args.pools.read_text(encoding="utf-8"))
    validation_all = load_jsonl(args.validation_candidates)
    from faktenretention.io import validate_candidate_keys
    for pool in pools.values():
        validate_candidate_keys(pool)
    validate_candidate_keys(validation_all)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    final_selections: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}

    for target in args.targets:
        pool = pools[str(target)]
        validation_stage_all = [row for row in validation_all if int(row["target_words"]) == target]
        validation = [row for row in validation_stage_all if stage_input_valid(row, target)]
        if len(pool) < args.sample_size or len(validation) < args.minimum_validation_size:
            raise RuntimeError(f"Insufficient pool/validation rows for target={target}")
        signature = policy_signature(target, pool, validation, args)
        state_path = checkpoint_dir / f"stage_{target}.pt"
        probs = torch.nn.Parameter(
            torch.full((args.sample_size, len(pool)), 1.0 / len(pool), dtype=torch.float64)
        )
        optimizer = torch.optim.Adam([probs], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        history: list[dict[str, Any]] = []
        start_epoch = 0
        start_batch = 0
        generator = torch.Generator().manual_seed(args.policy_seed + target)
        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            if state.get("checkpoint_protocol") != CHECKPOINT_PROTOCOL:
                raise RuntimeError(
                    f"Incompatible policy checkpoint at {state_path}: "
                    f"{state.get('checkpoint_protocol')!r}"
                )
            if state.get("policy_signature") != signature:
                raise RuntimeError(f"Checkpoint data/hyperparameter signature mismatch at {state_path}")
            probs.data.copy_(state["probabilities"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            generator.set_state(state["generator_state"])
            history = state["history"]
            start_epoch = int(state["next_epoch"])
            start_batch = int(state["next_batch"])

        batches = [validation[i:i + args.batch_size] for i in range(0, len(validation), args.batch_size)]
        for epoch in range(start_epoch, args.epochs):
            first_batch = start_batch if epoch == start_epoch else 0
            for batch_index in range(first_batch, len(batches)):
                sampled = [
                    sample_ordered_unique(probs, generator)
                    for _ in range(args.pge_avg_samples)
                ]
                losses: list[float] = []
                valid_rates: list[float] = []
                for sample_index, indices in enumerate(sampled):
                    loss, outcomes = mean_loss_for_set(
                        target, indices, pool, batches[batch_index], cache_dir, rouge, args
                    )
                    losses.append(loss)
                    valid_rates.append(sum(bool(row["hard_valid"]) for row in outcomes) / len(outcomes))
                    print(
                        f"target={target} epoch={epoch + 1}/{args.epochs} "
                        f"batch={batch_index + 1}/{len(batches)} sample={sample_index + 1}/"
                        f"{args.pge_avg_samples} loss={loss:.6f}", flush=True,
                    )
                baseline = sum(losses) / len(losses)
                log_probs = torch.stack([ordered_log_probability(probs, indices) for indices in sampled])
                advantages = torch.tensor(losses, dtype=probs.dtype) - baseline
                objective = torch.sum(advantages.detach() * log_probs) / (len(losses) - 1)
                optimizer.zero_grad()
                objective.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_([probs], args.gradient_clip))
                optimizer.step()
                project_probabilities(probs, args.probability_floor)
                record = {
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "mean_loss": baseline,
                    "mean_hard_valid_rate": sum(valid_rates) / len(valid_rates),
                    "gradient_norm_before_clip": gradient_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "sampled_candidate_ids": [
                        [pool[index]["candidate_id"] for index in indices] for indices in sampled
                    ],
                    "sample_losses": losses,
                }
                history.append(record)
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch >= len(batches):
                    next_epoch, next_batch = epoch + 1, 0
                    scheduler.step()
                torch_save_atomic(
                    state_path,
                    {
                        "checkpoint_protocol": CHECKPOINT_PROTOCOL,
                        "policy_signature": signature,
                        "probabilities": probs.detach().clone(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "generator_state": generator.get_state(),
                        "history": history,
                        "next_epoch": next_epoch,
                        "next_batch": next_batch,
                    },
                )

        chosen_indices = greedy_ordered_unique(probs)
        chosen = [dict(pool[index]) for index in chosen_indices]
        for position, row in enumerate(chosen, 1):
            row["automate_cot_position"] = position
            row["selection_probability"] = float(probs[position - 1, chosen_indices[position - 1]])
        final_loss, final_outcomes = mean_loss_for_set(
            target, chosen_indices, pool, validation, cache_dir, rouge, args
        )
        final_selections[str(target)] = chosen
        reports[str(target)] = {
            "pool_size": len(pool),
            "validation_size": len(validation),
            "validation_stage_rows_excluded_for_input_length": len(validation_stage_all) - len(validation),
            "ordered_candidate_ids": [row["candidate_id"] for row in chosen],
            "validation_mean_loss": final_loss,
            "validation_mean_utility": 1.0 - final_loss,
            "validation_hard_valid_rate": sum(bool(row["hard_valid"]) for row in final_outcomes) / len(final_outcomes),
            "training_updates": len(history),
            "history": history,
            "probabilities": probs.detach().tolist(),
            "policy_signature": signature,
        }
        write_json_atomic(args.output_dir / "automate_cot_demonstrations.partial.json", final_selections)
        write_json_atomic(args.output_dir / "automate_cot_selection_report.partial.json", reports)

    write_json_atomic(args.output_dir / "automate_cot_demonstrations.json", final_selections)
    write_json_atomic(
        args.output_dir / "automate_cot_selection_report.json",
        {
            "created_at_utc": now_utc(),
            "method": "summarization-adapted Automate-CoT ordered unique demonstration selection",
            "model": MODEL,
            "model_digest": MODEL_DIGEST,
            "num_ctx": args.num_ctx,
            "reflection_interval": 512,
            "policy": {
                "sample_size": args.sample_size,
                "pge_avg_samples": args.pge_avg_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "gradient_clip": args.gradient_clip,
                "probability_floor": args.probability_floor,
                "seed": args.policy_seed,
                "sampling": "position-specific categorical distributions, sequentially masked to enforce unique ordered demonstrations",
                "estimator": "leave-mean-out variance-reduced REINFORCE",
            },
            "objective": "loss = 1 - structurally hard-valid stemmed highlight ROUGE-L F1; hard-invalid output utility = 0; numeric novelty is diagnostic only",
            "glm_qa_used": False,
            "stages": reports,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
