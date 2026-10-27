"""Unchanged cache keys and eligibility without Torch imports."""
import hashlib
import json
from typing import Any
from faktenretention.text_rules import count_words

MODEL_DIGEST = '7a66a2f466bf48fdafa7004a7975a7f5fac6e667a6de7d01751aacb98b3f387c'
SEED, TEMPERATURE, TOP_P, REPEAT_PENALTY, NUM_PREDICT = 32, 1.1, 0.95, 1.0, 2400

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def stage_input_valid(row: dict[str, Any], target: int) -> bool:
    expected = {500: 800, 300: 500, 150: 300, 75: 150}[target]
    low, high = (795, 805) if target == 500 else (round(expected * 0.95), round(expected * 1.05))
    return low <= int(row.get("input_word_count") or count_words(str(row.get("input_text") or ""))) <= high

def cache_key(
    target: int,
    demos: list[dict[str, Any]],
    validation: dict[str, Any],
    num_ctx: int,
) -> str:
    payload = {
        "target": target,
        "ordered_demo_ids": [demo["candidate_id"] for demo in demos],
        "ordered_demo_content_sha256": [
            sha256_text(json.dumps({
                "input_text": demo["input_text"],
                "thinking": demo["thinking"],
                "response": demo["response"],
            }, sort_keys=True))
            for demo in demos
        ],
        "validation_candidate_id": validation["candidate_id"],
        "input_sha256": sha256_text(str(validation["input_text"])),
        "highlights_sha256": sha256_text(str(validation["highlights"])),
        "model_digest": MODEL_DIGEST,
        "seed": SEED,
        "num_ctx": num_ctx,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "num_predict": NUM_PREDICT,
        "reflection_interval": 512,
        "scoring_protocol": "v10_structural_validity_numeric_diagnostic_only",
    }
    return sha256_text(json.dumps(payload, sort_keys=True))
