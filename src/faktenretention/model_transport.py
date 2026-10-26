"""Original prompt controls and guarded transport; no historical pilot CLI."""
import json
import urllib.request
from typing import Any, Dict, Optional
from faktenretention.protocol import STAGES as TARGETS
from faktenretention.text_rules import count_words, bounds_for_target, extract_text, extract_thinking
def apply_model_prompt_controls(model: str, prompt: str, think: Optional[Any]) -> str:
    if model.startswith("qwen3:") and think is False:
        return "/no_think\n" + prompt
    return prompt

def ollama_generate(
    model: str,
    prompt: str,
    host: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    seed: int = 32,
    num_ctx: int = 8192,
    keep_alive: Optional[str] = None,
    think: Optional[Any] = None,
    num_predict: Optional[int] = None,
    response_format: Optional[Any] = None,
    raw: bool = False,
    timeout: int = 900,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
        },
    }
    if top_p is not None:
        payload["options"]["top_p"] = top_p
    if top_k is not None:
        payload["options"]["top_k"] = top_k
    if min_p is not None:
        payload["options"]["min_p"] = min_p
    if presence_penalty is not None:
        payload["options"]["presence_penalty"] = presence_penalty
    if repeat_penalty is not None:
        payload["options"]["repeat_penalty"] = repeat_penalty
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    if system_prompt:
        payload["system"] = system_prompt
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if think is not None:
        payload["think"] = think
    if response_format is not None:
        payload["format"] = response_format
    if raw:
        payload["raw"] = True
    req = urllib.request.Request(
        f"http://{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    from faktenretention.request_budget import guarded_generate
    return guarded_generate(payload, host, timeout)

def build_prompt(
    text: str,
    target: int,
    tolerance_pct: float,
    instruction_template: str = "summarize in {target_word_length}",
    aim_close_to_target: bool = True,
) -> str:
    min_words, max_words = bounds_for_target(target, tolerance_pct)
    length_guidance = (
        f"- Aim close to {target} words.\n" if aim_close_to_target else ""
    )
    return (
        f"Task: {instruction_template.format(target_word_length=target, min_words=min_words, max_words=max_words)}.\n"
        "Write a factual news summary of the TEXT below.\n"
        "Rules:\n"
        f"- Keep summary between {min_words} and {max_words} words.\n"
        f"{length_guidance}"
        "- Preserve key facts.\n"
        "- Plain prose only (no bullets, no title).\n"
        "- End with a complete sentence.\n"
        "- Do not ask for missing text; the TEXT is provided below.\n"
        "- Do not include reasoning, notes, planning, or a checklist in the answer.\n"
        "- Put the final summary after the exact label FINAL_SUMMARY:.\n"
        "- Do not put any text after the final summary.\n\n"
        "TEXT:\n"
        f"{text}"
    )
