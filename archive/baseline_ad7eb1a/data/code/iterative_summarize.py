#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple


TARGETS = [500, 300, 150, 75]
DEFAULT_TOLERANCE_PCT = 5.0
NEMOTRON_REASONING_BUDGET = 512


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def word_units(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text)


def bounds_for_target(target: int, tolerance_pct: float) -> Tuple[int, int]:
    margin = max(1, round(target * (tolerance_pct / 100.0)))
    return target - margin, target + margin


def trim_to_words(text: str, max_words: int) -> str:
    words = word_units(text)
    return " ".join(words[:max_words])


def ends_with_complete_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(re.search(r'[.!?]["\'\)\]”’»›]*\s*$', stripped))


def trim_to_last_sentence(text: str) -> str:
    stripped = (text or "").strip()
    matches = list(re.finditer(r'[.!?]["\')\]]*(?=\s|$)', stripped))
    if not matches:
        return stripped
    return stripped[: matches[-1].end()].strip()


def pad_with_source_words(summary: str, source_text: str, min_words: int) -> str:
    summary_words = word_units(summary)
    if len(summary_words) >= min_words:
        return summary

    source_words = word_units(source_text)
    needed = min_words - len(summary_words)
    summary_words.extend(source_words[:needed])
    return " ".join(summary_words)


def prediction_budget_for_target(
    target: int,
    tolerance_pct: float,
    model: str = "",
    think: Optional[Any] = None,
) -> int:
    _, max_words = bounds_for_target(target, tolerance_pct)
    base = max(256, min(1600, max_words * 3))
    if think is True:
        return min(4096, base + 1600)
    if isinstance(think, str):
        if model.startswith("gpt-oss:"):
            return 16384
        return 8192
    return base


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
    from request_budget import guarded_generate
    return guarded_generate(payload, host, timeout)


def split_inline_thinking(text: str) -> Tuple[str, str]:
    stripped = (text or "").strip()
    # Some imported Qwen3.5/Qwen3.8 GGUF chat templates prefill the opening
    # <think> token. Ollama then returns only the body, closing tag, and final
    # answer in `response` instead of populating its dedicated `thinking` key.
    # Split that representation before looking for project-specific markers.
    qwen_think = re.match(r"(?:<think>\s*)?(.*?)\s*</think>\s*(.*)", stripped, flags=re.DOTALL)
    if qwen_think and qwen_think.group(2).strip():
        return qwen_think.group(1).strip(), qwen_think.group(2).strip()
    final_marker = re.search(r"FINAL_SUMMARY\s*:\s*(.*)", stripped, flags=re.DOTALL | re.IGNORECASE)
    if final_marker and final_marker.group(1).strip():
        return stripped[: final_marker.start()].strip(), final_marker.group(1).strip()
    markdown_summary_patterns = [
        r"\*\*Step-by-step process:\*\*\s*(.*?)\s*\*\*Summary:\*\*\s*(.*)",
        r"Step-by-step process:\s*(.*?)\s*Summary:\s*(.*)",
    ]
    for pattern in markdown_summary_patterns:
        match = re.search(pattern, stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            thinking = match.group(1).strip()
            response = match.group(2).strip()
            return thinking, response
    patterns = [
        r"<\|channel\|>thought\s*(.*?)<\|channel\|>\s*(.*)",
        r"<\|channel\|>thought\s*(.*?)<channel\|>\s*(.*)",
        r"<channel\|>thought\s*(.*?)<\|channel\|>\s*(.*)",
        r"<channel\|>thought\s*(.*?)<channel\|>\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            response = match.group(2).strip()
            return thinking, response
    lines = stripped.splitlines()
    outline_lines: List[str] = []
    split_index: Optional[int] = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
            outline_lines.append(line.strip())
            continue
        if not outline_lines:
            break
        if not line.strip():
            remainder = "\n".join(lines[index + 1 :]).strip()
            if remainder:
                split_index = index + 1
            break
        outline_lines = []
        break
    if outline_lines and split_index is not None:
        thinking = "\n".join(outline_lines).strip()
        response = "\n".join(lines[split_index:]).strip()
        if response:
            return thinking, response
    return "", stripped


def extract_text(response_data: Dict[str, Any]) -> str:
    _, cleaned = split_inline_thinking(response_data.get("response") or "")
    return cleaned


def extract_thinking(response_data: Dict[str, Any]) -> str:
    explicit = (response_data.get("thinking") or "").strip()
    if explicit:
        inline, _ = split_inline_thinking(response_data.get("response") or "")
        if inline and inline not in explicit:
            return f"{explicit}\n\n{inline}".strip()
        return explicit
    inline, _ = split_inline_thinking(response_data.get("response") or "")
    return inline


def is_empty_or_missing_text_response(summary: str, source_text: str) -> bool:
    stripped = (summary or "").strip()
    if not stripped:
        return True
    if source_text.strip() and re.search(r"\bno text (?:was )?(?:supplied|provided)\b", stripped, flags=re.IGNORECASE):
        return True
    return False


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


def summarize_once(
    text: str,
    target: int,
    model: str,
    host: str,
    tolerance_pct: float,
    instruction_template: str = "summarize in {target_word_length}",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    seed: int = 32,
    num_ctx: int = 8192,
    keep_alive: Optional[str] = None,
    system_prompt: Optional[str] = None,
    think: Optional[Any] = None,
    num_predict: Optional[int] = None,
    trace_collector: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, int]:
    original_prompt = build_prompt(
        text,
        target,
        tolerance_pct,
        instruction_template=instruction_template,
    )

    def run_generation(think_value: Optional[Any], phase: str, num_predict_override: Optional[int] = None) -> str:
        prompt = apply_model_prompt_controls(model, original_prompt, think_value)
        requested_num_predict = (
            num_predict_override
            if num_predict_override is not None
            else (
                num_predict
                if num_predict is not None
                else prediction_budget_for_target(target, tolerance_pct, model=model, think=think_value)
            )
        )
        if model.startswith(("nemotron3:", "nemotron-3-nano:")) and think_value not in (None, False):
            reasoning_data = ollama_generate(
                model=model,
                prompt=prompt,
                host=host,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=True,
                num_predict=NEMOTRON_REASONING_BUDGET,
            )
            reasoning = extract_thinking(reasoning_data)
            if not reasoning.strip():
                raise ValueError("Nemotron returned no reasoning content for the budgeted CoT pass.")
            rendered_prefix = (
                "\n\n\n<|im_start|>system\n"
                + (system_prompt or "")
                + "<|im_end|>\n\n<|im_start|>user\n"
                + prompt
                + "<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
                + reasoning
                + "\n</think>\n"
            )
            response_data = ollama_generate(
                model=model,
                prompt=rendered_prefix,
                host=host,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=False,
                num_predict=prediction_budget_for_target(target, tolerance_pct, model=model, think=False),
                raw=True,
            )
            response_data["thinking"] = reasoning
            for timing_key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
                response_data[timing_key] = int(response_data.get(timing_key) or 0) + int(
                    reasoning_data.get(timing_key) or 0
                )
            for count_key in ("prompt_eval_count", "eval_count"):
                response_data[count_key] = int(response_data.get(count_key) or 0) + int(
                    reasoning_data.get(count_key) or 0
                )
        else:
            response_data = ollama_generate(
                model=model,
                prompt=prompt,
                host=host,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=think_value,
                num_predict=requested_num_predict,
            )
        candidate = extract_text(response_data)
        if trace_collector is not None:
            trace_collector(
                {
                    "phase": phase,
                    "target_words": target,
                    "prompt_input": text,
                    "system_prompt": system_prompt or "",
                    "instruction_template": instruction_template,
                    "response": candidate,
                    "thinking": extract_thinking(response_data),
                    "usage": {
                        "total_duration": response_data.get("total_duration"),
                        "load_duration": response_data.get("load_duration"),
                        "prompt_eval_count": response_data.get("prompt_eval_count"),
                        "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                        "eval_count": response_data.get("eval_count"),
                        "eval_duration": response_data.get("eval_duration"),
                    },
                }
            )
        return candidate

    candidate = run_generation(think, "summarize_to_target")
    keep_gpt_oss_cot_trace = model.startswith("gpt-oss:") and isinstance(think, str) and trace_collector is not None
    if is_empty_or_missing_text_response(candidate, text) and think not in (None, False) and not keep_gpt_oss_cot_trace:
        retry_budget = 16384 if model.startswith("gpt-oss:") else 8192
        candidate = run_generation(False, "summarize_to_target_retry_no_think", num_predict_override=retry_budget)
    if is_empty_or_missing_text_response(candidate, text):
        raise ValueError(f"Model returned an empty or invalid summary for target {target}.")
    return candidate, count_words(candidate)


def finalize_summary(
    summary: str,
    source_text: str,
    target: int,
    model: str,
    host: str,
    tolerance_pct: float,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    seed: int = 32,
    num_ctx: int = 8192,
    keep_alive: Optional[str] = None,
    system_prompt: Optional[str] = None,
    think: Optional[Any] = None,
    trace_collector: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, int]:
    current = (summary or "").strip()
    wc = count_words(current)
    min_words, max_words = bounds_for_target(target, tolerance_pct)

    if not current or ends_with_complete_sentence(current):
        return current, wc

    prompt = (
        "Rewrite the summary so it ends with a complete final sentence while preserving the same key facts.\n"
        "The summary to rewrite is provided below; do not ask for missing text.\n"
        f"Keep it close to {target} words and preferably within {min_words}-{max_words} words.\n"
        "If the ending is cut off, rewrite it so the summary finishes cleanly.\n"
        "Put the rewritten summary after the exact label FINAL_SUMMARY:.\n"
        "Do not put any text after the final summary.\n\n"
        f"SUMMARY:\n{current}"
    )
    prompt = apply_model_prompt_controls(model, prompt, think)
    response_data = ollama_generate(
        model=model,
        prompt=prompt,
        host=host,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        think=think,
        num_predict=prediction_budget_for_target(target, tolerance_pct, model=model, think=think),
    )
    rewritten = extract_text(response_data)
    if trace_collector is not None:
        trace_collector(
            {
                "phase": "finalize_summary",
                "target_words": target,
                "prompt_input": prompt,
                "system_prompt": system_prompt or "",
                "response": rewritten,
                "thinking": extract_thinking(response_data),
                "usage": {
                    "total_duration": response_data.get("total_duration"),
                    "load_duration": response_data.get("load_duration"),
                    "prompt_eval_count": response_data.get("prompt_eval_count"),
                    "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                    "eval_count": response_data.get("eval_count"),
                    "eval_duration": response_data.get("eval_duration"),
                },
            }
        )
    if rewritten.strip():
        return rewritten.strip(), count_words(rewritten)
    return current, wc


def summarize_to_target(
    text: str,
    target: int,
    model: str,
    host: str,
    tolerance_pct: float,
    instruction_template: str = "summarize in {target_word_length}",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    seed: int = 32,
    num_ctx: int = 8192,
    keep_alive: Optional[str] = None,
    system_prompt: Optional[str] = None,
    think: Optional[Any] = None,
    trace_collector: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_attempts: int = 4,
) -> Tuple[str, int]:
    best_summary = ""
    best_wc = 0
    best_distance = 10**9
    working_text = text
    min_words, max_words = bounds_for_target(target, tolerance_pct)

    for _ in range(max_attempts):
        prompt = build_prompt(
            working_text,
            target,
            tolerance_pct,
            instruction_template=instruction_template,
        )
        prompt = apply_model_prompt_controls(model, prompt, think)
        response_data = ollama_generate(
            model=model,
            prompt=prompt,
            host=host,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            think=think,
            num_predict=prediction_budget_for_target(target, tolerance_pct, model=model, think=think),
        )
        candidate = extract_text(response_data)
        if trace_collector is not None:
            trace_collector(
                {
                    "phase": "summarize_to_target",
                    "target_words": target,
                    "prompt_input": working_text,
                    "system_prompt": system_prompt or "",
                    "instruction_template": instruction_template,
                    "response": candidate,
                    "thinking": extract_thinking(response_data),
                    "usage": {
                        "total_duration": response_data.get("total_duration"),
                        "load_duration": response_data.get("load_duration"),
                        "prompt_eval_count": response_data.get("prompt_eval_count"),
                        "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                        "eval_count": response_data.get("eval_count"),
                        "eval_duration": response_data.get("eval_duration"),
                    },
                }
            )
        wc = count_words(candidate)
        distance = abs(wc - target)

        if distance < best_distance and candidate:
            best_summary = candidate
            best_wc = wc
            best_distance = distance

        if min_words <= wc <= max_words:
            return finalize_summary(
                summary=candidate,
                source_text=text,
                target=target,
                model=model,
                host=host,
                tolerance_pct=tolerance_pct,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                think=think,
                trace_collector=trace_collector,
            )

        # Ask again based on the last candidate to steer toward target.
        working_text = (
            f"Your previous summary had {wc} words. Rewrite it to be between "
            f"{min_words} and {max_words} words, targeting {target} words.\n\n{candidate}"
        )

    return finalize_summary(
        summary=best_summary,
        source_text=text,
        target=target,
        model=model,
        host=host,
        tolerance_pct=tolerance_pct,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        system_prompt=system_prompt,
        think=think,
        trace_collector=trace_collector,
    )


def enforce_window(
    summary: str,
    source_text: str,
    target: int,
    model: str,
    host: str,
    tolerance_pct: float,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    seed: int = 32,
    num_ctx: int = 8192,
    keep_alive: Optional[str] = None,
    system_prompt: Optional[str] = None,
    think: Optional[Any] = None,
    trace_collector: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_attempts: int = 4,
) -> Tuple[str, int]:
    min_words, max_words = bounds_for_target(target, tolerance_pct)
    wc = count_words(summary)

    if min_words <= wc <= max_words:
        return finalize_summary(
            summary=summary,
            source_text=source_text,
            target=target,
            model=model,
            host=host,
            tolerance_pct=tolerance_pct,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            system_prompt=system_prompt,
            think=think,
            trace_collector=trace_collector,
        )

    current = summary
    for _ in range(max_attempts):
        prompt = (
            "Rewrite the summary with the same key facts.\n"
            "The summary to rewrite is provided below; do not ask for missing text.\n"
            f"It must be between {min_words} and {max_words} words "
            f"(target {target}).\n"
            "Put the rewritten summary after the exact label FINAL_SUMMARY:.\n"
            "Do not put any text after the final summary.\n\n"
            f"SUMMARY:\n{current}"
        )
        prompt = apply_model_prompt_controls(model, prompt, think)
        response_data = ollama_generate(
            model=model,
            prompt=prompt,
            host=host,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            think=think,
            num_predict=prediction_budget_for_target(target, tolerance_pct, model=model, think=think),
        )
        current = extract_text(response_data)
        if trace_collector is not None:
            trace_collector(
                {
                    "phase": "enforce_window",
                    "target_words": target,
                    "prompt_input": prompt,
                    "system_prompt": system_prompt or "",
                    "response": current,
                    "thinking": extract_thinking(response_data),
                    "usage": {
                        "total_duration": response_data.get("total_duration"),
                        "load_duration": response_data.get("load_duration"),
                        "prompt_eval_count": response_data.get("prompt_eval_count"),
                        "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                        "eval_count": response_data.get("eval_count"),
                        "eval_duration": response_data.get("eval_duration"),
                    },
                }
            )
        wc = count_words(current)
        if min_words <= wc <= max_words:
            return finalize_summary(
                summary=current,
                source_text=source_text,
                target=target,
                model=model,
                host=host,
                tolerance_pct=tolerance_pct,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                think=think,
                trace_collector=trace_collector,
            )

    # Deterministic fallback for too-long summaries.
    if wc > max_words:
        trimmed = trim_to_words(current, max_words)
        trimmed_wc = count_words(trimmed)
        if min_words <= trimmed_wc <= max_words:
            return finalize_summary(
                summary=trimmed,
                source_text=source_text,
                target=target,
                model=model,
                host=host,
                tolerance_pct=tolerance_pct,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                think=think,
                trace_collector=trace_collector,
            )

    # Try one strict expansion pass if too short.
    if wc < min_words:
        for _ in range(3):
            prompt = (
                f"Rewrite this summary to EXACTLY {min_words} words while preserving key facts.\n"
                "The summary to rewrite is provided below; do not ask for missing text.\n"
                "Put the rewritten summary after the exact label FINAL_SUMMARY:.\n"
                "Do not put any text after the final summary.\n\n"
                f"SUMMARY:\n{current}"
            )
            prompt = apply_model_prompt_controls(model, prompt, think)
            response_data = ollama_generate(
                model=model,
                prompt=prompt,
                host=host,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=think,
                num_predict=prediction_budget_for_target(target, tolerance_pct, model=model, think=think),
            )
            current = extract_text(response_data)
            if trace_collector is not None:
                trace_collector(
                    {
                        "phase": "strict_expansion",
                        "target_words": target,
                        "prompt_input": prompt,
                        "system_prompt": system_prompt or "",
                        "response": current,
                        "thinking": extract_thinking(response_data),
                        "usage": {
                            "total_duration": response_data.get("total_duration"),
                            "load_duration": response_data.get("load_duration"),
                            "prompt_eval_count": response_data.get("prompt_eval_count"),
                            "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                            "eval_count": response_data.get("eval_count"),
                            "eval_duration": response_data.get("eval_duration"),
                        },
                    }
                )
            wc = count_words(current)
            if wc >= min_words:
                break

        if wc > max_words:
            current = trim_to_words(current, max_words)
            wc = count_words(current)

        if min_words <= wc <= max_words:
            return finalize_summary(
                summary=current,
                source_text=source_text,
                target=target,
                model=model,
                host=host,
                tolerance_pct=tolerance_pct,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                think=think,
                trace_collector=trace_collector,
            )

    # Final deterministic fallback for short outputs: append source words.
    if wc < min_words:
        padded = pad_with_source_words(current, source_text, min_words)
        padded_wc = count_words(padded)
        if padded_wc > max_words:
            padded = trim_to_words(padded, max_words)
            padded_wc = count_words(padded)
        if min_words <= padded_wc <= max_words:
            return finalize_summary(
                summary=padded,
                source_text=source_text,
                target=target,
                model=model,
                host=host,
                tolerance_pct=tolerance_pct,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                think=think,
                trace_collector=trace_collector,
            )

    raise RuntimeError(
        f"Could not produce summary in required window [{min_words}, {max_words}] for target {target}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively summarize selected articles with local Ollama model."
    )
    parser.add_argument("--input", default="selected_articles.csv")
    parser.add_argument("--output", default="iterative_summaries.csv")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "qwen3:30b"))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "127.0.0.1:11435"))
    parser.add_argument("--tolerance-pct", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int)
    parser.add_argument("--min-p", "--min_p", dest="min_p", type=float)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num-ctx", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input CSV not found: {args.input}")
        return 1

    rows_out: List[Dict[str, str]] = []

    try:
        with open(args.input, "r", encoding="utf-8", newline="") as f:
            rows_in = list(csv.DictReader(f))
    except Exception as exc:
        print(f"Failed to read input CSV: {exc}")
        return 1

    if len(rows_in) != 5:
        print(f"Warning: expected 5 rows, found {len(rows_in)} rows.")

    for i, row in enumerate(rows_in, start=1):
        article = (row.get("article") or "").strip()
        if not article:
            print(f"Row {i} has empty article, skipping.")
            continue

        print(f"Summarizing article {i}/{len(rows_in)}...")
        current_text = article

        output_row: Dict[str, str] = {
            "article_id": row.get("article_id", str(i)),
            "dataset_index": row.get("dataset_index", ""),
            "original_word_count": str(count_words(article)),
        }

        for target in TARGETS:
            try:
                summary, wc = summarize_to_target(
                    text=current_text,
                    target=target,
                    model=args.model,
                    host=args.host,
                    tolerance_pct=args.tolerance_pct,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    seed=args.seed,
                    num_ctx=args.num_ctx,
                )
                summary, wc = enforce_window(
                    summary=summary,
                    source_text=current_text,
                    target=target,
                    model=args.model,
                    host=args.host,
                    tolerance_pct=args.tolerance_pct,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    seed=args.seed,
                    num_ctx=args.num_ctx,
                )
            except urllib.error.URLError as exc:
                print(f"Cannot reach Ollama at http://{args.host}: {exc}")
                print("Start it with: ./run_test.sh")
                return 1
            except Exception as exc:
                print(f"Failed on target={target} for article {i}: {exc}")
                return 1

            output_row[f"summary_{target}"] = summary
            output_row[f"summary_{target}_word_count"] = str(wc)
            current_text = summary

        rows_out.append(output_row)

    fieldnames = [
        "article_id",
        "dataset_index",
        "original_word_count",
        "summary_500",
        "summary_500_word_count",
        "summary_300",
        "summary_300_word_count",
        "summary_150",
        "summary_150_word_count",
        "summary_75",
        "summary_75_word_count",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved iterative summaries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
