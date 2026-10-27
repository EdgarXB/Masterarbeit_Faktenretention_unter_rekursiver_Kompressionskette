# Originale reine Text-/Promptfunktionen; keine Modellaufrufe beim Import.
import re
from typing import Any,Dict,List,Optional,Tuple


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def bounds_for_target(target: int, tolerance_pct: float) -> Tuple[int, int]:
    margin = max(1, round(target * (tolerance_pct / 100.0)))
    return target - margin, target + margin

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

def seed_raw(
    prompt: str, cot: bool, reflection_interval: int = 128, thinking_budget: int = 512,
) -> str:
    if cot:
        if thinking_budget <= 0 or reflection_interval <= 0:
            raise ValueError("thinking_budget and reflection_interval must be positive")
        system = (
            "You are an intelligent assistant with reflective ability. In the process of thinking and "
            f"reasoning, you need to strictly follow the thinking budget, which is {thinking_budget}. "
            f"That is, you need to complete your thinking within {thinking_budget} tokens and start "
            "answering the user's questions. You "
            f"will reflect on your thinking process every {reflection_interval} tokens, stating how many "
            "tokens have been used and how many are left."
        )
        suffix = "<seed:bos>assistant\n"
    else:
        system = (
            "You are an intelligent assistant that can answer questions in one step without the need for "
            "reasoning and thinking, that is, your thinking budget is 0. Next, please skip the thinking "
            "process and directly start answering the user's questions."
        )
        suffix = (
            "<seed:bos>assistant\n<seed:think>\n<seed:cot_budget_reflect>The current thinking "
            "budget is 0, so I will directly start answering the question.</seed:cot_budget_reflect>\n"
            "</seed:think>"
        )
    return f"<seed:bos>system\n{system}<seed:eos><seed:bos>user\n{prompt}<seed:eos>{suffix}"

def complete_sentence(text: str) -> bool:
    return bool(re.search(r'[.!?]["\'\)\]”’»›]*\s*$', (text or "").strip()))

def lead_k(text: str, target: int) -> str:
    matches = list(re.finditer(r"\b\w+\b", text or ""))
    if len(matches) <= target:
        return (text or "").strip()
    return text[: matches[target - 1].end()].strip()

def direct_prompt(text: str, target: int, cot: bool) -> str:
    low, high = bounds_for_target(target, 5.0)
    reasoning = " Think step by step before the final answer." if cot else ""
    return (
        f"Task: Summarize the new input in approximately {target} words.{reasoning}\n"
        f"The accepted length interval is {low}-{high} words. Preserve the most important facts "
        "supported by the new input, do not invent facts, and end with a complete sentence.\n"
        "Output plain prose only after the exact label FINAL_SUMMARY:. Do not put text after it.\n\n"
        f"NEW INPUT:\n{text.strip()}"
    )
