# Exact function/constant source slice from score_model_selection_batch.py.
import re
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
}

FRACTION_DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
    "eleventh": 11,
    "elevenths": 11,
    "twelfth": 12,
    "twelfths": 12,
}


def fraction_token(value: Fraction) -> str:
    return f"frac{value.numerator}over{value.denominator}"


def parse_number_tokens(tokens: List[str]) -> Optional[int]:
    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token == "and":
            continue
        if token.isdigit():
            current += int(token)
            saw_number = True
        elif token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            saw_number = True
        elif token == "hundred":
            current = (current or 1) * NUMBER_SCALES[token]
            saw_number = True
        elif token in NUMBER_SCALES:
            total += (current or 1) * NUMBER_SCALES[token]
            current = 0
            saw_number = True
        else:
            return None
    if not saw_number:
        return None
    return total + current


def parse_longest_number(tokens: List[str], start: int, max_len: int = 8) -> Tuple[Optional[int], int]:
    best_value: Optional[int] = None
    best_end = start
    for end in range(start + 1, min(len(tokens), start + max_len) + 1):
        value = parse_number_tokens(tokens[start:end])
        if value is not None:
            best_value = value
            best_end = end
    return best_value, best_end


def normalize_symbolic_quantities(text: str) -> str:
    def replace_fraction(match: re.Match[str]) -> str:
        denominator = int(match.group(2))
        if denominator == 0:
            return match.group(0)
        return fraction_token(Fraction(int(match.group(1)), denominator))

    def replace_percent(match: re.Match[str]) -> str:
        return fraction_token(Fraction(int(match.group(1)), 100))

    text = re.sub(r"\b([0-9]+)\s*/\s*([0-9]+)\b", replace_fraction, text)
    text = re.sub(r"\b([0-9]+)\s*%", replace_percent, text)
    return text


def quantity_token(value: int | str, comparator: str = "") -> str:
    comparator_phrases = {
        "cmpgt": "more than",
        "cmplt": "less than",
        "cmpge": "at least",
        "cmple": "at most",
        "cmpapprox": "about",
    }
    if comparator:
        return f"{comparator_phrases[comparator]} {value}"
    return str(value)


def normalize_textual_quantities(text: str) -> str:
    tokens = text.split()
    out: List[str] = []
    i = 0
    comparators = {
        ("more", "than"): "cmpgt",
        ("greater", "than"): "cmpgt",
        ("over",): "cmpgt",
        ("above",): "cmpgt",
        ("less", "than"): "cmplt",
        ("under",): "cmplt",
        ("fewer", "than"): "cmplt",
        ("at", "least"): "cmpge",
        ("at", "most"): "cmple",
        ("about",): "cmpapprox",
        ("approximately",): "cmpapprox",
    }

    while i < len(tokens):
        matched_comparator = False
        for phrase, comparator in sorted(comparators.items(), key=lambda item: len(item[0]), reverse=True):
            if tuple(tokens[i : i + len(phrase)]) != phrase:
                continue
            value, end = parse_longest_number(tokens, i + len(phrase))
            if value is None:
                continue
            if end < len(tokens) and tokens[end] in {"decade", "decades"}:
                out.extend([quantity_token(value * 10, comparator), "years"])
                i = end + 1
            elif end < len(tokens) and tokens[end] in {"percent", "percentage"}:
                out.append(quantity_token(fraction_token(Fraction(value, 100)), comparator))
                i = end + 1
            else:
                out.append(quantity_token(value, comparator))
                i = end
            matched_comparator = True
            break
        if matched_comparator:
            continue

        if tokens[i] in {"a", "an"} and i + 1 < len(tokens) and tokens[i + 1] in FRACTION_DENOMINATORS:
            out.append(fraction_token(Fraction(1, FRACTION_DENOMINATORS[tokens[i + 1]])))
            i += 2
            continue

        value, end = parse_longest_number(tokens, i)
        if value is not None:
            if end < len(tokens) and tokens[end] in FRACTION_DENOMINATORS:
                out.append(fraction_token(Fraction(value, FRACTION_DENOMINATORS[tokens[end]])))
                i = end + 1
                continue
            if end < len(tokens) and tokens[end] in {"percent", "percentage"}:
                out.append(fraction_token(Fraction(value, 100)))
                i = end + 1
                continue
            if end < len(tokens) and tokens[end] in {"decade", "decades"}:
                out.extend([str(value * 10), "years"])
                i = end + 1
                continue
            out.append(str(value))
            i = end
            continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def canonicalize_for_retention(text: str) -> str:
    """Normalize safe paraphrases that the lexical QA scorer should accept.

    This intentionally stays conservative. It handles common wording and
    numeric equivalences observed in human validation, but it does not expand
    vague categories into named entities; e.g. "federal aid" is not treated as
    "Federal Emergency Management Agency".
    """
    canonical = normalize(normalize_symbolic_quantities(text.lower()))
    replacements = {
        "u s": "united states",
        "u s a": "united states",
        "usa": "united states",
        "suspended": "suspend",
        "suspension": "suspend",
    }
    for source, target in replacements.items():
        canonical = re.sub(rf"\b{re.escape(source)}\b", target, canonical)

    canonical = re.sub(r"\bmore then ([0-9]+)\b", r"more than \1", canonical)
    canonical = normalize_textual_quantities(canonical)
    canonical = re.sub(r"\bas an?\b", "as", canonical)
    canonical = re.sub(r"\bthe united states\b", "united states", canonical)
    canonical = re.sub(r"^(?:the|a|an) ", "", canonical)
    return re.sub(r"\s+", " ", canonical).strip()


def token_f1(reference: str, candidate: str) -> float:
    ref_tokens = normalize(reference).split()
    cand_tokens = normalize(candidate).split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts: Dict[str, int] = defaultdict(int)
    cand_counts: Dict[str, int] = defaultdict(int)
    for token in ref_tokens:
        ref_counts[token] += 1
    for token in cand_tokens:
        cand_counts[token] += 1
    overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts)
    if not overlap:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def retained(reference: str, answer: str) -> bool:
    if answer.upper() == "NOT_FOUND":
        return False
    norm_ref = normalize(reference)
    norm_answer = normalize(answer)
    if norm_ref and (norm_ref in norm_answer or norm_answer in norm_ref):
        return True
    if token_f1(reference, answer) >= 0.50:
        return True

    canon_ref = canonicalize_for_retention(reference)
    canon_answer = canonicalize_for_retention(answer)
    if canon_ref and (canon_ref in canon_answer or canon_answer in canon_ref):
        return True

    ref_len = len(canon_ref.split())
    answer_len = len(canon_answer.split())
    length_ratio = (answer_len / ref_len) if ref_len else 99.0
    return bool(
        0 < ref_len <= 10
        and length_ratio <= 4.0
        and token_f1(canon_ref, canon_answer) >= 0.43
    )


def retention_token_f1(reference: str, answer: str) -> float:
    return max(
        token_f1(reference, answer),
        token_f1(canonicalize_for_retention(reference), canonicalize_for_retention(answer)),
    )



def support_is_grounded(summary: str, answer: str, support: str) -> bool:
    """Require machine-checkable summary evidence for every non-NOT_FOUND answer."""
    normalized_summary = " ".join((summary or "").split()).casefold()
    normalized_support = " ".join((support or "").split()).casefold()
    if (answer or "").strip().upper() == "NOT_FOUND":
        return not normalized_support
    return bool(normalized_support and normalized_support in normalized_summary)

