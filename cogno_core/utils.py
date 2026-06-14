import re
import math
import string
from typing import Iterable

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates cosine similarity between two vector lists in pure Python."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if magnitude_v1 == 0.0 or magnitude_v2 == 0.0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)


def expand_slangs(text: str, slangs: dict[str, str]) -> str:
    """
    Normalizes slang/abbreviations based on an explicit key-value dictionary.
    Preserves surrounding punctuation of each word.
    """
    if not text or not slangs:
        return text

    words = text.split()
    normalized_words = []

    for word in words:
        # Separate punctuation to inspect the clean word
        clean_word = word.strip(string.punctuation).lower()
        if clean_word in slangs:
            # Reconstruct word preserving original prefix/suffix punctuation
            idx = word.lower().find(clean_word)
            if idx != -1:
                prefix = word[:idx]
                suffix = word[idx + len(clean_word):]
                normalized_words.append(f"{prefix}{slangs[clean_word]}{suffix}")
            else:
                normalized_words.append(slangs[clean_word])
        else:
            normalized_words.append(word)

    return " ".join(normalized_words)

STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "without", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "this", "that", "these", "those",
    "what", "how", "why", "when", "where", "which", "who", "whom",
    "me", "my", "mine", "you", "your", "yours", "i", "we", "our",
    "it", "its", "as", "by", "from", "at", "about", "into", "than",
    "then", "so", "if", "not", "can", "could", "should", "would",
    "please",
}


def clamp01(value: float) -> float:
    """Clamps a numeric value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def safe_float(value: object, default: float = 0.0) -> float:
    """Converts a value to float safely, falling back to default on error."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def word_count(text: str) -> int:
    """Counts lexical tokens in a dependency-free way."""
    if not text:
        return 0
    normalized = str(text).replace("_", " ").replace("-", " ").lower()
    return len(re.findall(r"[a-z0-9]+", normalized))


def content_words(text: str) -> set[str]:
    """Extracts normalized content words from text, filtering out stop words and short words."""
    if not text:
        return set()
    normalized = str(text).replace("_", " ").replace("-", " ").lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    return {
        word
        for word in words
        if len(word) > 3 and word not in STOPWORDS
    }


def extend_strings(target: list[str], values: Iterable[object] | object) -> None:
    """Safely appends non-empty string values to a target list."""
    if values is None:
        return

    if isinstance(values, str):
        value = values.strip()
        if value:
            target.append(value)
        return

    try:
        iterator = iter(values)  # type: ignore[call-overload]
    except TypeError:
        value = str(values).strip()
        if value:
            target.append(value)
        return

    for item in iterator:
        if item is None:
            continue
        value = str(item).strip()
        if value:
            target.append(value)

