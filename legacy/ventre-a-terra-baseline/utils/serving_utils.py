"""Serving phase utilities."""

from __future__ import annotations

import re

# Common intolerance patterns in orderText (multilingual).
_INTOLERANCE_PATTERNS = [
    # English
    re.compile(
        r"(?:\bintoleran(?:t|ce)\s+to\b|\ballerg(?:ic|y)\s+to\b|\bcannot eat\b|\bcan'?t eat\b)\s+([^.!?;]+)",
        re.I,
    ),
    # Italian
    re.compile(
        r"(?:\bintolleran(?:te|za)\s+a(?:l|ll[aeo]|gli|i)?\b|\ballergic[oa]\s+a(?:l|ll[aeo]|gli|i)?\b|\bnon posso mangiare\b)\s+([^.!?;]+)",
        re.I,
    ),
]


def _normalize_phrase(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zà-öø-ÿ\s]+", " ", text.lower())
    return " ".join(cleaned.split())


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _normalize_phrase(text).split() if len(tok) >= 3}


def extract_intolerances(order_text: str) -> set[str]:
    """Extract ingredient names the client is intolerant to from free-text order."""
    result: set[str] = set()
    for pat in _INTOLERANCE_PATTERNS:
        for m in pat.finditer(order_text):
            raw = m.group(1).strip().rstrip(".,!;")
            # Split on commas / "and" / "e" for multiple intolerances
            parts = re.split(r",\s*|\s+and\s+|\s+e\s+|/\s*", raw)
            for p in parts:
                cleaned = _normalize_phrase(p)
                if cleaned:
                    result.add(cleaned)
    return result


def dish_has_intolerance(recipe: dict, intolerances: set[str]) -> bool:
    """Check if a recipe contains any ingredient the client is intolerant to."""
    if not intolerances:
        return False
    intol_norm = {_normalize_phrase(i) for i in intolerances if _normalize_phrase(i)}
    intol_tokens = {i: _tokenize(i) for i in intol_norm}

    for ing_name in recipe.get("ingredients", {}):
        ing_norm = _normalize_phrase(ing_name)
        ing_tokens = _tokenize(ing_name)
        for intol in intol_norm:
            if ing_norm == intol:
                return True
            if " " in intol and intol in ing_norm:
                return True
            toks = intol_tokens.get(intol, set())
            if toks and toks.issubset(ing_tokens):
                return True
    return False


def rank_dishes_for_serving(cookable: list[dict]) -> list[dict]:
    """Sort cookable dishes for throughput-first serving.

    Primary key: preparationTimeMs ascending (faster service).
    Secondary key: prestige descending (keep quality when speed is equal).
    """
    return sorted(
        cookable,
        key=lambda x: (
            int(x.get("preparationTimeMs", 999999)),
            -int(x.get("prestige", 0)),
        ),
    )
