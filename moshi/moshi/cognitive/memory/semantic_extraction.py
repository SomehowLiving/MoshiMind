# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Rule-based semantic fact extraction (PHASES.md Phase 3.3).

PHASES.md describes semantic memory as facts distilled from episodic memory
"on a slower cadence (batch job, not per-turn)" — the intended real
implementation is an LLM pass over accumulated episodes, off the critical
path, the same pattern already used for RAG's reference generation. That
needs an LLM endpoint this environment doesn't have configured for testing.

This is a deliberately simple regex-pattern placeholder standing in for that
pass, so the storage/upsert/retrieval machinery has a concrete, free,
fully-offline producer to exercise. It WILL misfire on adversarial or unusual
phrasing (e.g. "I am not sure" extracting a bogus ``self_description`` fact) —
labeled as INFERENCE-quality, not production fact extraction.
"""

from __future__ import annotations

import re

# (pattern, fact key). The captured group is intentionally greedy up to a small
# word cap ({1,4} words) rather than a character cap -- a character cap doesn't
# stop at clause boundaries ("my name is Nidhi and I work as an engineer" would
# otherwise capture "Nidhi and I work as an engineer" as the name).
_WORD = r"[A-Za-z][\w'-]*"
_VALUE = rf"({_WORD}(?:\s+{_WORD}){{0,3}})"

_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(rf"\bmy name is {_VALUE}", re.IGNORECASE), "name"),
    (re.compile(rf"\bi work as (?:an? )?{_VALUE}", re.IGNORECASE), "occupation"),
    (re.compile(rf"\bi live in {_VALUE}", re.IGNORECASE), "location"),
    (re.compile(rf"\bi love {_VALUE}", re.IGNORECASE), "preference"),
    (re.compile(rf"\bi like {_VALUE}", re.IGNORECASE), "preference"),
)

# Cut a captured value short at the first conjunction/subordinate clause marker,
# since the word-count cap alone still lets through e.g. "Nidhi and I" (3 words,
# "and" and "I" both match \w+) for "my name is Nidhi and I work as...".
_CLAUSE_BREAK_RE = re.compile(r"\b(and|but|so|because|which|who|that)\b.*$", re.IGNORECASE)
_STRIP_TRAILING = ".,!?;: "


def _clean_value(raw: str) -> str:
    value = _CLAUSE_BREAK_RE.sub("", raw)
    for sep in (",", ".", ";"):
        idx = value.find(sep)
        if idx != -1:
            value = value[:idx]
    return value.strip().rstrip(_STRIP_TRAILING).strip()


def extract_facts(user_text: str) -> list[tuple[str, str]]:
    """Return ``(key, value)`` pairs found via simple regex patterns.

    Multiple facts can be extracted from one utterance (e.g. name and
    occupation in the same sentence). Values are trimmed at the first clause
    break and any trailing punctuation; empty matches are dropped.
    """
    text = (user_text or "").strip()
    if not text:
        return []
    facts: list[tuple[str, str]] = []
    for pattern, key in _PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_value(match.group(1))
            if value:
                facts.append((key, value))
    return facts
