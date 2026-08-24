# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Urgency classification for cognitive requests (PHASES.md Phase 1.3/2.2).

Not every request deserves the same latency budget: "what's 27 times 43"
wants an immediate answer; "what did I tell you about my project last month"
can resolve whenever and simply update the conditioning stream when ready.
This gives every cognitive request a coarse deadline class instead of one
global ``rag_timeout`` (``server.py`` CLI flag / ``RAGManager.rag_timeout``)
applied uniformly regardless of what's being asked.
"""

from __future__ import annotations

import enum
import re


class Urgency(enum.Enum):
    """Latency budget class for a cognitive request."""

    CRITICAL = "critical"
    """May block generation briefly (short, hard deadline). Reserve for
    cheap, fast operations only (arithmetic, unit conversion) — never for a
    network call to an LLM or search backend."""

    NORMAL = "normal"
    """The existing RAG behavior: async, non-blocking, but with a fairly
    tight timeout since it's racing the model's own generation."""

    BACKGROUND = "background"
    """No deadline pressure at all — apply whenever ready, or not. Suited to
    memory consolidation, prefetching, and speculative lookups."""


# Heuristic only — a real classifier would use the retrieval-triggering LLM
# itself or a small trained router. This exists so Phase 2's scheduler has a
# concrete signal to test against before that investment is made.
_ARITHMETIC_RE = re.compile(r"^\s*[\d\s()+\-*/×÷.]+\s*$")


def classify_urgency(query: str, kind: str = "generic") -> Urgency:
    """Best-effort urgency classification for a cognitive request.

    Args:
        query: The natural-language request or extracted expression.
        kind: Hint from the caller ("rag", "memory", "tool:calculator", ...).
    """
    if kind.startswith("tool:calculator") or _ARITHMETIC_RE.match(query or ""):
        return Urgency.CRITICAL
    if kind == "memory" or kind.startswith("tool:background"):
        return Urgency.BACKGROUND
    return Urgency.NORMAL
