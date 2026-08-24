# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Confidence-aware conditioning strength (PHASES.md Phase 1.1).

Today a retrieval either succeeds (full-strength conditioning bias applied,
see ``channel.py::_async_update_reference``) or fails (no conditioning at
all). This module makes the strength of that bias a continuous function of
how much the knowledge candidate should be trusted, instead of a boolean.

The scoring itself is a placeholder — real relevance/freshness scoring
(embedding similarity to the query, source timestamps, retriever-reported
confidence) belongs to whatever cognitive service produces the candidate.
This module only defines the contract and the aggregation function, so RAG,
memory, and future tools can all express "how much should this move Moshi"
in the same units.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class ConfidenceScore:
    """Per-candidate quality signal for one piece of retrieved/recalled knowledge.

    All fields are in ``[0, 1]``; out-of-range inputs are clamped rather than
    raising, since callers (LLM self-reports, heuristics) can't be trusted to
    respect the range.
    """

    relevance: float = 1.0
    """How well the candidate matches the query/turn that triggered retrieval."""

    confidence: float = 1.0
    """How much the producing service trusts the candidate itself (e.g. the
    retrieval LLM's own certainty, or 1.0 for a verbatim memory record)."""

    freshness: float = 1.0
    """1.0 = current; decays for stale facts/old memories. Callers with no
    notion of freshness (e.g. a calculator) should leave this at 1.0."""

    def __post_init__(self):
        object.__setattr__(self, "relevance", _clamp01(self.relevance))
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
        object.__setattr__(self, "freshness", _clamp01(self.freshness))

    def strength(self) -> float:
        """Conditioning-bias multiplier in ``[0, 1]``.

        Geometric mean rather than arithmetic: a candidate that is highly
        relevant but zero-confidence (or vice versa) should contribute ~0,
        not average out to a comfortable middle value that still biases
        generation on a candidate nobody trusts.
        """
        product = self.relevance * self.confidence * self.freshness
        return product ** (1 / 3) if product > 0 else 0.0

    @classmethod
    def empty(cls) -> "ConfidenceScore":
        """No usable candidate — conditioning strength is exactly 0."""
        return cls(relevance=0.0, confidence=0.0, freshness=0.0)
