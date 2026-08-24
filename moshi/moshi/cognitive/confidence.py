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

    def scaled(self, factor: float) -> "ConfidenceScore":
        """Apply an extra discount to ``confidence`` (not relevance/freshness).

        Used to mark a speculatively-retrieved reference as provisional (PHASES.md
        Phase 1.2): it might be trusted enough to nudge generation a little before
        the real trigger confirms it, but not as much as a confirmed retrieval.
        """
        return ConfidenceScore(relevance=self.relevance, confidence=self.confidence * factor, freshness=self.freshness)

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

    @classmethod
    def heuristic_from_llm_reference(
        cls, reference_text: str, elapsed_seconds: float, timeout_seconds: float
    ) -> "ConfidenceScore":
        """Rough score for a retrieval-LLM-generated reference, until a real relevance
        model exists (see PHASES.md Phase 1.1). Two cheap, available-today signals:

        - an empty/near-empty reference is almost certainly a non-answer or refusal
        - an answer that took nearly the full timeout budget was rushed under time
          pressure and is less trustworthy than one that came back quickly

        This has no access to the query itself, so ``relevance`` is left at 1.0 —
        a real implementation would compare embedding similarity between the query
        and the reference. ``freshness`` is likewise left at 1.0: generic
        OpenAI-compatible chat completions carry no source timestamp.
        """
        text = (reference_text or "").strip()
        if not text:
            return cls.empty()

        length_confidence = 1.0 if len(text) >= 10 else len(text) / 10
        if timeout_seconds > 0:
            time_pressure = _clamp01(elapsed_seconds / timeout_seconds)
        else:
            time_pressure = 0.0
        rushed_penalty = 1.0 - 0.3 * time_pressure

        return cls(relevance=1.0, confidence=length_confidence * rushed_penalty, freshness=1.0)
