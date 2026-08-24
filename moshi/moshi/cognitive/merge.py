# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Arbitrating between multiple knowledge candidates for one conditioning
channel (PHASES.md Phase 3.4).

Moshi has exactly one ``streaming_sum`` conditioning slot per batch element
(``models/lm.py``) — RAG and memory both want to influence it, but there's
only one channel to write. This is the merge policy: rank candidates by
``ConfidenceScore.strength()`` and greedily concatenate the strongest ones
that fit a character budget, on the assumption that RAG (world facts) and
memory (user facts) are usually complementary rather than conflicting, so
combining them beats picking just one and discarding the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from .confidence import ConfidenceScore


@dataclass(frozen=True)
class KnowledgeCandidate:
    source: str
    text: str
    confidence: ConfidenceScore


def merge_candidates(candidates: list[KnowledgeCandidate], max_chars: int = 800) -> KnowledgeCandidate | None:
    """Combine usable candidates into one, or return ``None`` if none are usable.

    A candidate is usable if it has non-empty text and positive confidence
    strength. The merged result's confidence is the strongest candidate's own
    score — averaging (even geometrically) across sources would unfairly
    punish one strong hit for being paired with one weak scrap, when the
    weak one is simply appended rather than diluting the strong one's trust.
    """
    usable = [c for c in candidates if c.text and c.confidence.strength() > 0]
    if not usable:
        return None

    usable.sort(key=lambda c: c.confidence.strength(), reverse=True)
    if len(usable) == 1:
        return usable[0]

    parts: list[str] = []
    sources: list[str] = []
    total_chars = 0
    for candidate in usable:
        addition_len = len(candidate.text) + (1 if parts else 0)  # +1 for the joining space
        if total_chars + addition_len > max_chars:
            continue
        parts.append(candidate.text)
        sources.append(candidate.source)
        total_chars += addition_len

    return KnowledgeCandidate(source="+".join(sources), text=" ".join(parts), confidence=usable[0].confidence)
