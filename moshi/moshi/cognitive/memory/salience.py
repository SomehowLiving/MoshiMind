# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Heuristic salience scoring for episodic memory (PHASES.md Phase 3.2).

PHASES.md's own description says episodic memory should retain turns "near a
topic change, an emotional peak, or an explicit 'remember this'" — the first
two need real signals this repo doesn't have yet (topic-change detection,
Phase 5's emotion work). This implements the third, plus a cheap length-based
proxy for "probably substantive," as a placeholder scorer: enough to prove the
storage/retention pipeline end to end, not a claim that this is how salience
should ultimately be judged.
"""

from __future__ import annotations

_REMEMBER_PHRASES = (
    "remember that",
    "remember this",
    "don't forget",
    "dont forget",
    "keep in mind",
    "note that",
    "for future reference",
)


def score_salience(text: str) -> float:
    """Return a salience score in ``[0, 1]`` for one turn of text.

    An explicit request to remember something scores highest (1.0); longer,
    more substantive-looking turns score moderately (0.6); short turns and
    bare questions (usually about the world, not about the user) score low.
    """
    t = (text or "").strip().lower()
    if not t:
        return 0.0
    if any(phrase in t for phrase in _REMEMBER_PHRASES):
        return 1.0
    word_count = len(t.split())
    if word_count >= 15:
        return 0.6
    if t.endswith("?"):
        return 0.2
    return 0.3
