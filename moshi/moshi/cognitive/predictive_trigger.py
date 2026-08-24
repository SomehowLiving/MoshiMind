# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Heuristic detector for "this utterance is probably heading toward a knowledge
need" (PHASES.md Phase 1.2), to start retrieval before Moshi's own
``rag_token_id`` emission rather than waiting for it.

This is explicitly a placeholder heuristic, not a real intent classifier — a
question word or a handful of trigger phrases in the streaming ASR transcript
so far. A better version would use embedding similarity or a small trained
router; this exists so the racing/reuse machinery (``cognitive.speculation``)
has a concrete, cheap producer to test against before that investment is
made. See PHASES.md 1.2.
"""

from __future__ import annotations

import string

_QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}
_INFO_PHRASES = (
    "tell me about",
    "what about",
    "how much",
    "how many",
    "how old",
    "what's the",
    "whats the",
    "can you look up",
    "do you know",
)


class PredictiveTrigger:
    """Fires at most once per turn: call ``reset()`` when a new user turn starts.

    Debouncing lives here rather than in the caller because the heuristic
    itself is what defines "the same question" for now (no semantic
    comparison between successive partial transcripts) — a caller resetting
    at the wrong granularity would otherwise cause repeated speculative
    retrievals for one growing utterance.
    """

    def __init__(self, min_words: int = 3):
        self.min_words = min_words
        self._fired = False

    def reset(self) -> None:
        self._fired = False

    def should_fire(self, partial_text: str) -> bool:
        """Whether ``partial_text`` (the accumulated partial transcript for the
        current user turn) looks like it's heading toward a knowledge need.
        """
        if self._fired:
            return False
        text = (partial_text or "").strip().lower()
        if not text:
            return False
        words = text.split()
        if len(words) < self.min_words:
            return False

        first_word = words[0].strip(string.punctuation)
        looks_like_question = first_word in _QUESTION_WORDS or text.endswith("?")
        looks_like_info_request = any(phrase in text for phrase in _INFO_PHRASES)

        if looks_like_question or looks_like_info_request:
            self._fired = True
            return True
        return False
