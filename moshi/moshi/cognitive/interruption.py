# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Heuristic classifier for user speech arriving while the model is talking
(PHASES.md Phase 4.2).

Same placeholder posture as ``cognitive.predictive_trigger``: real barge-in
detection wants prosody (interruption tone vs. a trailing-off filler sound
distinctly differ acoustically, per Phase 5) and semantic understanding of
whether the partial words form a real objection/question. This uses only
what's already available today -- VAD history (``turn_manager.TurnManager``'s
convention: lower value = voice present, higher = silence) and partial ASR
words -- enough to prove the classification/decision machinery end to end.
"""

from __future__ import annotations

import string

from .conversation_state import InterruptionKind

_BACKCHANNEL_PHRASES = {
    "mhm",
    "mm-hmm",
    "mmhmm",
    "uh-huh",
    "uhhuh",
    "yeah",
    "yep",
    "yup",
    "right",
    "okay",
    "ok",
    "sure",
    "i see",
    "got it",
}

_HESITATION_MARKERS = {"um", "uh", "erm", "hmm", "well", "wait"}


class InterruptionClassifier:
    """Classifies overlapping user speech into backchannel / hesitation / barge-in.

    ``min_barge_in_words`` and ``sustained_voice_frames`` are both required before
    classifying something as a genuine barge-in, since a single word overlapping the
    model is exactly what a backchannel looks like too -- the distinguishing signal
    is that a real barge-in keeps going.
    """

    def __init__(self, min_barge_in_words: int = 4, sustained_voice_frames: int = 3):
        self.min_barge_in_words = min_barge_in_words
        self.sustained_voice_frames = sustained_voice_frames

    def classify(
        self,
        words_so_far: str,
        vad_history: list[float],
        vad_threshold: float = 0.5,
    ) -> InterruptionKind:
        text = (words_so_far or "").strip().lower()
        if not text:
            return InterruptionKind.NONE

        normalized = text.strip(string.punctuation + " ")
        word_count = len(normalized.split())

        if normalized in _BACKCHANNEL_PHRASES and word_count <= 2:
            return InterruptionKind.BACKCHANNEL

        sustained_voice = len(vad_history) >= self.sustained_voice_frames and all(
            v < vad_threshold for v in vad_history[-self.sustained_voice_frames :]
        )

        if sustained_voice and word_count >= self.min_barge_in_words:
            return InterruptionKind.BARGE_IN

        last_word = normalized.split()[-1] if normalized.split() else ""
        if last_word in _HESITATION_MARKERS:
            return InterruptionKind.HESITATION
        if word_count <= 2 and not sustained_voice:
            # Too short to be substantive and VAD hasn't confirmed sustained speech --
            # most likely a false start or trailing-off, not a real interruption.
            return InterruptionKind.HESITATION

        # Enough words to be substantive, but VAD hasn't (yet) confirmed sustained
        # voice -- e.g. history still filling up. Not enough signal either way.
        return InterruptionKind.NONE
