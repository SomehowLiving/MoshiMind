# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Heuristic classifier for user speech arriving while the model is talking
(PHASES.md Phase 4.2, extended in Phase 5.1 with real acoustic signal).

Originally worked from VAD history + partial ASR words only. Now also takes an
optional energy trend from ``cognitive.prosody.ProsodyTracker`` (real
autocorrelation/RMS DSP over raw PCM, not a trained model) to override the
purely lexical/VAD-count heuristic when it disagrees with what the audio
itself is doing: a word count that looks like a committed barge-in but with
audibly declining energy is trailing off, not interrupting. Still a
placeholder relative to real prosody modeling (pitch-trend-based question
intonation, trained emotion classifiers) — this uses the one signal that's
both real and cheap enough to justify today.
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

    def __init__(
        self,
        min_barge_in_words: int = 4,
        sustained_voice_frames: int = 3,
        trailing_off_energy_threshold: float = -0.01,
    ):
        self.min_barge_in_words = min_barge_in_words
        self.sustained_voice_frames = sustained_voice_frames
        self.trailing_off_energy_threshold = trailing_off_energy_threshold

    def classify(
        self,
        words_so_far: str,
        vad_history: list[float],
        vad_threshold: float = 0.5,
        energy_trend: float | None = None,
    ) -> InterruptionKind:
        """``energy_trend``, if given (see ``ProsodyTracker.energy_trend``), is real
        acoustic signal from the raw audio: positive means rising/assertive energy,
        negative means declining/trailing-off energy. It only ever downgrades a
        lexical barge-in call to hesitation, never the reverse — word count and VAD
        can indicate someone is *trying* to interject, but declining energy is
        direct evidence they're trailing off rather than committing to it.
        """
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
            if energy_trend is not None and energy_trend < self.trailing_off_energy_threshold:
                return InterruptionKind.HESITATION
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
