# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Real acoustic feature extraction from raw PCM (PHASES.md Phase 5.1).

``cognitive.interruption.InterruptionClassifier`` originally worked from VAD
history (a single coarse "is there voice" signal) and partial ASR words only
— explicitly flagged in Phase 4 as needing "real prosody" to go further. This
module extracts genuine signal from the raw audio itself: RMS energy and a
per-frame F0 (pitch) estimate via autocorrelation, tracked over a short
rolling window to expose trends (energy rising/falling, pitch rising/falling).

This is a classical DSP technique, not a trained model — no torch, no GPU,
runs on plain numpy (which, unlike torch, is actually installed in this
environment, so this is genuinely tested against synthetic audio, not just
written blind). It is adequate for trend detection (is the user's voice
trailing off vs. staying assertive) but not for precise F0 tracking or
anything approaching real emotion/prosody modeling — that's a much larger
investment this phase doesn't attempt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def rms_energy(pcm: np.ndarray) -> float:
    """Root-mean-square energy of a PCM frame, in the same units as the input."""
    if pcm.size == 0:
        return 0.0
    x = pcm.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def estimate_pitch(
    pcm: np.ndarray,
    sample_rate: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    voicing_threshold: float = 0.3,
) -> float | None:
    """Autocorrelation-based F0 estimate for one frame of audio.

    Returns ``None`` when the frame isn't periodic enough to be confidently
    voiced (silence, noise, unvoiced consonants like "s" or "f") rather than
    guessing a number that isn't meaningful. ``fmin``/``fmax`` bound the
    search to the human speech F0 range; ``voicing_threshold`` is the minimum
    normalized autocorrelation peak (in ``[0, 1]``) required to trust the
    estimate at all.
    """
    if pcm.size < 2:
        return None
    x = pcm.astype(np.float64)
    x = x - x.mean()

    energy = np.sqrt(np.mean(x**2))
    if energy < 1e-6:
        return None

    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    if corr[0] <= 0:
        return None

    min_lag = int(sample_rate / fmax)
    max_lag = min(int(sample_rate / fmin), corr.size - 1)
    if min_lag >= max_lag:
        return None

    segment = corr[min_lag : max_lag + 1]
    if segment.size == 0:
        return None

    peak_offset = int(np.argmax(segment))
    peak_lag = min_lag + peak_offset
    if peak_lag <= 0:
        return None

    normalized_peak = segment[peak_offset] / corr[0]
    if normalized_peak < voicing_threshold:
        return None

    return float(sample_rate) / float(peak_lag)


@dataclass(frozen=True)
class ProsodyFeatures:
    pitch_hz: float | None
    energy: float
    voiced: bool


class ProsodyTracker:
    """Rolling window of ``ProsodyFeatures`` over recent audio frames, exposing trends.

    Fed one raw PCM frame at a time (matching how audio already arrives in
    ``channel.py``'s receive loop, frame by frame). Trends are computed as a
    simple first-half-vs-second-half average difference over the window --
    adequate to say "rising" or "falling", not precise enough for anything
    more quantitative.
    """

    def __init__(self, window: int = 10, silence_energy_threshold: float = 1e-3):
        self.window = window
        self.silence_energy_threshold = silence_energy_threshold
        self._history: deque[ProsodyFeatures] = deque(maxlen=window)

    def update(self, pcm: np.ndarray, sample_rate: int) -> ProsodyFeatures:
        energy = rms_energy(pcm)
        pitch = None
        if energy >= self.silence_energy_threshold:
            pitch = estimate_pitch(pcm, sample_rate)
        features = ProsodyFeatures(pitch_hz=pitch, energy=energy, voiced=pitch is not None)
        self._history.append(features)
        return features

    def _trend(self, values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        mid = len(values) // 2
        if mid == 0:
            return None
        first = sum(values[:mid]) / mid
        second = sum(values[mid:]) / (len(values) - mid)
        return second - first

    def energy_trend(self) -> float | None:
        """Positive => energy rising (assertive), negative => falling (trailing off)."""
        return self._trend([f.energy for f in self._history])

    def pitch_trend(self) -> float | None:
        """Positive => pitch rising (e.g. question intonation), negative => falling.
        ``None`` if fewer than two voiced frames are available in the window."""
        voiced_pitches = [f.pitch_hz for f in self._history if f.pitch_hz is not None]
        return self._trend(voiced_pitches)

    def is_trailing_off(self, threshold: float = -0.01) -> bool:
        trend = self.energy_trend()
        return trend is not None and trend < threshold

    def reset(self) -> None:
        self._history.clear()
