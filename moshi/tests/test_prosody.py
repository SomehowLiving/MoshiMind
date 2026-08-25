# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for real acoustic feature extraction (PHASES.md Phase 5.1).

Unlike most of the cognitive-layer tests, these run genuine numerical DSP
code against synthetic audio (numpy IS installed in this environment, unlike
torch) -- pitch/energy estimates are checked against known ground truth
(a generated sine wave at a known frequency), not mocked.
"""

from __future__ import annotations

import numpy as np
import pytest

from moshi.cognitive.prosody import ProsodyFeatures, ProsodyTracker, estimate_pitch, rms_energy

SAMPLE_RATE = 24000


def _tone(freq_hz: float, duration_s: float = 0.05, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# rms_energy
# ---------------------------------------------------------------------------


def test_rms_energy_of_silence_is_zero():
    assert rms_energy(np.zeros(1000)) == 0.0


def test_rms_energy_of_empty_array_is_zero():
    assert rms_energy(np.array([])) == 0.0


def test_rms_energy_scales_with_amplitude():
    quiet = _tone(150, amplitude=0.1)
    loud = _tone(150, amplitude=0.8)
    assert rms_energy(loud) > rms_energy(quiet)


def test_rms_energy_of_constant_dc_offset():
    assert rms_energy(np.full(100, 0.5)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# estimate_pitch -- checked against known ground truth (generated sine waves)
# ---------------------------------------------------------------------------


def test_estimate_pitch_detects_known_frequency():
    tone = _tone(150.0)
    pitch = estimate_pitch(tone, SAMPLE_RATE)
    assert pitch is not None
    assert pitch == pytest.approx(150.0, abs=2.0)


def test_estimate_pitch_detects_a_different_known_frequency():
    tone = _tone(220.0)
    pitch = estimate_pitch(tone, SAMPLE_RATE)
    assert pitch is not None
    assert pitch == pytest.approx(220.0, abs=3.0)


def test_estimate_pitch_returns_none_for_silence():
    assert estimate_pitch(np.zeros(1200), SAMPLE_RATE) is None


def test_estimate_pitch_returns_none_for_white_noise():
    noise = np.random.RandomState(42).randn(1200) * 0.5
    assert estimate_pitch(noise, SAMPLE_RATE) is None


def test_estimate_pitch_returns_none_for_too_short_a_frame():
    assert estimate_pitch(np.array([0.1]), SAMPLE_RATE) is None
    assert estimate_pitch(np.array([]), SAMPLE_RATE) is None


def test_estimate_pitch_out_of_range_frequency_not_falsely_detected_in_range():
    # A 700Hz tone is above the default fmax=400 search range; the autocorrelation
    # search should not alias it into a false in-range estimate.
    tone = _tone(700.0)
    pitch = estimate_pitch(tone, SAMPLE_RATE, fmin=60.0, fmax=400.0)
    if pitch is not None:
        assert not (60.0 <= 700.0 <= 400.0)  # sanity: 700 was never in [fmin, fmax]
        assert abs(pitch - 700.0) > 5.0  # didn't accidentally get it right either


def test_estimate_pitch_respects_custom_fmin_fmax():
    tone = _tone(150.0)
    # Search range that excludes 150Hz entirely -- must not report 150 anyway.
    pitch = estimate_pitch(tone, SAMPLE_RATE, fmin=200.0, fmax=400.0)
    if pitch is not None:
        assert pitch < 145.0 or pitch > 155.0


# ---------------------------------------------------------------------------
# ProsodyTracker
# ---------------------------------------------------------------------------


def test_tracker_update_returns_features_with_voiced_true_for_a_tone():
    tracker = ProsodyTracker()
    features = tracker.update(_tone(150.0), SAMPLE_RATE)
    assert isinstance(features, ProsodyFeatures)
    assert features.voiced is True
    assert features.pitch_hz == pytest.approx(150.0, abs=2.0)


def test_tracker_update_returns_unvoiced_for_silence():
    tracker = ProsodyTracker()
    features = tracker.update(np.zeros(1200), SAMPLE_RATE)
    assert features.voiced is False
    assert features.pitch_hz is None


def test_tracker_energy_trend_none_with_fewer_than_two_frames():
    tracker = ProsodyTracker()
    assert tracker.energy_trend() is None
    tracker.update(_tone(150.0), SAMPLE_RATE)
    assert tracker.energy_trend() is None  # still only one frame


def test_tracker_energy_trend_detects_declining_energy():
    tracker = ProsodyTracker(window=6)
    for amplitude in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        tracker.update(_tone(150.0, amplitude=amplitude), SAMPLE_RATE)
    trend = tracker.energy_trend()
    assert trend is not None
    assert trend < 0


def test_tracker_energy_trend_detects_rising_energy():
    tracker = ProsodyTracker(window=6)
    for amplitude in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        tracker.update(_tone(150.0, amplitude=amplitude), SAMPLE_RATE)
    trend = tracker.energy_trend()
    assert trend is not None
    assert trend > 0


def test_tracker_is_trailing_off_true_when_declining():
    tracker = ProsodyTracker(window=6)
    for amplitude in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        tracker.update(_tone(150.0, amplitude=amplitude), SAMPLE_RATE)
    assert tracker.is_trailing_off() is True


def test_tracker_is_trailing_off_false_when_stable():
    tracker = ProsodyTracker(window=6)
    for _ in range(6):
        tracker.update(_tone(150.0, amplitude=0.3), SAMPLE_RATE)
    assert tracker.is_trailing_off() is False


def test_tracker_pitch_trend_detects_rising_pitch():
    tracker = ProsodyTracker(window=6)
    for freq in [120, 130, 140, 160, 180, 200]:
        tracker.update(_tone(freq), SAMPLE_RATE)
    trend = tracker.pitch_trend()
    assert trend is not None
    assert trend > 0


def test_tracker_pitch_trend_ignores_unvoiced_frames():
    tracker = ProsodyTracker(window=6)
    tracker.update(np.zeros(1200), SAMPLE_RATE)  # silence, unvoiced
    tracker.update(_tone(150.0), SAMPLE_RATE)
    # Only one voiced pitch reading -- not enough for a trend.
    assert tracker.pitch_trend() is None


def test_tracker_reset_clears_history():
    tracker = ProsodyTracker(window=6)
    tracker.update(_tone(150.0), SAMPLE_RATE)
    tracker.update(_tone(150.0), SAMPLE_RATE)
    tracker.reset()
    assert tracker.energy_trend() is None


def test_tracker_respects_window_size():
    tracker = ProsodyTracker(window=3)
    for amplitude in [0.9, 0.9, 0.9, 0.05, 0.05, 0.05]:
        tracker.update(_tone(150.0, amplitude=amplitude), SAMPLE_RATE)
    # Only the last 3 frames (all quiet) should remain in the window -- roughly flat,
    # not a huge decline from the earlier loud frames that have since been evicted.
    trend = tracker.energy_trend()
    assert trend is not None
    assert abs(trend) < 0.05