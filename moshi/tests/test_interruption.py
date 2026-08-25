# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Phase 4 (full-duplex interruption / anticipation).

Torch-free, same as the rest of the cognitive-layer tests. Exercises the
classifier against recorded VAD/ASR event sequences per PHASES.md's own
acceptance criteria for this phase, without needing audio hardware.
"""

from __future__ import annotations

from moshi.cognitive import (
    ConversationState,
    InterruptionClassifier,
    InterruptionDecision,
    InterruptionKind,
    SpeakingState,
    decide_interruption_response,
)


# ---------------------------------------------------------------------------
# ConversationState (Phase 4.1)
# ---------------------------------------------------------------------------


def test_initial_state_is_idle():
    state = ConversationState()
    assert state.speaking_state == SpeakingState.IDLE
    assert state.interruption_state == InterruptionKind.NONE
    assert state.turn_id == 0
    assert state.user_confidence == 1.0


def test_begin_user_turn_increments_turn_id_and_resets():
    state = ConversationState()
    state.note_overlap(InterruptionKind.HESITATION)  # dirty the state first
    state.begin_user_turn()
    assert state.turn_id == 1
    assert state.speaking_state == SpeakingState.USER_SPEAKING
    assert state.interruption_state == InterruptionKind.NONE
    assert state.user_confidence == 1.0


def test_begin_user_turn_called_twice_increments_each_time():
    state = ConversationState()
    state.begin_user_turn()
    state.begin_user_turn()
    assert state.turn_id == 2


def test_begin_model_turn_does_not_change_turn_id():
    state = ConversationState()
    state.begin_user_turn()
    state.begin_model_turn()
    assert state.turn_id == 1
    assert state.speaking_state == SpeakingState.MODEL_SPEAKING


def test_note_overlap_barge_in_sets_overlapping_and_full_confidence():
    state = ConversationState()
    state.begin_model_turn()
    state.note_overlap(InterruptionKind.BARGE_IN)
    assert state.speaking_state == SpeakingState.OVERLAPPING
    assert state.interruption_state == InterruptionKind.BARGE_IN
    assert state.user_confidence == 1.0


def test_note_overlap_hesitation_lowers_confidence_but_not_speaking_state():
    state = ConversationState()
    state.begin_model_turn()
    state.note_overlap(InterruptionKind.HESITATION)
    assert state.speaking_state == SpeakingState.MODEL_SPEAKING  # unchanged
    assert state.interruption_state == InterruptionKind.HESITATION
    assert state.user_confidence == 0.4


def test_note_overlap_hesitation_never_raises_confidence_back_up():
    state = ConversationState()
    state.user_confidence = 0.2
    state.note_overlap(InterruptionKind.HESITATION)
    assert state.user_confidence == 0.2  # min() keeps the lower value


def test_note_overlap_backchannel_leaves_speaking_state_and_confidence_alone():
    state = ConversationState()
    state.begin_model_turn()
    state.note_overlap(InterruptionKind.BACKCHANNEL)
    assert state.speaking_state == SpeakingState.MODEL_SPEAKING
    assert state.interruption_state == InterruptionKind.BACKCHANNEL
    assert state.user_confidence == 1.0


def test_resolve_overlap_as_yield_transitions_to_user_speaking():
    state = ConversationState()
    state.begin_model_turn()
    state.note_overlap(InterruptionKind.BARGE_IN)
    state.resolve_overlap_as_yield()
    assert state.speaking_state == SpeakingState.USER_SPEAKING
    assert state.interruption_state == InterruptionKind.NONE


def test_resolve_overlap_as_continue_returns_to_model_speaking():
    state = ConversationState()
    state.begin_model_turn()
    state.note_overlap(InterruptionKind.HESITATION)
    state.resolve_overlap_as_continue()
    assert state.speaking_state == SpeakingState.MODEL_SPEAKING
    assert state.interruption_state == InterruptionKind.NONE


def test_to_dict_reflects_current_state():
    state = ConversationState()
    state.begin_user_turn()
    d = state.to_dict()
    assert d["turn_id"] == 1
    assert d["speaking_state"] == "user_speaking"
    assert d["interruption_state"] == "none"
    assert d["user_confidence"] == 1.0


# ---------------------------------------------------------------------------
# decide_interruption_response (policy)
# ---------------------------------------------------------------------------


def test_decide_barge_in_yields():
    state = ConversationState()
    assert decide_interruption_response(state, InterruptionKind.BARGE_IN) == InterruptionDecision.YIELD


def test_decide_backchannel_continues():
    state = ConversationState()
    assert decide_interruption_response(state, InterruptionKind.BACKCHANNEL) == InterruptionDecision.CONTINUE


def test_decide_hesitation_waits():
    state = ConversationState()
    assert decide_interruption_response(state, InterruptionKind.HESITATION) == InterruptionDecision.WAIT


def test_decide_none_continues():
    state = ConversationState()
    assert decide_interruption_response(state, InterruptionKind.NONE) == InterruptionDecision.CONTINUE


# ---------------------------------------------------------------------------
# InterruptionClassifier (Phase 4.2) -- against recorded VAD/ASR event sequences
# ---------------------------------------------------------------------------


def test_classifier_empty_text_is_none():
    clf = InterruptionClassifier()
    assert clf.classify("", vad_history=[0.1, 0.1, 0.1]) == InterruptionKind.NONE


def test_classifier_short_backchannel_phrase():
    clf = InterruptionClassifier()
    assert clf.classify("mhm", vad_history=[0.6, 0.6, 0.6]) == InterruptionKind.BACKCHANNEL
    assert clf.classify("yeah", vad_history=[0.6]) == InterruptionKind.BACKCHANNEL
    assert clf.classify("okay,", vad_history=[]) == InterruptionKind.BACKCHANNEL  # trailing punctuation stripped


def test_classifier_backchannel_phrase_used_in_a_longer_sentence_is_not_backchannel():
    clf = InterruptionClassifier()
    # "right" as a filler word inside a real sentence, not a standalone acknowledgement.
    result = clf.classify("right so I think we should actually go", vad_history=[0.1, 0.1, 0.1, 0.1])
    assert result != InterruptionKind.BACKCHANNEL


def test_classifier_sustained_voice_with_enough_words_is_barge_in():
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    # Recorded-style sequence: VAD stays low (voice present) for 3+ consecutive frames
    # while the partial transcript accumulates a real sentence.
    vad_sequence = [0.6, 0.6, 0.2, 0.15, 0.1]  # settles into sustained voice
    result = clf.classify("wait no that's not right at all", vad_history=vad_sequence)
    assert result == InterruptionKind.BARGE_IN


def test_classifier_short_burst_without_sustained_voice_is_not_barge_in():
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    # Word count high enough, but VAD never confirms sustained voice (e.g. a brief
    # spike that immediately settles back to silence) -- shouldn't fire barge-in.
    vad_sequence = [0.9, 0.9, 0.9]
    result = clf.classify("no wait that's actually wrong", vad_history=vad_sequence)
    assert result != InterruptionKind.BARGE_IN


def test_classifier_insufficient_vad_history_does_not_barge_in_even_with_words():
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    # Not enough frames recorded yet to confirm "sustained" -- ambiguous, not barge-in.
    result = clf.classify("no wait that's actually wrong", vad_history=[0.1, 0.1])
    assert result == InterruptionKind.NONE


def test_classifier_ends_with_hesitation_marker():
    clf = InterruptionClassifier()
    # Short enough, and VAD not sustained, that the barge-in condition can't fire --
    # isolates the trailing-filler-word signal specifically.
    result = clf.classify("well um", vad_history=[0.9, 0.9])
    assert result == InterruptionKind.HESITATION


def test_classifier_hesitation_marker_yields_to_barge_in_when_speech_is_sustained_and_long():
    """A trailing filler word doesn't override a genuinely sustained, substantive
    utterance -- the stronger barge-in signal wins."""
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    result = clf.classify("I think that maybe um", vad_history=[0.1, 0.1, 0.1])
    assert result == InterruptionKind.BARGE_IN


def test_classifier_very_short_non_backchannel_utterance_is_hesitation():
    clf = InterruptionClassifier()
    result = clf.classify("wait", vad_history=[0.9, 0.9])
    assert result == InterruptionKind.HESITATION


def test_classifier_respects_custom_vad_threshold():
    clf = InterruptionClassifier(min_barge_in_words=2, sustained_voice_frames=2)
    vad_sequence = [0.35, 0.35]
    # With a stricter (lower) threshold, 0.35 no longer counts as "voice present".
    assert clf.classify("please stop now", vad_history=vad_sequence, vad_threshold=0.3) != InterruptionKind.BARGE_IN
    # With the default-ish looser threshold, it does.
    assert clf.classify("please stop now", vad_history=vad_sequence, vad_threshold=0.5) == InterruptionKind.BARGE_IN


def test_classifier_declining_energy_downgrades_barge_in_to_hesitation():
    """Word count + VAD alone would call this a barge-in, but real acoustic evidence
    (declining energy -- the user is trailing off, not committing) overrides it."""
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    vad_sequence = [0.2, 0.15, 0.1]
    result = clf.classify("wait no that's not right", vad_history=vad_sequence, energy_trend=-0.05)
    assert result == InterruptionKind.HESITATION


def test_classifier_rising_energy_does_not_prevent_barge_in():
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    vad_sequence = [0.2, 0.15, 0.1]
    result = clf.classify("wait no that's not right", vad_history=vad_sequence, energy_trend=0.05)
    assert result == InterruptionKind.BARGE_IN


def test_classifier_energy_trend_none_behaves_like_no_acoustic_signal():
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    vad_sequence = [0.2, 0.15, 0.1]
    result = clf.classify("wait no that's not right", vad_history=vad_sequence, energy_trend=None)
    assert result == InterruptionKind.BARGE_IN


def test_classifier_energy_trend_does_not_affect_backchannel_classification():
    clf = InterruptionClassifier()
    # Even with declining energy, a clear backchannel phrase is still a backchannel.
    assert clf.classify("mhm", vad_history=[0.6], energy_trend=-0.5) == InterruptionKind.BACKCHANNEL


def test_classifier_full_conversation_style_sequence():
    """A recorded-style event sequence: backchannel, then a hesitation, then a real
    barge-in -- exercising the classifier the way it would actually be called, once
    per incoming partial-transcript update, per PHASES.md's acceptance criteria."""
    clf = InterruptionClassifier(min_barge_in_words=4, sustained_voice_frames=3)
    events = [
        ("mhm", [0.6, 0.6, 0.6], InterruptionKind.BACKCHANNEL),
        ("um", [0.6, 0.6, 0.6], InterruptionKind.HESITATION),
        ("wait", [0.6, 0.6, 0.6], InterruptionKind.HESITATION),
        ("wait actually hold on a second", [0.2, 0.15, 0.1], InterruptionKind.BARGE_IN),
    ]
    for text, vad, expected in events:
        assert clf.classify(text, vad_history=vad) == expected