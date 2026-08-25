# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Formalized conversational state (PHASES.md Phase 4.1).

Today this state is implicit and scattered: ``turn_manager.TurnManager.active_speaker``
tracks who's nominally talking, ``Channel._rag_seq``/``_turn_index`` track RAG and
memory turn identity separately, and nothing tracks whether the user is currently
talking *over* the model (full-duplex overlap) at all. This pulls the pieces PHASES.md
Phase 4 needs -- turn id, speaking state, interruption state, user confidence -- into
one explicit object, without trying to replace ``TurnManager`` (which still owns VAD
smoothing and text buffering) or duplicate the RAG-specific identity already handled
by ``cognitive.task_registry``.

Deliberately scoped to what Phase 4 needs: the fuller ``ConversationState`` sketched
in the original direction doc (user_emotion, topic, retrieval_state, memory_context)
either already exists elsewhere (retrieval state in ``RAGManager``, memory context in
``cognitive.memory``) or belongs to a later phase (emotion is Phase 5).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class SpeakingState(enum.Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    MODEL_SPEAKING = "model_speaking"
    OVERLAPPING = "overlapping"
    """Both parties producing voice at once -- the case a half-duplex turn-taking
    model can't represent at all, and the whole point of tracking this separately
    from TurnManager's single active_speaker."""


class InterruptionKind(enum.Enum):
    NONE = "none"
    BACKCHANNEL = "backchannel"
    """A short acknowledgement ("mhm", "right") while the model is speaking -- the
    user is listening, not trying to take the floor."""
    HESITATION = "hesitation"
    """The user started speaking, trailed off, or used a filler word -- ambiguous
    signal, not yet a clear intent to interrupt."""
    BARGE_IN = "barge_in"
    """Sustained, substantive speech while the model is talking -- a real attempt
    to take the floor."""


class InterruptionDecision(enum.Enum):
    CONTINUE = "continue"
    """The model should keep talking through this."""
    YIELD = "yield"
    """The model should stop and let the user speak -- see PHASES.md Phase 4.3 for
    why this decision is not yet acted on by cutting live generation."""
    WAIT = "wait"
    """Not enough signal yet; re-classify as more speech arrives."""


@dataclass
class ConversationState:
    """Per-channel conversational state, updated as turns and overlaps occur."""

    turn_id: int = 0
    speaking_state: SpeakingState = SpeakingState.IDLE
    interruption_state: InterruptionKind = InterruptionKind.NONE
    user_confidence: float = 1.0
    """Inverse of hesitation: 1.0 = fluent/confident, lower = hesitant. Not derived
    from prosody (no real signal for that yet -- Phase 5); only from the interruption
    classifier's HESITATION calls for now."""

    def begin_user_turn(self) -> None:
        self.turn_id += 1
        self.speaking_state = SpeakingState.USER_SPEAKING
        self.interruption_state = InterruptionKind.NONE
        self.user_confidence = 1.0

    def begin_model_turn(self) -> None:
        self.speaking_state = SpeakingState.MODEL_SPEAKING
        self.interruption_state = InterruptionKind.NONE

    def note_overlap(self, kind: InterruptionKind) -> None:
        """Call when user speech is detected while the model is (nominally) talking."""
        self.interruption_state = kind
        if kind == InterruptionKind.BARGE_IN:
            self.speaking_state = SpeakingState.OVERLAPPING
            self.user_confidence = 1.0
        elif kind == InterruptionKind.HESITATION:
            self.user_confidence = min(self.user_confidence, 0.4)
        # BACKCHANNEL: acknowledged in interruption_state but doesn't change
        # speaking_state or confidence -- the model is expected to keep talking.

    def resolve_overlap_as_yield(self) -> None:
        """The model actually stops and the user takes the floor."""
        self.speaking_state = SpeakingState.USER_SPEAKING
        self.interruption_state = InterruptionKind.NONE

    def resolve_overlap_as_continue(self) -> None:
        """The overlap wasn't a real barge-in; the model keeps talking."""
        self.speaking_state = SpeakingState.MODEL_SPEAKING
        self.interruption_state = InterruptionKind.NONE

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "speaking_state": self.speaking_state.value,
            "interruption_state": self.interruption_state.value,
            "user_confidence": self.user_confidence,
        }


def decide_interruption_response(state: ConversationState, kind: InterruptionKind) -> InterruptionDecision:
    """Policy: given the current state and a freshly classified overlap, should the
    model yield the floor, keep talking, or wait for more signal?

    Pure function, independent of ``ConversationState`` mutation, so the policy
    itself is testable without needing to also exercise the state machine.
    """
    if kind == InterruptionKind.BARGE_IN:
        return InterruptionDecision.YIELD
    if kind == InterruptionKind.BACKCHANNEL:
        return InterruptionDecision.CONTINUE
    if kind == InterruptionKind.HESITATION:
        return InterruptionDecision.WAIT
    return InterruptionDecision.CONTINUE
