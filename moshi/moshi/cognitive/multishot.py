# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bounding how many retrievals one model response may spend (PHASES.md Phase 1.3).

The plumbing for a second conditioning update mid-response already works without
any change here: ``LMGen.update_streaming_sum_tensors`` (``models/lm.py``) accepts
a fresh tensor for a slot at any time, and ``Channel``/``RAGManager`` already treat
every ``rag_token_id`` emission as an independent trigger — the model can already
ask twice in one turn and get two conditioning updates.

What's missing is *judgment*: repeated ``rag_token_id`` emissions in quick
succession are as likely to be the model looping/uncertain as they are to be N
genuinely distinct information needs, and each one costs a full retrieval-LLM +
ARC-encoder round trip. ``MultiShotGate`` is a cheap policy in front of that
existing plumbing — it does not change what a retrieval does, only whether a
given trigger is honored as a fresh one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultiShotGate:
    """Per-response budget for confirmed retrieval triggers.

    ``now`` is passed in explicitly rather than read from the clock internally,
    so this stays a plain, fully deterministic function of its inputs — trivial
    to unit test and independent of any particular time source.
    """

    max_shots_per_turn: int = 3
    min_cooldown_seconds: float = 2.0
    _shots_this_turn: int = 0
    _last_shot_time: float | None = None

    def reset(self) -> None:
        """Call when a new model response/turn begins."""
        self._shots_this_turn = 0
        self._last_shot_time = None

    def should_allow(self, now: float) -> bool:
        """Whether a trigger arriving at ``now`` should be honored as a fresh shot."""
        if self._shots_this_turn >= self.max_shots_per_turn:
            return False
        if self._last_shot_time is not None and (now - self._last_shot_time) < self.min_cooldown_seconds:
            return False
        return True

    def record_shot(self, now: float) -> None:
        """Call once a trigger has actually been allowed and dispatched."""
        self._shots_this_turn += 1
        self._last_shot_time = now

    @property
    def shots_this_turn(self) -> int:
        return self._shots_this_turn
