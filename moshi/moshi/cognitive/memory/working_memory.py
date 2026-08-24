# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded conversational working memory (PHASES.md Phase 3.1).

``turn_manager.TurnManager.conversation_context`` (the live implementation
feeding the retrieval-LLM prompt today) is a single string that grows without
bound for the lifetime of a connection — fine for short conversations, a
real cost for long ones (see PHASES.md Phase 6, context management). This
formalizes what a bounded window and eviction policy actually look like, as a
standalone, torch-free, fully testable component.

Not wired into ``turn_manager.py`` yet: swapping the live context accumulator
out is a Phase 6 concern (context management) once this has been proven out
here, following the same "prove the primitive, wire it later" pattern as
``cognitive.rag_service``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    """A recent-turns window bounded by both turn count and total characters.

    Two independent limits because either can dominate depending on the
    conversation: many short turns (turn-count-bound) or a few very long
    ones (character-bound). Eviction always removes from the oldest end.
    """

    max_turns: int = 20
    max_chars: int = 4000
    _turns: deque = field(default_factory=deque)

    def add_turn(self, role: str, text: str) -> None:
        self._turns.append((role, text))
        while len(self._turns) > self.max_turns:
            self._turns.popleft()
        self._evict_by_chars()

    def _evict_by_chars(self) -> None:
        while self._char_count() > self.max_chars and len(self._turns) > 1:
            self._turns.popleft()

    def _char_count(self) -> int:
        return sum(len(text) for _, text in self._turns)

    def get_context(self) -> str:
        return "\n".join(f"{role}: {text}" for role, text in self._turns)

    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def reset(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
