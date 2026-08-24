# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Identity and staleness tracking for asynchronous cognitive operations.

Generalizes the ad-hoc ``trigger_seq`` counter added to ``Channel`` for RAG
(see ``inference_utils/channel.py``): once memory lookups, tool calls, and
predictive retrieval all race against the live conversation, every one of
them needs the same guard — "is this result still for the turn that's
currently live, or did the conversation move on while I was working?"

A task is identified by ``(conversation_id, turn_id, task_id)``. ``turn_id``
advances every time the user's turn changes; a task started against an older
turn is stale the moment ``turn_id`` advances further, regardless of whether
the task itself has completed, failed, or is still running.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskHandle:
    """Identity of one in-flight (or completed) cognitive operation."""

    conversation_id: str
    turn_id: int
    task_id: int
    kind: str = "generic"

    def is_stale(self, registry: "TaskRegistry") -> bool:
        return registry.current_turn(self.conversation_id) != self.turn_id


@dataclass
class _ConversationState:
    turn_id: int = 0
    live_task_ids: set[int] = field(default_factory=set)


class TaskRegistry:
    """Tracks the current turn per conversation and hands out task identities.

    Not thread-safe by design: intended to be driven from a single asyncio
    event loop per process, matching how ``Channel``/``RAGManager`` already
    operate (one ``asyncio.TaskGroup`` per channel).
    """

    def __init__(self) -> None:
        self._conversations: dict[str, _ConversationState] = {}
        self._task_id_counter = itertools.count(1)

    def _state(self, conversation_id: str) -> _ConversationState:
        return self._conversations.setdefault(conversation_id, _ConversationState())

    def current_turn(self, conversation_id: str) -> int:
        return self._state(conversation_id).turn_id

    def advance_turn(self, conversation_id: str) -> int:
        """Call once per new user/model turn. Invalidates every task from the prior turn."""
        state = self._state(conversation_id)
        state.turn_id += 1
        state.live_task_ids.clear()
        return state.turn_id

    def new_task(self, conversation_id: str, kind: str = "generic") -> TaskHandle:
        """Mint a task handle bound to the conversation's *current* turn."""
        state = self._state(conversation_id)
        task_id = next(self._task_id_counter)
        state.live_task_ids.add(task_id)
        return TaskHandle(conversation_id=conversation_id, turn_id=state.turn_id, task_id=task_id, kind=kind)

    def is_current(self, handle: TaskHandle) -> bool:
        """True if ``handle`` was minted for the turn that is still live."""
        return self.current_turn(handle.conversation_id) == handle.turn_id

    def forget_conversation(self, conversation_id: str) -> None:
        """Drop all state for a closed conversation (call on channel teardown)."""
        self._conversations.pop(conversation_id, None)
