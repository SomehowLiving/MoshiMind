# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reuse a speculative attempt when the real trigger confirms it (PHASES.md Phase 1.2).

The interesting move in predictive retrieval isn't just "start earlier" — it's
"don't pay for retrieval twice." If a guess is already running (or has already
finished) by the time the real trigger fires, the real trigger should adopt
that in-flight/completed work instead of cancelling it and starting over.

This primitive is deliberately generic (no RAG-specific types) so the
scheduling logic — the part with actual concurrency hazards — can be unit
tested against plain async functions, independent of ``RAGManager``'s
LLM/HTTP dependencies (which aren't importable in a torch/openai-free
environment; see the repo's execution audit).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class SpeculativeSlot:
    """Holds at most one in-flight or completed speculative attempt.

    Not safe to share across turns without calling ``clear()`` — callers are
    expected to reset the slot themselves when the conversation moves on
    (e.g. staleness is handled by the caller's own task-identity check, same
    as the confirmed retrieval path already does).
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    def has_attempt(self) -> bool:
        """True if a speculative attempt was started and not yet confirmed/cleared."""
        return self._task is not None

    def start(self, task_group: asyncio.TaskGroup, coro_factory: Callable[[], Awaitable[T]]) -> bool:
        """Start a speculative attempt if none is already running.

        Returns ``True`` if a new attempt was started, ``False`` if one was
        already in flight (the caller's heuristic should generally debounce
        before even calling this, but this guards against a race).
        """
        if self.has_attempt():
            return False
        self._task = task_group.create_task(coro_factory())
        return True

    async def confirm(self) -> T | None:
        """Adopt the speculative attempt: await it if still running, or take its
        already-computed result. Returns ``None`` if there was no attempt, it
        was cancelled, or it raised.
        """
        task = self._task
        self._task = None
        if task is None:
            return None
        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    def clear(self) -> None:
        """Cancel and discard any in-flight attempt without adopting its result.

        Fire-and-forget: does not await the cancellation. Use this from
        synchronous call sites (e.g. resetting on a new conversational turn);
        use ``cancel_and_clear`` when you can await and want to be sure the
        task has actually finished unwinding first (e.g. on teardown).
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def cancel_and_clear(self) -> None:
        """Cancel and await any in-flight attempt, suppressing its outcome."""
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
