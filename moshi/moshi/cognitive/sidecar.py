# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The cognitive sidecar (PHASES.md Phase 0.2 / Phase 2).

Today ``Channel`` talks directly to one hardcoded cognitive producer: the RAG
path (``RAGManager`` -> LLM -> ARC encoder, see ``inference_utils/channel.py``
and ``inference_utils/rag_manager.py``). Adding memory or tools the same way
means hand-wiring a second/third bespoke async path into ``Channel`` for each
one.

``CognitiveSidecar`` is the single seam instead: any number of
``CognitiveService`` producers register with it, ``Channel`` dispatches one
``CognitiveRequest`` per trigger, and the sidecar handles urgency-aware
scheduling, per-service circuit breaking, and turn-staleness discarding via
``TaskRegistry``. This module intentionally does not know anything about
Moshi's tensors — it hands back ``CognitiveResult`` objects with text +
``ConfidenceScore``; turning that into a conditioning tensor is the caller's
job (``channel.py``), same as it is today for the RAG path.

The existing RAG mechanism is not replaced by this — ``RAGManager`` can be
wrapped as one ``CognitiveService`` (see ``RAGManager`` -> sidecar adapter,
left for the Phase 1 integration step) so the trigger-token-driven path
keeps working unchanged while gaining the registry/breaker machinery for
free.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from .confidence import ConfidenceScore
from .task_registry import TaskHandle, TaskRegistry
from .urgency import Urgency

logger = logging.getLogger(__name__)

# Deadlines in seconds per urgency class, used when a request doesn't specify its own.
_DEFAULT_DEADLINES: dict[Urgency, float | None] = {
    Urgency.CRITICAL: 0.3,
    Urgency.NORMAL: 1.5,
    Urgency.BACKGROUND: None,
}


@dataclass(frozen=True)
class CognitiveRequest:
    """One request dispatched to a single cognitive service."""

    query: str
    urgency: Urgency = Urgency.NORMAL
    context: str = ""
    deadline: float | None = None
    """Override the urgency class's default deadline, in seconds. ``None`` uses the default."""

    def effective_deadline(self) -> float | None:
        return self.deadline if self.deadline is not None else _DEFAULT_DEADLINES[self.urgency]


@dataclass(frozen=True)
class CognitiveResult:
    """What a ``CognitiveService`` hands back."""

    text: str
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore)
    source: str = ""


class CognitiveService(Protocol):
    """Anything the sidecar can dispatch a request to (RAG, memory, a tool)."""

    name: str

    async def handle(self, request: CognitiveRequest) -> CognitiveResult: ...


class _CircuitBreaker:
    """Suspends a repeatedly-failing service instead of retrying into a dead endpoint every turn."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def is_open(self, now: float | None = None) -> bool:
        if self._opened_at is None:
            return False
        now = now if now is not None else time.monotonic()
        if now - self._opened_at >= self.cooldown_seconds:
            # Half-open: let the next call through as a trial.
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = now if now is not None else time.monotonic()


class CognitiveSidecar:
    """Dispatches cognitive requests to registered services, off Moshi's critical path.

    Callers never await the sidecar to get a result inline for anything above
    ``Urgency.CRITICAL`` — the intended usage is ``dispatch()`` returning a
    ``TaskHandle`` immediately, with the eventual ``CognitiveResult`` (or
    ``None`` if superseded/failed/breaker-open) delivered to ``on_result``.
    """

    def __init__(self, registry: TaskRegistry | None = None):
        self.registry = registry or TaskRegistry()
        self._services: dict[str, CognitiveService] = {}
        self._breakers: dict[str, _CircuitBreaker] = {}

    def register(self, service: CognitiveService) -> None:
        self._services[service.name] = service
        self._breakers[service.name] = _CircuitBreaker()

    def dispatch(
        self,
        conversation_id: str,
        service_name: str,
        request: CognitiveRequest,
        on_result: Callable[[TaskHandle, CognitiveResult | None], Awaitable[None]],
    ) -> TaskHandle:
        """Fire off ``service_name`` for ``request``; deliver the result via ``on_result``.

        Returns immediately with the minted ``TaskHandle`` so the caller can
        track/cancel it. The background task itself never raises into the
        caller — service failures, timeouts, and breaker-open are all
        reported as ``on_result(handle, None)``.
        """
        handle = self.registry.new_task(conversation_id, kind=service_name)
        asyncio.ensure_future(self._run(handle, service_name, request, on_result))
        return handle

    async def _run(
        self,
        handle: TaskHandle,
        service_name: str,
        request: CognitiveRequest,
        on_result: Callable[[TaskHandle, CognitiveResult | None], Awaitable[None]],
    ) -> None:
        service = self._services.get(service_name)
        breaker = self._breakers.get(service_name)
        if service is None:
            logger.error("[Sidecar] unknown cognitive service %r", service_name)
            await on_result(handle, None)
            return

        if breaker is not None and breaker.is_open():
            logger.warning("[Sidecar] service %r circuit open, skipping", service_name)
            await on_result(handle, None)
            return

        deadline = request.effective_deadline()
        try:
            if deadline is not None:
                result = await asyncio.wait_for(service.handle(request), timeout=deadline)
            else:
                result = await service.handle(request)
            if breaker is not None:
                breaker.record_success()
        except asyncio.TimeoutError:
            logger.warning("[Sidecar] service %r timed out after %ss", service_name, deadline)
            if breaker is not None:
                breaker.record_failure()
            await on_result(handle, None)
            return
        except Exception as e:
            logger.error("[Sidecar] service %r failed: %s", service_name, e)
            if breaker is not None:
                breaker.record_failure()
            await on_result(handle, None)
            return

        if not self.registry.is_current(handle):
            logger.info(
                "[Sidecar] discarding result from %r: turn advanced past turn_id=%s",
                service_name,
                handle.turn_id,
            )
            await on_result(handle, None)
            return

        await on_result(handle, result)
