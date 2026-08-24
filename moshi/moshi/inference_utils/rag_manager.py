# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RAG (Retrieval Augmented Generation) manager for reference text generation.

The manager owns per-channel reference history and pending generation task
lifecycle. Use it as an async context manager so pending background tasks are
cancelled and awaited on exit.
"""

import asyncio
import contextlib
import time
from typing import Awaitable, Callable
import logging
from ..cognitive.confidence import ConfidenceScore
from ..cognitive.speculation import SpeculativeSlot
from ..reference import LLMReferenceGenerator
from ..reference.llm_reference_generator import ReferenceHistory

logger = logging.getLogger(__name__)

# How much to discount a speculative (unconfirmed) reference's trust before applying
# it, relative to the same reference once the real rag_token_id trigger confirms it.
_SPECULATIVE_CONFIDENCE_DISCOUNT = 0.6


class RAGManager:
    """Manages asynchronous RAG reference generation scoped to one channel."""

    def __init__(
        self,
        reference_generator: LLMReferenceGenerator,
        rag_timeout: float = 1.5,
        max_tokens: int = 512,
        gt_reference_text: str | None = None,
        metrics=None,
        telemetry=None,
    ):
        self.reference_generator = reference_generator
        self.rag_timeout = rag_timeout
        self.max_tokens = max_tokens
        self.gt_reference_text = gt_reference_text
        self.metrics = metrics
        self.telemetry = telemetry
        self._history: ReferenceHistory = []
        self._wait_steps_remaining: int = 0
        self._wait_event: asyncio.Event | None = None
        self._pending_task: asyncio.Task | None = None
        self._speculative = SpeculativeSlot()
        self._stack: contextlib.AsyncExitStack | None = None
        self._active_profile_id: str | None = None

    async def __aenter__(self) -> "RAGManager":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        self._stack.push_async_callback(self._speculative.cancel_and_clear)
        self._stack.push_async_callback(self._cancel_and_await_pending)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._stack is not None
        try:
            return await self._stack.__aexit__(exc_type, exc, tb)
        finally:
            self._stack = None

    async def _cancel_and_await_pending(self):
        task = self._pending_task
        self._pending_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def get_reference_text(
        self, context: str, kind: str = "confirmed"
    ) -> tuple[str, str, float, str, ConfidenceScore]:
        """Returns (query_context, reference_text, elapsed_seconds, lm_display_name, confidence).

        ``kind`` is "confirmed" (triggered by rag_token_id) or "speculative"
        (``PredictiveTrigger``) — recorded in telemetry only, doesn't affect behavior.
        """
        try:
            logger.info("[Reference] Generating reference")

            if self.gt_reference_text is not None:
                logger.info(f"[Reference] Using ground truth reference text: {self.gt_reference_text}")
                if self.telemetry is not None:
                    self.telemetry.record(context, self.gt_reference_text, 1.0, 0.0, kind=kind)
                return "", self.gt_reference_text, 0.0, "Ground truth", ConfidenceScore()

            retrieval_start_time = time.time()
            query, reference_text, num_turns, lm_label = await self.reference_generator.generate_reference_text(
                context,
                self._history,
                active_profile_id=self._active_profile_id,
                llm_call_timeout=self.rag_timeout,
                max_tokens=self.max_tokens,
            )
            retrieval_elapsed = time.time() - retrieval_start_time
            if num_turns > 0 and reference_text:
                self._history.append((num_turns, reference_text))
            confidence = ConfidenceScore.heuristic_from_llm_reference(
                reference_text, retrieval_elapsed, self.rag_timeout
            )
            logger.info(
                f"[Reference] Generated reference in {retrieval_elapsed:.3f}s "
                f"(strength={confidence.strength():.2f}): {reference_text}",
            )
            if self.telemetry is not None:
                self.telemetry.record(query, reference_text, confidence.strength(), retrieval_elapsed, kind=kind)
            return query, reference_text, retrieval_elapsed, lm_label, confidence
        except asyncio.TimeoutError:
            logger.warning(
                f"[Reference] Reference generation timed out after {self.rag_timeout}s, returning empty string"
            )
            if self.metrics is not None:
                self.metrics.increment("rag_llm_timeouts_total")
            if self.telemetry is not None:
                self.telemetry.record(context, "", 0.0, self.rag_timeout, kind=kind)
            return "", "", self.rag_timeout, "", ConfidenceScore.empty()
        except Exception as e:
            logger.error(f"[Reference] Error generating reference: {e}, returning empty string")
            if self.metrics is not None:
                self.metrics.increment("rag_llm_errors_total")
            if self.telemetry is not None:
                self.telemetry.record(context, "", 0.0, self.rag_timeout, kind=kind)
            return "", "", self.rag_timeout, "", ConfidenceScore.empty()

    def warmup(self):
        """Warmup reference generator."""
        self.reference_generator.warmup()

    def set_retrieval_profile_id(self, profile_id: str) -> None:
        self._active_profile_id = profile_id

    async def trigger(
        self,
        task_group: asyncio.TaskGroup,
        wait_steps: int = 0,
        handle_reference_fn: Callable[..., Awaitable[None]] | None = None,
        context_provider: Callable[[], str] | None = None,
    ):
        """Trigger reference text generation in background (the confirmed rag_token_id path).

        If a speculative retrieval (``trigger_speculative``) is already in flight or has
        already completed for this turn, adopt it instead of paying for retrieval twice.
        """
        if self._stack is None:
            raise RuntimeError("RAGManager.trigger called outside of `async with` scope")

        await self._cancel_and_await_pending()

        if self._speculative.has_attempt():
            logger.info("[Reference] confirming trigger: reusing speculative retrieval")
            self._pending_task = task_group.create_task(
                self._confirm_speculative(wait_steps, handle_reference_fn, context_provider)
            )
            return

        if wait_steps > 0:
            self._wait_steps_remaining = wait_steps
            self._wait_event = asyncio.Event()
            logger.info(f"[Reference] Started waiting for {wait_steps} steps")
        else:
            self._wait_event = None
            self._wait_steps_remaining = 0

        self._pending_task = task_group.create_task(self._background_task(handle_reference_fn, context_provider))

    async def _confirm_speculative(
        self,
        wait_steps: int,
        handle_reference_fn: Callable[..., Awaitable[None]] | None,
        context_provider: Callable[[], str] | None,
    ):
        """Adopt an in-flight/completed speculative attempt; fall back to a fresh
        retrieval if it produced nothing usable (still cheaper than never having tried)."""
        try:
            result = await self._speculative.confirm()
        except asyncio.CancelledError:
            logger.info("[Reference] speculative confirmation cancelled")
            raise

        if result is not None and result[0]:
            reference_text, lm_label, confidence = result
            if self.metrics is not None:
                self.metrics.increment("rag_speculative_confirmed_total")
            logger.info(f"[Reference] speculative retrieval confirmed (strength={confidence.strength():.2f})")
            if handle_reference_fn is not None:
                await handle_reference_fn(reference_text, lm_label, confidence=confidence)
            return

        logger.info("[Reference] speculative attempt produced nothing usable; falling back to fresh retrieval")
        if self.metrics is not None:
            self.metrics.increment("rag_speculative_miss_total")
        if wait_steps > 0:
            self._wait_steps_remaining = wait_steps
            self._wait_event = asyncio.Event()
        else:
            self._wait_event = None
            self._wait_steps_remaining = 0
        await self._background_task(handle_reference_fn, context_provider)

    async def trigger_speculative(
        self,
        task_group: asyncio.TaskGroup,
        handle_reference_fn: Callable[..., Awaitable[None]] | None = None,
        context_provider: Callable[[], str] | None = None,
        confidence_discount: float = _SPECULATIVE_CONFIDENCE_DISCOUNT,
    ):
        """Start a retrieval before Moshi's own ``rag_token_id`` fires, from a heuristic
        guess (see ``cognitive.predictive_trigger``) that one will be needed soon.

        No-op if a confirmed retrieval is already in flight (never let a guess preempt
        real work) or if a speculative attempt is already running/pending confirmation
        for this turn.
        """
        if self._stack is None:
            raise RuntimeError("RAGManager.trigger_speculative called outside of `async with` scope")
        if self._pending_task is not None and not self._pending_task.done():
            logger.info("[Reference] skipping speculative trigger: a confirmed retrieval is already in flight")
            return
        if self._speculative.has_attempt():
            return

        async def _run():
            context = context_provider() if context_provider is not None else ""
            _, reference_text, _, lm_label, confidence = await self.get_reference_text(context, kind="speculative")
            if reference_text and handle_reference_fn is not None:
                provisional = confidence.scaled(confidence_discount)
                logger.info(
                    f"[Reference] applying provisional speculative reference "
                    f"(strength={provisional.strength():.2f}, pending confirmation)"
                )
                await handle_reference_fn(reference_text, lm_label, confidence=provisional)
            return reference_text, lm_label, confidence

        started = self._speculative.start(task_group, _run)
        if started:
            logger.info("[Reference] started speculative retrieval ahead of rag_token_id")
            if self.metrics is not None:
                self.metrics.increment("rag_speculative_started_total")

    def clear_speculative(self):
        """Drop any speculative attempt without adopting it — call when a new user
        turn starts so a stale, never-confirmed guess doesn't block the next one."""
        self._speculative.clear()

    async def _background_task(
        self,
        handle_reference_fn: Callable[..., Awaitable[None]] | None,
        context_provider: Callable[[], str] | None,
    ):
        logger.info("[Reference] Started new reference generation task in background")
        try:
            if self._wait_event is not None:
                await self._wait_event.wait()
                self._wait_event = None
            logger.info("[Reference] Waiting ended (including zero wait_steps)")
            if context_provider is not None:
                context = context_provider()
            else:
                context = ""
                logger.warning("[Reference] No context provider supplied, generating reference with empty context")
            logger.info(
                f"[Reference] Triggering retrieval with context_len={len(context)} snippet='...{context[-200:]}'"
            )
            _, reference_text, _, lm_label, confidence = await self.get_reference_text(context)
            if handle_reference_fn is not None:
                await handle_reference_fn(reference_text, lm_label, confidence=confidence)
            logger.info("[Reference] Background reference generation task completed")
        except asyncio.CancelledError:
            logger.info("[Reference] Reference generation cancelled")
            raise
        except Exception as e:
            logger.error(f"[Reference] Error generating reference: {e}")

    def step(self):
        """Signal that a step has passed. Used for step-based waiting."""
        if self._wait_steps_remaining > 0:
            self._wait_steps_remaining -= 1
            if self._wait_steps_remaining == 0 and self._wait_event is not None:
                self._wait_event.set()

    def cancel_pending(self):
        """Cancel any pending reference generation task."""
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

    def reset(self, gt_reference_text: str | None = None):
        """Reset RAG manager state."""
        self.cancel_pending()
        self.clear_speculative()
        self._wait_steps_remaining = 0
        self._wait_event = None
        self.gt_reference_text = gt_reference_text
        self._history = []
