# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the cognitive sidecar layer (PHASES.md Phase 0).

Deliberately torch-free and GPU-free: these exercise scheduling, staleness,
and scoring logic in isolation from the real Moshi generation loop, which is
untestable in a non-GPU environment (see the repo's execution audit).
"""

from __future__ import annotations

import asyncio

import pytest

from moshi.cognitive import (
    CognitiveRequest,
    CognitiveResult,
    CognitiveSidecar,
    ConfidenceScore,
    TaskRegistry,
    Urgency,
    classify_urgency,
)


# ---------------------------------------------------------------------------
# TaskRegistry / staleness
# ---------------------------------------------------------------------------


def test_task_registry_new_task_binds_to_current_turn():
    reg = TaskRegistry()
    handle = reg.new_task("conv-1", kind="rag")
    assert handle.conversation_id == "conv-1"
    assert handle.turn_id == 0
    assert reg.is_current(handle)


def test_task_registry_advance_turn_invalidates_prior_tasks():
    reg = TaskRegistry()
    handle = reg.new_task("conv-1", kind="rag")
    assert reg.is_current(handle)

    reg.advance_turn("conv-1")

    assert not reg.is_current(handle)
    assert handle.is_stale(reg)


def test_task_registry_is_per_conversation():
    reg = TaskRegistry()
    h1 = reg.new_task("conv-1")
    h2 = reg.new_task("conv-2")

    reg.advance_turn("conv-1")

    assert not reg.is_current(h1)
    assert reg.is_current(h2)  # unaffected by conv-1's turn advancing


def test_task_registry_forget_conversation_resets_state():
    reg = TaskRegistry()
    reg.advance_turn("conv-1")
    reg.advance_turn("conv-1")
    assert reg.current_turn("conv-1") == 2

    reg.forget_conversation("conv-1")

    assert reg.current_turn("conv-1") == 0  # fresh state, as if never seen


def test_task_ids_are_unique_across_conversations():
    reg = TaskRegistry()
    h1 = reg.new_task("conv-1")
    h2 = reg.new_task("conv-2")
    assert h1.task_id != h2.task_id


# ---------------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------------


def test_confidence_full_trust_gives_full_strength():
    score = ConfidenceScore(relevance=1.0, confidence=1.0, freshness=1.0)
    assert score.strength() == pytest.approx(1.0)


def test_confidence_zero_confidence_gives_zero_strength_even_if_relevant():
    score = ConfidenceScore(relevance=1.0, confidence=0.0, freshness=1.0)
    assert score.strength() == 0.0


def test_confidence_empty_is_zero_strength():
    assert ConfidenceScore.empty().strength() == 0.0


def test_confidence_clamps_out_of_range_inputs():
    score = ConfidenceScore(relevance=1.5, confidence=-0.5, freshness=2.0)
    assert score.relevance == 1.0
    assert score.confidence == 0.0
    assert score.freshness == 1.0


def test_heuristic_confidence_empty_reference_is_zero_strength():
    score = ConfidenceScore.heuristic_from_llm_reference("", elapsed_seconds=0.1, timeout_seconds=1.5)
    assert score.strength() == 0.0


def test_heuristic_confidence_whitespace_only_reference_is_zero_strength():
    score = ConfidenceScore.heuristic_from_llm_reference("   \n\t  ", elapsed_seconds=0.1, timeout_seconds=1.5)
    assert score.strength() == 0.0


def test_heuristic_confidence_fast_answer_beats_rushed_answer():
    fast = ConfidenceScore.heuristic_from_llm_reference(
        "NVIDIA's CEO is Jensen Huang.", elapsed_seconds=0.1, timeout_seconds=1.5
    )
    rushed = ConfidenceScore.heuristic_from_llm_reference(
        "NVIDIA's CEO is Jensen Huang.", elapsed_seconds=1.45, timeout_seconds=1.5
    )
    assert fast.strength() > rushed.strength()


def test_heuristic_confidence_very_short_reference_is_penalized():
    short = ConfidenceScore.heuristic_from_llm_reference("Yes.", elapsed_seconds=0.1, timeout_seconds=1.5)
    long = ConfidenceScore.heuristic_from_llm_reference(
        "Yes, that is correct according to the source.", elapsed_seconds=0.1, timeout_seconds=1.5
    )
    assert short.strength() < long.strength()


def test_heuristic_confidence_zero_timeout_does_not_crash():
    score = ConfidenceScore.heuristic_from_llm_reference("some answer text", elapsed_seconds=0.0, timeout_seconds=0.0)
    assert 0.0 <= score.strength() <= 1.0


def test_confidence_geometric_mean_penalizes_one_weak_axis_more_than_arithmetic_mean():
    # relevance perfect, confidence very low: should read as "mostly untrustworthy",
    # not average out to a comfortable ~0.6 the way (1.0+0.9+0.9)/3 would.
    score = ConfidenceScore(relevance=1.0, confidence=0.1, freshness=1.0)
    geometric = score.strength()
    arithmetic = (score.relevance + score.confidence + score.freshness) / 3
    assert geometric < arithmetic


# ---------------------------------------------------------------------------
# Urgency classification
# ---------------------------------------------------------------------------


def test_arithmetic_query_is_critical():
    assert classify_urgency("27 * 43") == Urgency.CRITICAL


def test_memory_kind_is_background():
    assert classify_urgency("what did I say about my project", kind="memory") == Urgency.BACKGROUND


def test_generic_rag_query_is_normal():
    assert classify_urgency("who is the CEO of NVIDIA", kind="rag") == Urgency.NORMAL


def test_calculator_tool_kind_is_critical_regardless_of_text():
    assert classify_urgency("what's the tip on $42.50", kind="tool:calculator") == Urgency.CRITICAL


# ---------------------------------------------------------------------------
# CognitiveSidecar dispatch/scheduling
# ---------------------------------------------------------------------------


class _FakeService:
    """A cognitive service with a scriptable delay/outcome, for testing the sidecar."""

    def __init__(self, name: str, delay: float = 0.0, result: CognitiveResult | None = None, raises: bool = False):
        self.name = name
        self.delay = delay
        self.result = result or CognitiveResult(text="ok", confidence=ConfidenceScore())
        self.raises = raises
        self.calls = 0

    async def handle(self, request: CognitiveRequest) -> CognitiveResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise RuntimeError("simulated failure")
        return self.result


def test_sidecar_delivers_successful_result():
    async def run():
        sidecar = CognitiveSidecar()
        service = _FakeService("rag")
        sidecar.register(service)

        results = []

        async def on_result(handle, result):
            results.append(result)

        handle = sidecar.dispatch("conv-1", "rag", CognitiveRequest(query="hi"), on_result)
        await asyncio.sleep(0)  # let the dispatched task start
        # Wait for the background task to actually complete.
        for _ in range(50):
            if results:
                break
            await asyncio.sleep(0.01)

        assert len(results) == 1
        assert results[0] is not None
        assert results[0].text == "ok"
        assert handle.kind == "rag"

    asyncio.run(run())


def test_sidecar_drops_stale_result_after_turn_advances():
    async def run():
        registry = TaskRegistry()
        sidecar = CognitiveSidecar(registry=registry)
        service = _FakeService("rag", delay=0.05)
        sidecar.register(service)

        results = []

        async def on_result(handle, result):
            results.append(result)

        sidecar.dispatch("conv-1", "rag", CognitiveRequest(query="hi"), on_result)
        # Conversation moves on before the slow service finishes.
        registry.advance_turn("conv-1")

        for _ in range(50):
            if results:
                break
            await asyncio.sleep(0.01)

        assert len(results) == 1
        assert results[0] is None  # discarded as stale, not applied

    asyncio.run(run())


def test_sidecar_reports_none_on_timeout_and_never_raises():
    async def run():
        sidecar = CognitiveSidecar()
        service = _FakeService("rag", delay=10.0)  # far longer than the default NORMAL deadline
        sidecar.register(service)

        results = []

        async def on_result(handle, result):
            results.append(result)

        sidecar.dispatch(
            "conv-1", "rag", CognitiveRequest(query="hi", urgency=Urgency.NORMAL, deadline=0.02), on_result
        )

        for _ in range(50):
            if results:
                break
            await asyncio.sleep(0.01)

        assert results == [None]

    asyncio.run(run())


def test_sidecar_circuit_breaker_opens_after_repeated_failures():
    async def run():
        sidecar = CognitiveSidecar()
        service = _FakeService("flaky", raises=True)
        sidecar.register(service)

        results = []

        async def on_result(handle, result):
            results.append(result)

        # Breaker threshold is 3 consecutive failures.
        for _ in range(3):
            sidecar.dispatch("conv-1", "flaky", CognitiveRequest(query="x"), on_result)
            for _ in range(50):
                if len(results) > len(results) - 1 and service.calls >= 1:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.01)

        calls_before_breaker_opens = service.calls
        assert calls_before_breaker_opens == 3

        # Fourth dispatch should be short-circuited: service.handle is not called again.
        sidecar.dispatch("conv-1", "flaky", CognitiveRequest(query="x"), on_result)
        await asyncio.sleep(0.05)

        assert service.calls == calls_before_breaker_opens  # breaker skipped the call

    asyncio.run(run())


def test_sidecar_dispatch_to_unknown_service_reports_none():
    async def run():
        sidecar = CognitiveSidecar()
        results = []

        async def on_result(handle, result):
            results.append(result)

        sidecar.dispatch("conv-1", "does-not-exist", CognitiveRequest(query="x"), on_result)
        await asyncio.sleep(0.05)

        assert results == [None]

    asyncio.run(run())
