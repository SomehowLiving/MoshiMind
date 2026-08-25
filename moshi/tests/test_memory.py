# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Phase 3 (long-term conversational memory).

Torch-free and GPU-free, same as test_cognitive.py -- these exercise the
storage, extraction, and arbitration logic in isolation. sqlite3 is real
stdlib execution (not mocked), so MemoryStore tests are genuine integration
tests of the persistence layer, just against an in-memory database.
"""

from __future__ import annotations

import asyncio

import pytest

from moshi.cognitive import (
    CognitiveRequest,
    ConfidenceScore,
    KnowledgeCandidate,
    MemoryCognitiveService,
    MemoryStore,
    WorkingMemory,
    extract_facts,
    merge_candidates,
    resolve_reference_conditioning,
    score_salience,
)


# ---------------------------------------------------------------------------
# WorkingMemory (Phase 3.1)
# ---------------------------------------------------------------------------


def test_working_memory_accumulates_context():
    wm = WorkingMemory()
    wm.add_turn("user", "hello")
    wm.add_turn("model", "hi there")
    assert wm.get_context() == "user: hello\nmodel: hi there"


def test_working_memory_evicts_by_turn_count():
    wm = WorkingMemory(max_turns=2, max_chars=10_000)
    wm.add_turn("user", "one")
    wm.add_turn("model", "two")
    wm.add_turn("user", "three")
    assert len(wm) == 2
    assert wm.turns() == [("model", "two"), ("user", "three")]


def test_working_memory_evicts_by_char_budget():
    wm = WorkingMemory(max_turns=100, max_chars=15)
    wm.add_turn("user", "0123456789")  # 10 chars
    wm.add_turn("user", "abcdefghij")  # 10 more -> 20 > 15, must evict oldest
    assert wm.turns() == [("user", "abcdefghij")]


def test_working_memory_always_keeps_at_least_one_turn_even_if_over_budget():
    wm = WorkingMemory(max_turns=100, max_chars=5)
    wm.add_turn("user", "this single turn is already longer than the budget")
    assert len(wm) == 1  # never evicted down to zero


def test_working_memory_reset_clears_everything():
    wm = WorkingMemory()
    wm.add_turn("user", "hi")
    wm.reset()
    assert len(wm) == 0
    assert wm.get_context() == ""


# ---------------------------------------------------------------------------
# score_salience (Phase 3.2)
# ---------------------------------------------------------------------------


def test_salience_explicit_remember_request_is_maximal():
    assert score_salience("remember that I hate cilantro") == 1.0
    assert score_salience("don't forget I have a meeting at 5") == 1.0


def test_salience_long_turn_scores_moderately():
    text = "I've been working on a really big project for the last three months and it's finally coming together"
    assert len(text.split()) >= 15
    assert score_salience(text) == 0.6


def test_salience_bare_question_scores_low():
    assert score_salience("what time is it?") == 0.2


def test_salience_short_statement_scores_baseline():
    assert score_salience("sounds good") == 0.3


def test_salience_empty_text_is_zero():
    assert score_salience("") == 0.0
    assert score_salience("   ") == 0.0


# ---------------------------------------------------------------------------
# extract_facts (Phase 3.3)
# ---------------------------------------------------------------------------


def test_extract_facts_name():
    assert extract_facts("my name is Nidhi") == [("name", "Nidhi")]


def test_extract_facts_multiple_in_one_utterance():
    facts = extract_facts("my name is Nidhi and I work as an engineer")
    assert ("name", "Nidhi") in facts
    assert ("occupation", "engineer") in facts


def test_extract_facts_stops_at_conjunction_not_greedy_across_clauses():
    facts = extract_facts("my name is Nidhi and I live in Bangalore")
    names = [v for k, v in facts if k == "name"]
    assert names == ["Nidhi"]  # not "Nidhi and I live in Bangalore"


def test_extract_facts_stops_at_comma():
    facts = extract_facts("I live in San Francisco, but I travel a lot")
    assert ("location", "San Francisco") in facts


def test_extract_facts_preference():
    assert extract_facts("I like coffee") == [("preference", "coffee")]


def test_extract_facts_no_match_returns_empty():
    assert extract_facts("what's the weather like today") == []


def test_extract_facts_empty_input():
    assert extract_facts("") == []
    assert extract_facts(None) == []


# ---------------------------------------------------------------------------
# MemoryStore (Phase 3.2/3.3, real sqlite3 execution)
# ---------------------------------------------------------------------------


def test_store_upsert_and_get_fact():
    store = MemoryStore()
    store.upsert_fact("u1", "name", "Nidhi", confidence=0.9, now=100.0)
    facts = store.get_facts("u1")
    assert len(facts) == 1
    assert facts[0].key == "name"
    assert facts[0].value == "Nidhi"


def test_store_upsert_same_key_overwrites_not_duplicates():
    store = MemoryStore()
    store.upsert_fact("u1", "name", "Nidhi", confidence=0.9, now=100.0)
    store.upsert_fact("u1", "name", "N.", confidence=0.95, now=200.0)
    facts = store.get_facts("u1")
    assert len(facts) == 1
    assert facts[0].value == "N."
    assert facts[0].confidence == 0.95


def test_store_facts_are_isolated_per_user():
    store = MemoryStore()
    store.upsert_fact("u1", "name", "Alice", confidence=0.9, now=0.0)
    store.upsert_fact("u2", "name", "Bob", confidence=0.9, now=0.0)
    assert [f.value for f in store.get_facts("u1")] == ["Alice"]
    assert [f.value for f in store.get_facts("u2")] == ["Bob"]


def test_store_get_fact_single_lookup():
    store = MemoryStore()
    store.upsert_fact("u1", "occupation", "engineer", confidence=0.8, now=0.0)
    fact = store.get_fact("u1", "occupation")
    assert fact is not None
    assert fact.value == "engineer"
    assert store.get_fact("u1", "nonexistent") is None


def test_store_add_and_search_episodes():
    store = MemoryStore()
    store.add_episode("u1", turn_index=1, text="I'm building MoshiRAG", salience=0.6, now=0.0)
    store.add_episode("u1", turn_index=2, text="the weather is nice", salience=0.3, now=1.0)
    results = store.search_episodes("u1", "MoshiRAG")
    assert len(results) == 1
    assert "MoshiRAG" in results[0].text


def test_store_search_episodes_empty_keyword_returns_nothing():
    store = MemoryStore()
    store.add_episode("u1", 1, "some text", 0.5, 0.0)
    assert store.search_episodes("u1", "") == []


def test_store_recent_episodes_respects_limit_and_order():
    store = MemoryStore()
    for i in range(5):
        store.add_episode("u1", i, f"turn {i}", 0.5, now=float(i))
    recent = store.recent_episodes("u1", limit=2)
    assert [e.text for e in recent] == ["turn 4", "turn 3"]  # newest first


def test_store_persists_across_close_and_reopen(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = MemoryStore(db_path)
    store.upsert_fact("u1", "name", "Nidhi", confidence=0.9, now=0.0)
    store.close()

    reopened = MemoryStore(db_path)
    facts = reopened.get_facts("u1")
    assert len(facts) == 1
    assert facts[0].value == "Nidhi"
    reopened.close()


# ---------------------------------------------------------------------------
# MemoryCognitiveService (Phase 3.4)
# ---------------------------------------------------------------------------


def test_memory_service_returns_empty_when_nothing_known():
    async def run():
        store = MemoryStore()
        service = MemoryCognitiveService(store)
        result = await service.handle(CognitiveRequest(query="anything", context="u1"))
        assert result.text == ""
        assert result.confidence.strength() == 0.0

    asyncio.run(run())


def test_memory_service_returns_known_facts():
    async def run():
        store = MemoryStore()
        store.upsert_fact("u1", "name", "Nidhi", confidence=0.9, now=0.0)
        service = MemoryCognitiveService(store)
        result = await service.handle(CognitiveRequest(query="", context="u1"))
        assert "name=Nidhi" in result.text
        assert result.confidence.strength() > 0

    asyncio.run(run())


def test_memory_service_includes_matching_episodes():
    async def run():
        store = MemoryStore()
        store.add_episode("u1", 1, "I'm building MoshiRAG", 0.6, now=0.0)
        service = MemoryCognitiveService(store)
        result = await service.handle(CognitiveRequest(query="MoshiRAG", context="u1"))
        assert "MoshiRAG" in result.text

    asyncio.run(run())


def test_memory_service_uses_custom_user_id_resolver():
    async def run():
        store = MemoryStore()
        store.upsert_fact("real-user-id", "name", "Nidhi", confidence=0.9, now=0.0)
        service = MemoryCognitiveService(store, user_id_resolver=lambda req: "real-user-id")
        result = await service.handle(CognitiveRequest(query="", context="some-other-conversation-id"))
        assert "name=Nidhi" in result.text

    asyncio.run(run())


def test_memory_service_isolates_users_via_default_resolver():
    async def run():
        store = MemoryStore()
        store.upsert_fact("conv-1", "name", "Alice", confidence=0.9, now=0.0)
        service = MemoryCognitiveService(store)
        result = await service.handle(CognitiveRequest(query="", context="conv-2"))
        assert result.text == ""  # conv-2 has no facts of its own

    asyncio.run(run())


# ---------------------------------------------------------------------------
# merge_candidates (Phase 3.4 arbitration)
# ---------------------------------------------------------------------------


def test_merge_empty_list_returns_none():
    assert merge_candidates([]) is None


def test_merge_all_unusable_returns_none():
    candidates = [
        KnowledgeCandidate("rag", "", ConfidenceScore()),
        KnowledgeCandidate("memory", "text", ConfidenceScore.empty()),
    ]
    assert merge_candidates(candidates) is None


def test_merge_single_usable_candidate_returned_as_is():
    candidate = KnowledgeCandidate("rag", "Jensen Huang is NVIDIA's CEO", ConfidenceScore())
    result = merge_candidates([candidate])
    assert result == candidate


def test_merge_combines_two_usable_candidates():
    rag = KnowledgeCandidate("rag", "NVIDIA's CEO is Jensen Huang.", ConfidenceScore(0.9, 0.9, 0.9))
    memory = KnowledgeCandidate("memory", "Known about the user: occupation=engineer", ConfidenceScore(0.6, 0.9, 1.0))
    merged = merge_candidates([rag, memory])
    assert "Jensen Huang" in merged.text
    assert "engineer" in merged.text
    assert merged.source == "rag+memory"


def test_merge_confidence_is_the_strongest_candidates_not_an_average():
    strong = KnowledgeCandidate("rag", "strong fact", ConfidenceScore(1.0, 1.0, 1.0))
    weak = KnowledgeCandidate("memory", "weak scrap", ConfidenceScore(0.1, 0.1, 0.1))
    merged = merge_candidates([strong, weak])
    assert merged.confidence.strength() == pytest.approx(strong.confidence.strength())


def test_merge_respects_char_budget_dropping_lower_ranked_candidates():
    strong = KnowledgeCandidate("rag", "x" * 500, ConfidenceScore(1.0, 1.0, 1.0))
    weak = KnowledgeCandidate("memory", "y" * 500, ConfidenceScore(0.5, 0.5, 0.5))
    merged = merge_candidates([strong, weak], max_chars=600)
    assert "x" * 500 in merged.text
    assert "y" * 500 not in merged.text  # didn't fit the budget after the stronger one
    assert merged.source == "rag"  # only the fitting candidate contributes to the source label


def test_merge_ranks_by_strength_not_input_order():
    weak = KnowledgeCandidate("memory", "weak", ConfidenceScore(0.1, 0.1, 0.1))
    strong = KnowledgeCandidate("rag", "strong", ConfidenceScore(1.0, 1.0, 1.0))
    merged = merge_candidates([weak, strong], max_chars=1000)
    assert merged.text == "strong weak"  # strongest first regardless of input order


# ---------------------------------------------------------------------------
# resolve_reference_conditioning (the exact decision channel.py's live wiring
# makes once RAG's own retrieval completes and a memory candidate is in hand)
# ---------------------------------------------------------------------------


def test_resolve_rag_only_no_memory_candidate():
    text, label, confidence = resolve_reference_conditioning(
        "NVIDIA's CEO is Jensen Huang.", "gpt-4", ConfidenceScore(0.9, 0.9, 0.9), None
    )
    assert text == "NVIDIA's CEO is Jensen Huang."
    assert label == "gpt-4"
    assert confidence.strength() > 0


def test_resolve_merges_rag_and_memory():
    memory = KnowledgeCandidate("memory", "Known about the user: name=Nidhi", ConfidenceScore(0.8, 0.9, 1.0))
    text, label, confidence = resolve_reference_conditioning(
        "NVIDIA's CEO is Jensen Huang.", "gpt-4", ConfidenceScore(0.9, 0.9, 0.9), memory
    )
    assert "Jensen Huang" in text
    assert "Nidhi" in text


def test_resolve_falls_back_to_memory_when_rag_produced_nothing():
    """If the retrieval LLM times out/fails but memory has something, the user still
    gets grounded (in what's known about them) instead of nothing at all."""
    memory = KnowledgeCandidate("memory", "Known about the user: occupation=engineer", ConfidenceScore(0.8, 0.9, 1.0))
    text, label, confidence = resolve_reference_conditioning("", "", ConfidenceScore.empty(), memory)
    assert "engineer" in text
    assert confidence.strength() > 0


def test_resolve_rag_failure_with_no_memory_falls_through_to_ret_failed_case():
    """Empty RAG result and no memory candidate must round-trip as empty text with
    empty confidence -- that's what channel.py uses to decide to show [RET_FAILED]."""
    text, label, confidence = resolve_reference_conditioning("", "", ConfidenceScore.empty(), None)
    assert text == ""
    assert confidence.strength() == 0.0


def test_resolve_rag_failure_with_unusable_memory_also_falls_through():
    unusable_memory = KnowledgeCandidate("memory", "", ConfidenceScore.empty())
    text, label, confidence = resolve_reference_conditioning("", "", ConfidenceScore.empty(), unusable_memory)
    assert text == ""
    assert confidence.strength() == 0.0


def test_resolve_preserves_lm_label_when_merge_happens():
    memory = KnowledgeCandidate("memory", "some fact", ConfidenceScore(0.5, 0.9, 1.0))
    _, label, _ = resolve_reference_conditioning("a fact", "claude-3", ConfidenceScore(0.9, 0.9, 0.9), memory)
    assert label == "claude-3"  # the LLM display name, not the merged source string