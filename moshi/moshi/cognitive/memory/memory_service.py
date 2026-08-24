# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Memory as a second ``CognitiveService`` competing with RAG (PHASES.md Phase 3.4).

RAG answers "what does the world know"; this answers "what does this user/
conversation know" (see PHASES.md's architecture split). Structurally
identical to ``cognitive.rag_service.RAGCognitiveService`` — same interface,
same sidecar — so the two can be dispatched side by side and arbitrated by
``cognitive.merge.merge_candidates`` using the same ``ConfidenceScore``
currency, instead of memory needing its own bespoke injection path.
"""

from __future__ import annotations

from typing import Callable

from ..confidence import ConfidenceScore
from ..sidecar import CognitiveRequest, CognitiveResult
from .store import MemoryStore


class MemoryCognitiveService:
    """Recalls known facts and relevant past episodes for a user.

    ``user_id_resolver`` maps a ``CognitiveRequest`` to a memory user id;
    defaults to using ``request.context`` verbatim (typically the caller's
    conversation/session id, in the absence of any real identity system —
    see ``cognitive.memory.store`` for why that's a known limitation, not an
    oversight).
    """

    name = "memory"

    def __init__(
        self,
        store: MemoryStore,
        episode_limit: int = 3,
        user_id_resolver: Callable[[CognitiveRequest], str] | None = None,
    ):
        self.store = store
        self.episode_limit = episode_limit
        self._user_id_resolver = user_id_resolver

    def _resolve_user_id(self, request: CognitiveRequest) -> str:
        if self._user_id_resolver is not None:
            return self._user_id_resolver(request)
        return request.context or "default"

    async def handle(self, request: CognitiveRequest) -> CognitiveResult:
        # sqlite3 is synchronous; these calls are local-disk-fast in practice, but a
        # future version serving many concurrent channels should run them via
        # loop.run_in_executor rather than blocking the event loop directly.
        user_id = self._resolve_user_id(request)
        facts = self.store.get_facts(user_id)
        episodes = self.store.search_episodes(user_id, request.query, limit=self.episode_limit) if request.query else []

        if not facts and not episodes:
            return CognitiveResult(text="", confidence=ConfidenceScore.empty(), source=self.name)

        text = self._format(facts, episodes)
        # Facts are asserted (upserted) with explicit confidence at write time; episodes
        # matched by keyword search carry less certainty about relevance than a real
        # semantic search would. Freshness is left at 1.0: no staleness model for facts yet.
        relevance = 0.85 if episodes else 0.5
        confidence = ConfidenceScore(relevance=relevance, confidence=0.9, freshness=1.0)
        return CognitiveResult(text=text, confidence=confidence, source=self.name)

    def _format(self, facts, episodes) -> str:
        parts = []
        if facts:
            parts.append("Known about the user: " + "; ".join(f"{f.key}={f.value}" for f in facts))
        if episodes:
            parts.append("Relevant past conversation: " + " | ".join(e.text for e in episodes))
        return " ".join(parts)
