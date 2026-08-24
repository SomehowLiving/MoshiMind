# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Long-term conversational memory (PHASES.md Phase 3).

Three layers, per the roadmap:

- ``working_memory.WorkingMemory`` — seconds-to-minutes, the current conversation.
- ``store.MemoryStore`` (episodic side) — days-to-months, salient past turns.
- ``store.MemoryStore`` (semantic side) — persistent facts/preferences.

Kept structurally separate from RAG (``cognitive.rag_service``) even though
both ultimately compete for the same Moshi conditioning channel — see
``memory_service.MemoryCognitiveService`` and ``cognitive.merge`` for how
that competition is arbitrated.
"""

from .store import MemoryStore, EpisodicRecord, SemanticFact
from .working_memory import WorkingMemory
from .salience import score_salience
from .semantic_extraction import extract_facts
from .memory_service import MemoryCognitiveService

__all__ = [
    "MemoryStore",
    "EpisodicRecord",
    "SemanticFact",
    "WorkingMemory",
    "score_salience",
    "extract_facts",
    "MemoryCognitiveService",
]
