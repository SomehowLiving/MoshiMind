# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``RAGManager`` as a ``CognitiveService`` (PHASES.md Phase 2.1).

``channel.py`` does not use this yet: RAG's live wiring has bespoke behavior
(the ``wait_steps`` STT delay, speculative-attempt reuse, the multi-shot
budget — see ``inference_utils/{channel,rag_manager}.py``) that the generic
sidecar doesn't replicate, and rewiring a working, tested path onto the
sidecar for its own sake would be churn without a behavior change. This
adapter exists to prove ``CognitiveService`` actually fits RAG structurally —
required before memory/tools can be said to share "one interface" with it —
and is the on-ramp for a future channel.py that dispatches RAG through the
sidecar alongside other services instead of calling ``RAGManager`` directly.

Depends only on a duck-typed ``_ReferenceGeneratorLike`` (anything exposing an
async ``get_reference_text``), not the concrete ``RAGManager`` class, so this
stays importable without torch/openai — matching every other module in
``cognitive/``.
"""

from __future__ import annotations

from typing import Awaitable, Protocol

from .confidence import ConfidenceScore
from .sidecar import CognitiveRequest, CognitiveResult


class _ReferenceGeneratorLike(Protocol):
    def get_reference_text(
        self, context: str, kind: str = "confirmed"
    ) -> Awaitable[tuple[str, str, float, str, ConfidenceScore]]: ...


class RAGCognitiveService:
    """Adapts ``RAGManager.get_reference_text`` to the ``CognitiveService`` interface."""

    name = "rag"

    def __init__(self, reference_generator: _ReferenceGeneratorLike, kind: str = "confirmed"):
        self._reference_generator = reference_generator
        self._kind = kind

    async def handle(self, request: CognitiveRequest) -> CognitiveResult:
        _, reference_text, _, lm_label, confidence = await self._reference_generator.get_reference_text(
            request.context or request.query, kind=self._kind
        )
        return CognitiveResult(text=reference_text, confidence=confidence, source=lm_label or self.name)
