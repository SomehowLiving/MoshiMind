# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Structured retrieval log for the Phase 9 benchmark suite (PHASES.md Phase 1.4).

Every retrieval-LLM call — confirmed (``rag_token_id``) or speculative
(``PredictiveTrigger``) — gets one entry: query, reference text, confidence,
latency. This is deliberately scoped to the retrieval step itself: whether the
resulting conditioning bias actually made it into generation is a separate,
later decision (it can still be skipped for falling below
``--rag-min-conditioning-strength``, or dropped for being stale — see
``cognitive.confidence`` and ``channel.py::_async_update_reference``) and is
already tracked by the existing ``rag_low_confidence_skipped_total`` /
``rag_stale_reference_dropped_total`` counters in ``server.Metrics``.
``applied`` here means only "did the retrieval step itself produce a usable
candidate" — not "did it end up biasing generation".
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field


@dataclass
class RetrievalEvent:
    event_id: int
    query: str
    reference_text: str
    confidence: float
    latency_seconds: float
    applied: bool
    kind: str = "confirmed"
    """"confirmed" (triggered by rag_token_id) or "speculative" (PredictiveTrigger)."""


@dataclass
class RetrievalTelemetry:
    """Bounded, in-memory log of retrieval attempts; replace with a real sink
    (a file, a metrics backend) once Phase 9's benchmark suite needs one."""

    max_events: int = 1000
    _events: list[RetrievalEvent] = field(default_factory=list)
    _id_counter: "itertools.count" = field(default_factory=lambda: itertools.count(1))

    def record(
        self,
        query: str,
        reference_text: str,
        confidence: float,
        latency_seconds: float,
        kind: str = "confirmed",
    ) -> RetrievalEvent:
        event = RetrievalEvent(
            event_id=next(self._id_counter),
            query=query,
            reference_text=reference_text,
            confidence=confidence,
            latency_seconds=latency_seconds,
            applied=bool(reference_text) and confidence > 0,
            kind=kind,
        )
        self._events.append(event)
        if len(self._events) > self.max_events:
            del self._events[: len(self._events) - self.max_events]
        return event

    def snapshot(self) -> list[dict]:
        return [asdict(e) for e in self._events]

    def clear(self) -> None:
        self._events.clear()
