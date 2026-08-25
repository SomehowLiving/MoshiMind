# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Asynchronous cognitive layer around the real-time Moshi core.

Everything in this package is deliberately independent of torch/CUDA: it
schedules and scores work, it does not run models. The one exception is the
conditioning-strength application in ``moshi.inference_utils.channel``, which
consumes ``ConfidenceScore.strength()`` from here but does the actual tensor
math where the tensors already live.

See ``PHASES.md`` at the repo root for the roadmap this package implements.
"""

from .confidence import ConfidenceScore
from .urgency import Urgency, classify_urgency
from .task_registry import TaskRegistry, TaskHandle
from .sidecar import CognitiveSidecar, CognitiveRequest, CognitiveResult, CognitiveService
from .speculation import SpeculativeSlot
from .predictive_trigger import PredictiveTrigger
from .multishot import MultiShotGate
from .telemetry import RetrievalEvent, RetrievalTelemetry
from .rag_service import RAGCognitiveService
from .tools.calculator import CalculatorService, evaluate_arithmetic, UnsafeExpressionError
from .merge import KnowledgeCandidate, merge_candidates, resolve_reference_conditioning
from .memory import MemoryStore, EpisodicRecord, SemanticFact, WorkingMemory, score_salience, extract_facts, MemoryCognitiveService
from .conversation_state import (
    SpeakingState,
    InterruptionKind,
    InterruptionDecision,
    ConversationState,
    decide_interruption_response,
)
from .interruption import InterruptionClassifier

__all__ = [
    "ConfidenceScore",
    "Urgency",
    "classify_urgency",
    "TaskRegistry",
    "TaskHandle",
    "CognitiveSidecar",
    "CognitiveRequest",
    "CognitiveResult",
    "CognitiveService",
    "SpeculativeSlot",
    "PredictiveTrigger",
    "MultiShotGate",
    "RetrievalEvent",
    "RetrievalTelemetry",
    "RAGCognitiveService",
    "CalculatorService",
    "evaluate_arithmetic",
    "UnsafeExpressionError",
    "KnowledgeCandidate",
    "merge_candidates",
    "resolve_reference_conditioning",
    "MemoryStore",
    "EpisodicRecord",
    "SemanticFact",
    "WorkingMemory",
    "score_salience",
    "extract_facts",
    "MemoryCognitiveService",
    "SpeakingState",
    "InterruptionKind",
    "InterruptionDecision",
    "ConversationState",
    "decide_interruption_response",
    "InterruptionClassifier",
]
