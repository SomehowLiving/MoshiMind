# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete ``CognitiveService`` producers (PHASES.md Phase 2), beyond RAG.

Deliberately small: the point of this session's work is proving the sidecar
abstraction generalizes, not building out a tool ecosystem (that's explicitly
deprioritized — see PHASES.md Phase 10, "generic voice-agent plumbing").
"""

from .calculator import CalculatorService

__all__ = ["CalculatorService"]
