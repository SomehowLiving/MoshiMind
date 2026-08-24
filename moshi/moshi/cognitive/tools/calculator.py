# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A real, minimal ``Urgency.CRITICAL`` cognitive service (PHASES.md Phase 2).

Arithmetic is the canonical example of a request that should get an immediate
answer rather than the async, non-blocking treatment RAG gets — see
``cognitive.urgency.classify_urgency``. This exists to prove the sidecar's
urgency-based scheduling against a real (if intentionally tiny) producer
instead of only a mock, and to give ``CognitiveService`` a second concrete
implementer alongside RAG so the interface is proven to generalize.

Evaluation is a hand-rolled AST walk over a whitelist of numeric operators —
not ``eval()`` — since the input is arbitrary text derived from user speech.
"""

from __future__ import annotations

import ast
import operator

from .. import confidence as _confidence
from ..sidecar import CognitiveRequest, CognitiveResult

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when the input contains anything beyond numbers and arithmetic operators."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpressionError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise UnsafeExpressionError(f"disallowed expression element: {type(node).__name__}")


def evaluate_arithmetic(expression: str) -> float:
    """Evaluate a plain arithmetic expression (+ - * / // % **, parentheses, unary -).

    Raises ``UnsafeExpressionError`` for anything else (names, calls, comprehensions,
    attribute access, ...) rather than silently coercing or executing it.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"not a valid expression: {e}") from e
    return _eval_node(tree.body)


class CalculatorService:
    """``CognitiveService`` wrapper around ``evaluate_arithmetic``."""

    name = "calculator"

    async def handle(self, request: CognitiveRequest) -> CognitiveResult:
        value = evaluate_arithmetic(request.query)
        text = str(int(value)) if float(value).is_integer() else str(value)
        return CognitiveResult(text=text, confidence=_confidence.ConfidenceScore(), source=self.name)
