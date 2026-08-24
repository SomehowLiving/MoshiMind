# MoshiMind — Phased Roadmap

**Thesis:** don't turn Moshi into another STT→LLM→TTS voice-agent framework. Moshi's
unique asset is a native full-duplex speech-to-speech model with a real-time generation
loop (`moshi/moshi/models/lm.py`) and an existing, working async side-channel for
injecting external signal into that loop without blocking it (`rag_token_id` →
`RAGManager` → ARC encoder → `streaming_sum` conditioning — see `moshi/moshi/inference_utils/{channel,rag_manager}.py`
and `moshi/moshi/models/lm.py:404-410,729-765`). Every phase below either widens that
one mechanism (more producers of conditioning, better-scored conditioning, predictive
triggering) or protects it (task identity, fault isolation, benchmarking regressions).
Nothing here reaches for LiveKit/Whisper/a generic agent framework unless the phase
explicitly says the boundary is infrastructure, not cognition.

Status legend: 🔲 not started · 🟨 in progress · ✅ done · 🧪 has tests

---

## Phase 0 — Foundations (prerequisite for everything else)

Already partially landed in the reliability pass (`channel.py`, `rag_manager.py`,
`server.py` — see git history). This phase generalizes those one-off fixes into
reusable infrastructure the later phases build on.

| # | Task | Files | Status |
|---|---|---|---|
| 0.1 | **Task identity system** — every async cognitive op (retrieval, memory lookup, tool call) gets `(conversation_id, turn_id, task_id)`; a result for a superseded turn is dropped, not applied | `moshi/moshi/cognitive/task_registry.py` (new) | ✅ 🧪 |
| 0.2 | **Cognitive sidecar scaffold** — a single async orchestrator that RAG/memory/tools register into, so `channel.py` talks to one interface instead of hand-wiring each service | `moshi/moshi/cognitive/sidecar.py` (new) | ✅ 🧪 |
| 0.3 | **Confidence/urgency data model** — every knowledge candidate carries `relevance/confidence/freshness`; every cognitive request carries an `urgency` class | `moshi/moshi/cognitive/confidence.py`, `urgency.py` (new) | ✅ 🧪 |
| 0.4 | Fault isolation for existing RAG path (ARC-encoder failure doesn't kill the session) | `channel.py` | ✅ (prior commit) |
| 0.5 | Observability counters for the retrieval path | `server.py` (`Metrics`, `/api/metrics`) | ✅ (prior commit) |

**Acceptance:** unit tests for 0.1/0.3 run without a GPU or torch (pure Python state
machines); 0.2 is exercised with a fake async producer in tests. This phase is the one
being implemented in this session — see "Implementation log" at the bottom.

---

## Phase 1 — 🔴 Native Moshi + RAG integration (the core bet)

**Research question:** *can Moshi dynamically acquire knowledge during generation
without a conventional blocking retrieval pipeline* — and can we make the *quality* of
that injection a first-class, measurable variable instead of a boolean "reference
text present or not"?

| # | Task | Approach | Files |
|---|---|---|---|
| 1.1 ✅🧪 | Confidence-aware conditioning strength | Scale the `streaming_sum` tensor by `f(relevance, confidence, freshness)` before injection, instead of injecting at fixed strength whenever retrieval succeeds | `cognitive/confidence.py`, `channel.py::_async_update_reference`, `inference_job.py` |
| 1.2 ✅🧪 | Predictive/speculative retrieval | Start a retrieval as soon as the transcript looks like it's heading toward a knowledge need (heuristic first: named entity / question-word detector on the streaming ASR transcript; embedding-similarity trigger later), racing it against the model's own `rag_token_id` emission — first one to a usable result wins, the other is cancelled via the task registry | `cognitive/predictive_trigger.py`, `cognitive/speculation.py`, `rag_manager.py`, `channel.py` |
| 1.3 | Dynamic (multi-shot) conditioning | Today one reference → one conditioning tensor → whole response. Extend `RAGManager` to accept a follow-up retrieval mid-response (second `rag_token_id`, or the predictive trigger firing again) and push a second tensor through the *existing* per-slot streaming update path — the plumbing (`update_streaming_sum_tensors` accepts a fresh tensor per slot at any time) already supports this; what's missing is deciding *when* a second retrieval is worth the interruption | `rag_manager.py`, `cognitive/sidecar.py` |
| 1.4 | Retrieval quality benchmark hook | Every retrieval logs `(query, reference_text, confidence, latency, applied: bool)` to a structured log the Phase 9 benchmark suite can replay | `cognitive/telemetry.py` |

**Acceptance:** 1.1 is testable today with mocked tensors (no GPU) — assert the scaled
tensor's norm tracks the confidence score. 1.2/1.3 need a running server + LLM +
ARC encoder to validate against real conversations — **BLOCKED in this environment**,
written and unit-tested against fakes, flagged for integration testing on GPU hardware.

**1.1 status:** implemented. `RAGManager.get_reference_text` now scores every
retrieval-LLM reference with `ConfidenceScore.heuristic_from_llm_reference` (empty
answer → 0, rushed-under-timeout and very-short answers penalized) and threads it
through `_handle_reference_text` → `_async_update_reference` in both `channel.py`
(live server) and `inference_job.py` (offline batch eval). Below
`--rag-min-conditioning-strength` (default 0.15) the ARC-encoder call is skipped
entirely; otherwise the returned tensor is scaled by `strength()` before
`update_streaming_sum_tensors`. The heuristic itself has no query access yet (no
real relevance signal) — real embedding-similarity relevance scoring is the next
increment, not this one. Tensor-scaling path is BLOCKED for execution here (no
torch/GPU); the scoring function itself has 10 new unit tests.

**1.2 status:** implemented. `cognitive/predictive_trigger.py::PredictiveTrigger`
watches the partial ASR transcript for the current user turn (question words,
trailing "?", a handful of info-request phrases) and fires at most once per turn.
`channel.py::_stt_recv_loop` calls it after every `STTWordMessage` and, on a hit,
calls the new `RAGManager.trigger_speculative` — which no-ops if a confirmed
retrieval is already running (a guess never preempts real work) and otherwise
starts the same `get_reference_text` call the confirmed path uses, applying its
result early at a discounted confidence (`ConfidenceScore.scaled`, 0.6x). The
reuse half of the mechanism lives in `cognitive/speculation.py::SpeculativeSlot`:
when the model's own `rag_token_id` later fires, `RAGManager.trigger` adopts the
speculative attempt (awaiting it if still running) instead of starting a second,
redundant retrieval, and re-applies it at full (undiscounted) confidence. A new
user turn (detected via the turn-manager's speaker-switch marker) clears any
stale, never-confirmed speculative attempt so it can't block the next guess.
16 new unit tests (`PredictiveTrigger`, `SpeculativeSlot`) — all torch-free, all
passing (40/40 total in `test_cognitive.py`). `rag_manager.py`/`channel.py`
wiring itself is BLOCKED for execution here (needs torch + openai + a running
server), same as the rest of the RAG path.

---

## Phase 2 — 🔴 Streaming / asynchronous cognition (the sidecar, generalized)

Phase 0.2 built the skeleton. This phase populates it with real producers beyond RAG
and enforces the non-blocking guarantee everywhere.

| # | Task |
|---|---|
| 2.1 | Formalize the "cognitive service" interface (`async def handle(request) -> CognitiveResult`) so RAG, memory, and tools are structurally identical to the sidecar |
| 2.2 | Urgency-based scheduling: `critical` requests get a short deadline and may block a few hundred ms (e.g. arithmetic); `background` requests never block, results arrive whenever and are applied only if still relevant (task registry handles staleness) |
| 2.3 | Circuit breaker per cognitive service — N consecutive failures suspends that service for a cooldown window instead of retrying into a dead endpoint every turn |

**Acceptance:** unit tests with fake slow/failing services proving (a) `critical`
requests respect their deadline, (b) `background` requests never block the caller,
(c) the breaker opens/half-opens/closes correctly. All torch-free.

---

## Phase 3 — 🔴 Long-term conversational memory

Kept structurally separate from RAG (`RAG` = what does the world know, `memory` =
what does *this* user/conversation know) even though both ultimately compete for the
same conditioning channel into Moshi.

| # | Task |
|---|---|
| 3.1 | Working memory — already exists as `turn_manager.get_context()`; formalize its window and eviction policy |
| 3.2 | Episodic memory — persist salient turns (heuristic: turns near a topic change, an emotional peak, or an explicit "remember this") to a per-user store keyed off session id |
| 3.3 | Semantic memory — durable facts/preferences extracted from episodic memory on a slower cadence (batch job, not per-turn) |
| 3.4 | Memory retrieval as a second cognitive-sidecar producer, competing with RAG for conditioning bandwidth — needs a merge policy (Phase 1.1's confidence score is the natural arbitration signal) |

**Acceptance:** memory store and retrieval logic are pure Python/SQLite — fully
testable without a GPU. Integration with the conditioning channel is BLOCKED here,
same as Phase 1.

---

## Phase 4 — 🔴 Full-duplex interruption / anticipation

This is the one area where the plan explicitly says *don't* let infrastructure
(LiveKit) own the interesting part. LiveKit/WebRTC-level VAD and endpointing are fine
for transport; conversational-state decisions (is this a real interruption, is the
user hesitating, should Moshi yield) belong in the Moshi-facing layer.

| # | Task |
|---|---|
| 4.1 | Formalize `ConversationState` (turn_id, speaking_state, interruption_state, user_confidence) referenced by later phases — currently implicit across `turn_manager.py`/`channel.py` |
| 4.2 | Interruption classifier: barge-in vs. backchannel ("mhm", "right") vs. hesitation, from VAD + partial ASR words already available in `turn_manager.py` |
| 4.3 | Mid-generation response change: when a real interruption is detected, cut generation and re-condition rather than finish-then-listen |

**Acceptance:** 4.1/4.2 are pure state-machine logic, testable against recorded
VAD/ASR event sequences without audio hardware. 4.3 requires the live generation loop
— BLOCKED here, GPU-only.

---

## Phase 5 — 🟠 Prosody, emotion, conversational behavior

| # | Task |
|---|---|
| 5.1 | Extract prosodic features already implicit in Mimi's audio tokens/VAD signal (pitch trend, rate, pause structure) into `ConversationState.user_emotion` |
| 5.2 | Feed emotion/uncertainty signal into the retrieval urgency classifier (Phase 2.2) and the conditioning-strength function (Phase 1.1) — e.g. a hesitant question gets more cautious grounding |

**Acceptance:** feature extraction testable against recorded audio fixtures once
available; full loop is BLOCKED without GPU + real audio.

---

## Phase 6 — 🟠 Context management

| # | Task |
|---|---|
| 6.1 | Bound `turn_manager.get_context()` growth for long conversations — summarize old turns instead of dropping or unboundedly growing the prompt sent to the retrieval LLM |
| 6.2 | Let episodic memory (Phase 3.2) subsume old context instead of duplicating it |

---

## Phase 7 — 🟠 Reliability / fault isolation

Mostly the audit-driven fixes already shipped (Phase 0.4/0.5), extended:

| # | Task |
|---|---|
| 7.1 | Circuit breaker generalized to all cognitive services (Phase 2.3) |
| 7.2 | Startup validation for every required env var (`LLM_BASE_URL` done; extend the pattern to `REFERENCE_ENCODER_URL`-style checks project-wide) |
| 7.3 | Unify `requirements.txt`/`pyproject.toml` drift permanently — CI check that fails if they diverge |

---

## Phase 8 — 🟡 Performance optimization

GPU-bound; **entirely BLOCKED in this environment**. Recorded here so the benchmark
suite (Phase 9) has a fixed target list: token/frame latency, KV-cache behavior under
long conversations, batching efficiency, quantization quality/VRAM tradeoff,
speculative decoding applicability, Mimi codec latency, frame-size sensitivity.

## Phase 9 — 🟡 Developer architecture / API

| # | Task |
|---|---|
| 9.1 | Document the cognitive-sidecar interface so a new service (a calculator tool, a calendar API) is a ~30-line adapter, not a `channel.py` edit |
| 9.2 | **Moshi Conversational Intelligence Benchmark** — latency, interruption correctness, grounding accuracy, memory recall, turn-taking correctness, prosody appropriateness, long-context coherence; run against `baseline / +RAG / +memory / +predictive RAG / +memory+RAG / full stack` configurations |

## Phase 10 — 🟢 Generic voice-agent plumbing

Explicitly deprioritized — WebRTC transport, room management, etc. are LiveKit's job.
Only touched if it's blocking a higher-priority phase.

---

## Engineering time allocation (from the direction doc)

30% Phase 1 (RAG/dynamic conditioning) · 20% Phase 3 (memory) · 15% Phase 4
(full-duplex) · 15% Phase 5 (prosody) · 10% Phase 7 (reliability) · 10% Phase 8
(performance).

---

## Implementation log

**This session:** Phase 0 (0.1–0.3) — `moshi/moshi/cognitive/` package: `task_registry.py`,
`confidence.py`, `urgency.py`, `sidecar.py`, plus unit tests in `moshi/tests/test_cognitive.py`.
Torch/GPU is unavailable in this environment (see prior audit) — everything in Phase 0
is deliberately torch-free and independently testable; Phase 1.1's tensor-scaling code
is written against the real `streaming_sum` shape but its execution is **BLOCKED** here
and needs validation on GPU hardware with the actual checkpoint.
