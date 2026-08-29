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
| 1.3 ✅🧪 | Dynamic (multi-shot) conditioning | Today one reference → one conditioning tensor → whole response. Extend `RAGManager` to accept a follow-up retrieval mid-response (second `rag_token_id`, or the predictive trigger firing again) and push a second tensor through the *existing* per-slot streaming update path — the plumbing (`update_streaming_sum_tensors` accepts a fresh tensor per slot at any time) already supports this; what's missing is deciding *when* a second retrieval is worth the interruption | `cognitive/multishot.py`, `channel.py` |
| 1.4 ✅🧪 | Retrieval quality benchmark hook | Every retrieval logs `(query, reference_text, confidence, latency, applied: bool)` to a structured log the Phase 9 benchmark suite can replay | `cognitive/telemetry.py` |

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

**1.3 status:** implemented. The multi-shot *plumbing* needed no changes —
`update_streaming_sum_tensors` already accepts a fresh tensor per slot at any
time, and `Channel._output_loop` already treats every `rag_token_id` emission
as an independent trigger, so a second retrieval mid-response already worked.
What was missing was the *judgment* call: `cognitive/multishot.py::MultiShotGate`
bounds confirmed triggers to `--rag-max-shots-per-turn` (default 3) with a
`--rag-shot-cooldown-seconds` (default 2.0) between them, so several
`rag_token_id` emissions in quick succession — plausibly model looping/
uncertainty rather than distinct information needs — don't each pay for a full
retrieval + ARC-encoder round trip. `Channel` checks the gate before honoring a
trigger (declining ones fall through to `rag_multishot_declined_total` and the
model just continues on its existing conditioning) and resets the gate's budget
at the start of each new model response, detected via the same turn-manager
speaker-switch marker already used to reset the Phase 1.2 predictive trigger.
Deliberately scoped to the live server only, not `inference_job.py`'s offline
batch eval path, where seeing every genuine shot is more useful than bounding
cost. 5 new unit tests, all torch-free (45/45 total in `test_cognitive.py`).

**1.4 status:** implemented, closing out Phase 1. `cognitive/telemetry.py::RetrievalTelemetry`
is a bounded structured log; `RAGManager.get_reference_text` records one event per
attempt (query, reference text, confidence, latency), tagged `kind="confirmed"` or
`"speculative"` depending on which trigger called it. `applied` here means only
"the retrieval step produced a usable candidate" — whether it actually ended up
biasing generation is a separate, later decision already tracked by
`rag_low_confidence_skipped_total`/`rag_stale_reference_dropped_total`; conflating
the two would overclaim what this log proves. Exposed at `GET /api/retrieval_telemetry`
alongside the existing `/api/metrics`. 7 new unit tests, all torch-free (52/52 total
in `test_cognitive.py`). Phase 1 (native Moshi + RAG integration) is now complete.

---

## Phase 2 — 🔴 Streaming / asynchronous cognition (the sidecar, generalized)

Phase 0.2 built the skeleton. This phase populates it with real producers beyond RAG
and enforces the non-blocking guarantee everywhere.

| # | Task | Status |
|---|---|---|
| 2.1 | Formalize the "cognitive service" interface (`async def handle(request) -> CognitiveResult`) so RAG, memory, and tools are structurally identical to the sidecar | ✅🧪 |
| 2.2 | Urgency-based scheduling: `critical` requests get a short deadline and may block a few hundred ms (e.g. arithmetic); `background` requests never block, results arrive whenever and are applied only if still relevant (task registry handles staleness) | ✅🧪 |
| 2.3 | Circuit breaker per cognitive service — N consecutive failures suspends that service for a cooldown window instead of retrying into a dead endpoint every turn | ✅🧪 |

**Acceptance:** unit tests with fake slow/failing services proving (a) `critical`
requests respect their deadline, (b) `background` requests never block the caller,
(c) the breaker opens/half-opens/closes correctly. All torch-free.

**Status:** the interface, deadlines, and breaker were actually built back in Phase 0
(`cognitive/sidecar.py`) as part of the initial scaffold, but had only ever been
exercised by throwaway fakes — Phase 2's job was proving they hold up against real
producers and pinning the three acceptance criteria down with explicit tests, not
building new scheduling machinery.

- **Two real `CognitiveService` implementers**, closing 2.1: `cognitive/tools/calculator.py::CalculatorService`
  — a genuine `Urgency.CRITICAL` service (arithmetic, evaluated via a whitelisted AST
  walk, not `eval()`, since the input is user-speech-derived text) — and
  `cognitive/rag_service.py::RAGCognitiveService`, which adapts `RAGManager.get_reference_text`
  to the interface. The RAG adapter is *not* wired into `channel.py` — RAG's live path
  has bespoke behavior (STT wait steps, speculative reuse, the multi-shot budget) the
  generic sidecar doesn't replicate, and rewiring a working, tested path for its own
  sake would be churn without a behavior change. It exists to prove the interface fits,
  and as the on-ramp for a future migration.
- **2.2 acceptance nailed down**: `test_critical_request_respects_short_deadline` proves
  a `CRITICAL` request against a 5s-slow fake returns in well under 1s (the 0.3s default
  deadline held); `test_background_request_dispatch_returns_immediately_even_for_a_slow_service`
  proves `dispatch()` itself never blocks regardless of urgency (it's `asyncio.ensure_future`
  under the hood); `test_background_request_never_times_out_no_matter_how_slow` proves
  `BACKGROUND` truly has no deadline.
- **2.3 acceptance nailed down**: `test_circuit_breaker_full_lifecycle_open_half_open_closed`
  drives a fake failing service through the whole cycle — three failures open the breaker,
  it stays open within the cooldown, half-opens to allow one trial after the cooldown
  elapses, and a successful trial closes it again.

15 new unit tests, all torch-free (67/67 total in `test_cognitive.py`).

**Pushed further:** staleness was only *passive* — a superseded task ran to completion
and its result was silently discarded on arrival (`_run`'s `registry.is_current` check).
That leaves a stale LLM call/HTTP connection/GPU-adjacent work running for no reason
after the conversation has moved on. `CognitiveSidecar` now tracks every dispatched
task's `asyncio.Task` alongside its `TaskHandle` and adds:

- `cancel(handle)` — cancel one specific dispatched task if still running.
- `cancel_stale(conversation_id)` — cancel every outstanding task whose `turn_id` no
  longer matches the registry's current turn; call this right after
  `TaskRegistry.advance_turn` so superseded work actually stops instead of running to
  a discarded result.
- `cancel_all(conversation_id)` — cancel everything for a conversation, for teardown.
- `dispatch_and_wait(...)` — a direct-await convenience for `Urgency.CRITICAL` callers
  (e.g. the calculator) who don't want to set up an `on_result` callback just to get
  one blocking answer; warns (doesn't refuse) if used for non-critical urgency, since
  that would defeat the point of dispatching those off the critical path.

Not yet wired into `channel.py`: no live `CognitiveSidecar` instance exists in the
server today (RAG's live path still calls `RAGManager` directly, per the 2.1 scoping
note above), so calling `cancel_stale` on turn transitions has nothing to attach to
yet. This machinery is ready for the first live sidecar user — most likely Phase 3
(memory), where the cost of letting a superseded background lookup keep running is
more concrete than it is for the not-yet-migrated RAG path. 11 new unit tests, all
torch-free (76/76 total in `test_cognitive.py`).

---

## Phase 3 — 🔴 Long-term conversational memory

Kept structurally separate from RAG (`RAG` = what does the world know, `memory` =
what does *this* user/conversation know) even though both ultimately compete for the
same conditioning channel into Moshi.

| # | Task | Status |
|---|---|---|
| 3.1 | Working memory — already exists as `turn_manager.get_context()`; formalize its window and eviction policy | ✅🧪 |
| 3.2 | Episodic memory — persist salient turns (heuristic: turns near a topic change, an emotional peak, or an explicit "remember this") to a per-user store keyed off session id | ✅🧪 |
| 3.3 | Semantic memory — durable facts/preferences extracted from episodic memory on a slower cadence (batch job, not per-turn) | ✅🧪 |
| 3.4 | Memory retrieval as a second cognitive-sidecar producer, competing with RAG for conditioning bandwidth — needs a merge policy (Phase 1.1's confidence score is the natural arbitration signal) | ✅🧪 |

**Acceptance:** memory store and retrieval logic are pure Python/SQLite — fully
testable without a GPU. Integration with the conditioning channel is BLOCKED here,
same as Phase 1.

**Status:** all four implemented under `cognitive/memory/`, none wired into
`channel.py`/`turn_manager.py` live yet (same "prove the primitive, wire it later"
pattern as `cognitive.rag_service` and the Phase 2 cancellation machinery) — this
phase is about proving the storage, extraction, and arbitration logic is correct in
isolation, not about swapping out a working live conversation path.

- **3.1** `cognitive/memory/working_memory.py::WorkingMemory` — bounded by both turn
  count and total characters (either can dominate depending on the conversation),
  evicting oldest-first, always keeping at least one turn even if a single turn
  exceeds the character budget.
- **3.2** `cognitive/memory/store.py::MemoryStore` (SQLite, stdlib — no new dependency,
  genuinely durable across process restarts, not just in-process state) plus
  `cognitive/memory/salience.py::score_salience`, a placeholder heuristic (explicit
  "remember this"-style phrases score highest; long turns moderately; bare questions
  low) standing in for the topic-change/emotion signals PHASES.md's own description
  calls for but this repo can't detect yet (emotion is Phase 5).
- **3.3** `cognitive/memory/semantic_extraction.py::extract_facts` — a deliberately
  simple regex-pattern extractor (name/occupation/location/preference), explicitly
  labeled as a placeholder for the real implementation (an LLM pass over accumulated
  episodes, off the critical path — the same pattern RAG's reference generation
  already uses). Caught and fixed a real bug during testing: the initial pattern was
  greedy past clause boundaries ("my name is Nidhi and I work as an engineer" was
  extracting "Nidhi and I work as an engineer" as the name) — fixed with a word-count
  cap plus a conjunction cutoff, verified with a regression test.
- **3.4** `cognitive/memory/memory_service.py::MemoryCognitiveService` — structurally
  identical `CognitiveService` to `RAGCognitiveService`, so both can be dispatched
  through the same sidecar; `cognitive/merge.py::merge_candidates` arbitrates between
  whatever candidates come back, ranking by `ConfidenceScore.strength()` and greedily
  concatenating the strongest ones within a character budget (RAG and memory are
  usually complementary, not conflicting, so combining beats picking one and
  discarding the other) — the merged result's confidence is the strongest single
  candidate's own score, not an average, so one weak scrap can't dilute a strong hit.
- `user_id` is a known placeholder: this repo has no authentication layer (per the
  earlier execution audit), so there's no real "same user across sessions" concept
  yet — callers currently have nothing better than a per-connection id, making
  memory effectively per-conversation until a real identity system exists. That
  limitation lives in the caller, documented in `cognitive/memory/store.py`.

36 new unit tests, all torch-free, including genuine sqlite3 execution (not mocked)
and a persistence round-trip test (write, close, reopen, read back). 113/113 total
across `test_cognitive.py` + `test_memory.py`.

**Live-wired, breaking the "prove it, wire it later" pattern for the first time**:
this is the sidecar's and memory's first real integration into `channel.py`, not
another isolated addition — `ServerState` now owns a `CognitiveSidecar` and a
`MemoryStore`, with `MemoryCognitiveService` registered on it (`--memory-db-path`,
default `:memory:`; `--memory-lookup-deadline`, default 0.2s; `--memory-salience-threshold`,
default 0.5). At both RAG trigger sites (confirmed `rag_token_id` and speculative),
`Channel` now:

1. Dispatches a memory lookup as its own task (`task_group.create_task`, *not*
   awaited inline) at the same moment RAG's own retrieval starts.
2. Passes that task into `_bind_reference_handler`, which awaits it only inside the
   handler that already runs once RAG's retrieval completes — never in
   `_output_loop`/`_stt_recv_loop` directly. **This was a real mistake caught and
   fixed during this session**: the first version awaited the memory lookup inline
   before triggering RAG, which would have blocked subsequent audio-frame processing
   in `_output_loop` for up to the full `--memory-lookup-deadline` (200ms) on every
   single RAG trigger — a direct violation of the project's own "never destroy
   Moshi's low-latency architecture" thesis. Since a local sqlite lookup is far
   cheaper than RAG's network round trip, awaiting it after RAG completes costs
   effectively nothing in practice.
3. Merges RAG's result with the memory candidate via `cognitive.merge.resolve_reference_conditioning`
   — a new pure function factored out of the handler specifically so this exact
   decision (merge when memory has something; fall back to RAG's own result,
   including its failure case, when it doesn't) is unit-testable without importing
   `channel.py` (torch-gated, unavailable in this environment). 6 new tests cover it,
   including the case where RAG fails entirely but memory still grounds the response.
4. Commits the just-completed user utterance to memory (`extract_facts` → `upsert_fact`,
   `score_salience` → `add_episode` if above threshold) the moment the model's next
   response starts — i.e., right after the user's turn ends, reusing
   `_current_user_utterance` (already accumulated for the Phase 1.2 predictive trigger).

Also added `GET /api/memory/{user_id}` for inspection. This wiring itself remains
**BLOCKED for execution** in this environment (torch/openai unavailable, per the
execution audit) — the decision logic it depends on (`resolve_reference_conditioning`,
`merge_candidates`, the memory store/extraction pipeline) is fully tested; the
`asyncio.Task`-based non-blocking dispatch pattern is the same one already proven
correct for the confirmed-vs-speculative RAG race in Phase 1.2. 6 new unit tests,
119/119 total across `test_cognitive.py` + `test_memory.py`.

---

## Phase 4 — 🔴 Full-duplex interruption / anticipation

This is the one area where the plan explicitly says *don't* let infrastructure
(LiveKit) own the interesting part. LiveKit/WebRTC-level VAD and endpointing are fine
for transport; conversational-state decisions (is this a real interruption, is the
user hesitating, should Moshi yield) belong in the Moshi-facing layer.

| # | Task | Status |
|---|---|---|
| 4.1 | Formalize `ConversationState` (turn_id, speaking_state, interruption_state, user_confidence) referenced by later phases — currently implicit across `turn_manager.py`/`channel.py` | ✅🧪 |
| 4.2 | Interruption classifier: barge-in vs. backchannel ("mhm", "right") vs. hesitation, from VAD + partial ASR words already available in `turn_manager.py` | ✅🧪 |
| 4.3 | Mid-generation response change: when a real interruption is detected, cut generation and re-condition rather than finish-then-listen | ✅🧪 (output-level cut; see status) |

**Acceptance:** 4.1/4.2 are pure state-machine logic, testable against recorded
VAD/ASR event sequences without audio hardware. 4.3 requires the live generation loop
— BLOCKED here, GPU-only.

**Status:** 4.1/4.2 implemented, tested, *and* live-wired into `channel.py` — this
phase does not follow the "prove it, wire it later" pattern used for RAG/memory
because the wiring here is pure bookkeeping/observability, not a change to what
Moshi generates, so there was no reason to hold it back.

- **4.1** `cognitive/conversation_state.py::ConversationState` — `turn_id`,
  `speaking_state` (`IDLE`/`USER_SPEAKING`/`MODEL_SPEAKING`/`OVERLAPPING`),
  `interruption_state`, `user_confidence`. `OVERLAPPING` is the state
  `TurnManager.active_speaker` (a single string) structurally cannot represent —
  the entire point of tracking this separately: a half-duplex turn-taking model has
  no way to say "both are talking right now," which is exactly the full-duplex case
  this phase cares about. A separate `decide_interruption_response(state, kind)`
  policy function maps a classification to `CONTINUE`/`YIELD`/`WAIT`.
- **4.2** `cognitive/interruption.py::InterruptionClassifier` — heuristic, same
  placeholder posture as `PredictiveTrigger`: real barge-in detection wants prosody
  (Phase 5) and semantic judgment of the partial words, neither of which exists yet.
  Uses only what's already available: VAD history (matching `TurnManager`'s own
  convention — lower value means voice present) and the partial ASR transcript.
  Backchannel = short exact-phrase match ("mhm", "right", ≤2 words); barge-in =
  sustained low-VAD frames *and* enough words to be substantive; hesitation = a
  trailing filler word, or a short utterance without confirmed sustained voice;
  otherwise `NONE` (ambiguous, insufficient signal — deliberately not defaulted to
  hesitation, since a real multi-word utterance whose VAD just hasn't caught up yet
  shouldn't be mislabeled). Tested against recorded-style VAD/ASR event sequences,
  including one exercising backchannel → hesitation → barge-in as a single session
  the way the classifier is actually called, once per incoming partial transcript.
- **Live-wired**: `Channel` now owns a `ConversationState` and `InterruptionClassifier`.
  `begin_user_turn()`/`begin_model_turn()` fire at the same turn-transition points
  already used for the Phase 1.2/1.3/3 resets. When `STTWordMessage`s keep arriving
  while `TurnManager.active_speaker` still says `"model"` — literal full-duplex
  overlap — the classifier runs, `ConversationState.note_overlap` updates, the
  decision is logged, and `interruption_{kind}_total` metrics are incremented.
- **4.3 update — implemented at the output level, deliberately not at the generation
  level.** A `YIELD` decision now actually cuts what the user hears: `Channel` stops
  forwarding this channel's Opus audio bytes to the client (`_model_audio_muted`,
  gated in `_output_loop`) and sends an `[INTERRUPTED]` marker, immediately, the
  moment a barge-in is classified. The opus encoder keeps being fed regardless of
  mute state so its internal buffering stays continuous; only the websocket send is
  gated. What this deliberately does *not* do is touch the shared batched generation
  loop (`BatchRunner`/`LMGen`) — that loop is shared across every concurrently
  connected channel's GPU batch slot, and altering its per-step behavior is GPU-only
  to build and verify safely; a bug there risks every channel sharing the batch, not
  just the one being interrupted. The justification for why an output-level cut is a
  legitimate (not a compromise) interruption mechanism: Moshi's full-duplex design
  means the model is *always* processing the user's live audio regardless of whose
  turn `TurnManager` thinks it is — muting the model's audio to the user is exactly
  what "yielding the floor" means from the user's actual experience, achieved without
  any risk to shared state. Un-mutes automatically at the start of the model's next
  response (`begin_model_turn()`).
- **5.1 folded in here**: `cognitive/prosody.py` adds real acoustic signal —
  autocorrelation-based pitch (F0) estimation and RMS energy over raw PCM, tracked
  over a rolling window (`ProsodyTracker`) to expose energy/pitch trends. Classical
  DSP, not a trained model, but genuinely tested against synthetic audio (numpy is
  actually installed in this environment, unlike torch) rather than written blind —
  a 150Hz sine wave is detected as 150Hz, silence and white noise correctly return
  `None` rather than a meaningless number, and a declining-amplitude tone sequence is
  correctly detected as "trailing off." Fed frame-by-frame from the same raw PCM
  `_recv_loop` already decodes (before the torch conversion), so no new audio path
  was needed. `InterruptionClassifier.classify` now accepts an optional
  `energy_trend` and uses it as a one-way override: real acoustic evidence of
  declining energy downgrades a lexical/VAD barge-in call to hesitation (someone
  trailing off, not committing), but never the reverse — word count and VAD can
  suggest an attempt to interject; declining energy is direct evidence they didn't
  follow through on it.

47 new unit tests, all torch-free (171/171 total across `test_cognitive.py` +
`test_memory.py` + `test_interruption.py` + `test_prosody.py`). Two real bugs caught
during testing this round: the classifier one below, and a `RuntimeWarning` in
`estimate_pitch` from calling `.mean()` on a possibly-empty array before checking its
size — reordered the check, now silent and correct on the empty/too-short input case.
One real classifier bug caught during testing: the initial version defaulted every
non-barge-in, non-backchannel overlap to `HESITATION`, which would have mislabeled a
substantive multi-word utterance whose VAD signal simply hadn't confirmed "sustained"
yet — fixed to fall through to
`NONE` in that ambiguous case, with a regression test for the distinction.

---

## Phase 5 — 🟠 Prosody, emotion, conversational behavior

| # | Task | Status |
|---|---|---|
| 5.1 | Extract prosodic features already implicit in Mimi's audio tokens/VAD signal (pitch trend, rate, pause structure) into `ConversationState.user_emotion` | ✅🧪 partial (see Phase 4 status) |
| 5.2 | Feed emotion/uncertainty signal into the retrieval urgency classifier (Phase 2.2) and the conditioning-strength function (Phase 1.1) — e.g. a hesitant question gets more cautious grounding | 🔲 |

**Acceptance:** feature extraction testable against recorded audio fixtures once
available; full loop is BLOCKED without GPU + real audio.

**5.1 status:** pulled forward and implemented alongside Phase 4 — `cognitive/prosody.py`
(`ProsodyTracker`, `estimate_pitch`, `rms_energy`) extracts real pitch/energy from raw
PCM and is already consumed by the Phase 4.2 interruption classifier. Genuinely
tested against synthetic audio (ground-truth sine waves), not audio fixtures, since
none exist in this repo yet — see the Phase 4 status block above for detail. Not yet
feeding `ConversationState.user_emotion` (no such field exists — Phase 4.1 only
formalized `user_confidence`, which the classifier does update) or a real emotion
model; that's still 5.2-adjacent future work, tracked separately below.

**5.2 status:** not started. The natural next step for this signal: thread
`ProsodyTracker.energy_trend()`/`pitch_trend()` into `classify_urgency` (Phase 2.2)
and `ConfidenceScore` (Phase 1.1) the same way it's now used in the interruption
classifier — e.g. a hesitant, declining-energy question could get a lower-confidence
grounding pass, exactly as PHASES.md originally sketched.

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

## Live validation session (GPU hardware, first real execution)

Everything in this section happened on a machine with a real GPU (RTX 5050 Laptop,
8GB VRAM, Blackwell/sm_120) — the first time any of this repo's torch-dependent code
has actually been executed, as opposed to reasoned through and unit-tested against
fakes. See `RECAP.md` for the full handoff context this picked up from.

**Environment bugs found only by executing the code (not visible from reading it):**
- `moshi/pyproject.toml` had `readme = "../README.md"`, which a modern `hatchling`
  rejects (readme must be inside the project directory) — `pip install -e moshi/`
  failed metadata generation before installing anything. Fixed by adding
  `moshi/README.md` (copy of the root one) and pointing `readme` at it.
- `websockets` is imported directly in `channel.py` but was in neither
  `pyproject.toml` nor `requirements.txt` — added to both (`>=13.0,<16.0`), the same
  drift class the prior session's audit had already fixed once for other packages.

**8GB VRAM doesn't fit the real checkpoint — quantization wired in:** the real
`kyutai/moshika-rag-pytorch-bf16` checkpoint is ~7B params / ~15GB in bf16, which does
not fit an 8GB card. The codebase already had a dormant, correct-but-unwired
int8 quantization utility (`moshi/moshi/utils/quantize.py::QLinear`/
`replace_linear_with_qlinear`, bitsandbytes-based) that nothing called. Wired it into
`moshi/moshi/models/loaders.py::get_moshi_lm` behind a new `--quantize-8bit` CLI flag
(`server.py`): the state dict is staged on CPU (not the GPU) when quantizing, then
every `nn.Linear` is converted to int8 one layer at a time — moved to the target
device, quantized, replaced — so peak memory never holds the full float model on the
GPU at once. **Verified working**: the full stack (Mimi + quantized LM + LMGen,
`--batch-size 1`) loads and runs in ~8.4GB, all 338 Linear layers converted, 0 left in
float. This is the single biggest previously-BLOCKED item in this file — Phase 1/3/4's
tensor-scaling and conditioning code has now actually executed on real hardware, not
just against unit-test fakes.

**A real, pre-existing reliability gap fixed**: `LLMReferenceGenerator.__init__`
called `_warmup_all_retrieval_llms()` with no fault isolation — a bad API key,
exhausted quota, or unreachable endpoint raised out of `__init__` and crashed the
*entire server* before it could serve anything, taking every RAG-independent feature
(memory, interruption, prosody) down with it. This is inconsistent with the project's
own fault-isolation philosophy for the ARC-encoder path (Phase 0.4) — fixed to log and
continue startup instead, retrying at request time.

**A real, previously-undetectable concurrency bug found and fixed**: once a WebSocket
client connected, the server appeared to hang — even a completely unrelated `GET
/api/health` request from a different client would time out for 20+ seconds, while
GPU utilization stayed near 0% (ruling out "just slow inference"). Root cause: `handle_chat`
in `server.py` called `Channel(self, ws, mimi=deepcopy(self.mimi_copy))` inline,
synchronously, inside the async handler. Two things inside that construction are slow:
deep-copying a CUDA-resident Mimi model, and — more expensive — `Channel.__init__`
building a `LocalSpeechToText`, which loads and constructs an entire separate
**~1B-parameter STT checkpoint from HF, from scratch, on every single new connection**
(`moshi/moshi/stt/local_stt.py`). Python's asyncio event loop is single-threaded, so
this synchronous work blocked *every* connection's requests, not just the new one.
Fixed by moving both the `deepcopy` and the full `Channel(...)` construction onto
`loop.run_in_executor`, so the event loop stays responsive while a new connection's
(unavoidably slow) setup happens on a worker thread. Verified: `/api/health` stayed
fast (<0.3s) while a client was actively connecting, and 3 repeated connect/disconnect
cycles left GPU memory completely stable (no leak from the per-connection STT model).

**RAG grounding specifically is still blocked**, not by hardware but by external
access: the reference/ARC-encoder conditioner needs `meta-llama/Llama-3.2-3B-Instruct`,
a gated HF repo requiring the user's HF account to request and be granted access, plus
an `HF_TOKEN`. The retrieval LLM itself (RAG's *answer-generation* step, separate from
the ARC encoder) now runs live against Groq's OpenAI-compatible API
(`https://api.groq.com/openai/v1`, model `openai/gpt-oss-20b`) after the original
OpenAI key hit `insufficient_quota` — confirmed via `LLMClient`'s existing
`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL_NAME` env vars, no code changes needed for the
swap itself.

**Also fixed**: `moshi-server-py` (`server.py::main`) referenced `args.dtype` with no
`--half`/`--dtype`-style flag defining it in that file (present in `run_inference.py`
but not here) — would have crashed on the very first real run. (Turned out a `--half`
flag already existed further down in the same file, just not matched by an initial
grep for `dtype` in the argument name — no duplicate fix needed once found, but
recorded here since it's exactly the kind of "only visible by reading the whole file /
running it" issue this section is about.)

Server confirmed healthy end-to-end: boots, loads the quantized model, warms up,
accepts a real WebSocket client, and stays responsive to other clients throughout —
all genuinely executed for the first time, not reasoned through against fakes.

---

## Implementation log

**This session:** Phase 0 (0.1–0.3) — `moshi/moshi/cognitive/` package: `task_registry.py`,
`confidence.py`, `urgency.py`, `sidecar.py`, plus unit tests in `moshi/tests/test_cognitive.py`.
Torch/GPU is unavailable in this environment (see prior audit) — everything in Phase 0
is deliberately torch-free and independently testable; Phase 1.1's tensor-scaling code
is written against the real `streaming_sum` shape but its execution is **BLOCKED** here
and needs validation on GPU hardware with the actual checkpoint.
