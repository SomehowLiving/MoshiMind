# MoshiMind — Session Recap & Continuation Guide

**The machine you're running on has an NVIDIA GPU.** That's the single most
important fact in this document: everything below was built on a machine with
*no* GPU, no torch, and no openai/httpx installed, which made real execution of
the torch-dependent code impossible and shaped how the work was done (see §2).
You don't have that constraint. Your highest-value first move is almost certainly
§6 item 1 — actually running the server and validating the live wiring for real,
since that's the one category of work that was categorically impossible before now.

**Purpose of this file:** you (a fresh Claude session, on a different machine, with
zero memory of prior conversation) are picking up work on this repo. This document
tells you what the project is, what's already been decided and built, what
environment constraints shaped those decisions, and exactly how to keep going
without re-deriving everything from scratch or accidentally contradicting choices
that were made deliberately.

**Read `PHASES.md` at the repo root next** — it's the living roadmap with a
per-task status table and a detailed "status" writeup under every completed item
(what was built, why, what bugs were caught during testing, what's still open).
This file is the *narrative* onboarding; `PHASES.md` is the *source of truth* for
exact status. If the two ever disagree, trust `PHASES.md` and re-read the repo.

---

## 1. What this project is

**Repo:** `github.com/SomehowLiving/MoshiMind` (a fork of `kyutai-labs/moshi-rag`,
origin repointed — see §5). Branch: `main`.

MoshiMind is **not** a generic voice-agent framework (not "another LiveKit," not
"STT → LLM → TTS"). It's built on Kyutai's Moshi — a native full-duplex
speech-to-speech model — and the entire point of the work here is to exploit what's
actually unique about Moshi's architecture rather than bolt a conventional pipeline
onto it. The founding insight, from the original direction-setting conversation:

> Moshi already has a working mechanism for injecting external signal into its
> live generation loop without blocking it: the model emits a special
> `rag_token_id` text token mid-generation, which triggers an async retrieval
> (LLM call + a separate ARC-encoder microservice), and the result gets pushed
> into the transformer's input stream as a `streaming_sum` conditioning bias,
> one frame at a time. This is genuinely more interesting than a conventional
> retrieval pipeline, and it's the mechanism every phase of this project either
> widens (more producers, better-scored injection, predictive triggering) or
> protects (task identity, fault isolation, benchmarking).

The priority order that shaped what got built, and in what sequence:

1. 🔴 Native Moshi + RAG integration
2. 🔴 Streaming / asynchronous cognition (the "sidecar")
3. 🔴 Long-term conversational memory
4. 🔴 Full-duplex interruption / anticipation
5. 🟠 Prosody + emotion + conversational behavior
6. 🟠 Context management
7. 🟠 Reliability / fault isolation
8. 🟡 Performance optimization
9. 🟡 Developer architecture/API
10. 🟢 Generic voice-agent plumbing (explicitly deprioritized)

`PHASES.md` breaks each of these into numbered sub-tasks (1.1, 1.2, …) with
checkboxes and detailed status notes.

---

## 2. The environment constraint that shaped everything

**No GPU, no torch, no openai/httpx packages were available in the sandbox this
session ran in.** This is not a hypothetical — it was independently confirmed via
`pip`/`import` checks, and it's the single biggest thing that shaped *how* work got
done. Two consequences to carry forward:

1. **`moshi/moshi/__init__.py` eagerly imports torch-dependent submodules**
   (`models`, `conditioners`, `modules`, `quantization`). This means *anything*
   that does `import moshi` — including `import moshi.cognitive`, even though the
   `cognitive` package itself is pure Python — fails without torch installed. A
   workaround lives in `moshi/tests/conftest.py`: if torch genuinely isn't
   importable, it stubs `sys.modules["moshi"]` as an empty namespace package
   pointing at the real `moshi/moshi` directory, so `moshi.cognitive.*` imports
   work for testing without pulling in the torch-heavy submodules. **This has zero
   effect when torch is actually installed** (the normal import path is used
   unmodified) — it's purely a test-collection aid for torch-free environments.

2. **A whole new package, `moshi/moshi/cognitive/`, was built specifically to be
   torch-free and independently testable**, separate from the existing
   torch-dependent runtime code (`server.py`, `inference_utils/channel.py`,
   `inference_utils/rag_manager.py`, etc.). The pattern used throughout: implement
   the interesting logic (scheduling, scoring, classification, storage) as a pure
   Python module with full unit-test coverage, *then* wire it into the live,
   torch-dependent server code with as small and careful a diff as possible. The
   wiring itself could only be `py_compile`-checked here, never actually executed
   (no GPU/torch), so **if you're on a machine with a GPU and torch installed, your
   first move should be to actually run the live server and validate this wiring
   for real** — nothing about it has been executed end-to-end yet, only reasoned
   through carefully and unit-tested at the decision-logic level.

If your new machine *does* have torch/GPU: great, that unblocks the single biggest
category of "BLOCKED" items across `PHASES.md`. Test the live wiring first before
building further on top of it.

---

## 3. What's been built (`moshi/moshi/cognitive/`)

All of this is new code from this session, torch-free, unit-tested. Current test
count: **171 passing** across `test_cognitive.py`, `test_memory.py`,
`test_interruption.py`, `test_prosody.py` (in `moshi/tests/`). Run them with:

```bash
cd moshi
python -m pytest tests/test_cognitive.py tests/test_memory.py tests/test_interruption.py tests/test_prosody.py -q
```

(Don't run bare `pytest tests/` — `test_lm.py` is a pre-existing torch-dependent
test that will fail to collect without torch; that's expected and unrelated.)

### Phase 0 — Foundations
- `task_registry.py` — `TaskRegistry`/`TaskHandle`: `(conversation_id, turn_id,
  task_id)` identity for async ops; a result for a superseded turn is discarded.
- `confidence.py` — `ConfidenceScore(relevance, confidence, freshness)` →
  `strength()` via geometric mean (penalizes one weak axis harder than an
  arithmetic mean would); `.scaled(factor)` for provisional discounting.
- `urgency.py` — `Urgency` enum (CRITICAL/NORMAL/BACKGROUND) + heuristic classifier.
- `sidecar.py` — `CognitiveSidecar`: dispatch requests to registered
  `CognitiveService`s with urgency-based deadlines, per-service circuit breaking,
  and **active** staleness cancellation (`cancel`, `cancel_stale`, `cancel_all` —
  not just passive discard-on-arrival). `dispatch_and_wait` for a direct-await
  convenience on `CRITICAL` requests.

### Phase 1 — Native Moshi + RAG integration (**live-wired** into `channel.py`)
- **1.1** Confidence-weighted conditioning: `RAGManager.get_reference_text` scores
  every retrieval with `ConfidenceScore.heuristic_from_llm_reference` (empty/rushed
  answers penalized); `channel.py` scales the `streaming_sum` tensor by that
  strength before injection, or skips the ARC-encoder call entirely below
  `--rag-min-conditioning-strength`.
- **1.2** Predictive/speculative retrieval: `predictive_trigger.py::PredictiveTrigger`
  fires on the partial ASR transcript (question words, trailing "?", info-request
  phrases) before `rag_token_id` ever appears. `speculation.py::SpeculativeSlot`
  lets the real trigger *adopt* an in-flight/completed speculative attempt instead
  of retrieving twice.
- **1.3** `multishot.py::MultiShotGate` bounds confirmed `rag_token_id` triggers per
  model response (default 3, 2s cooldown) — repeated triggers in quick succession
  read as looping/uncertainty, not N genuine information needs.
- **1.4** `telemetry.py::RetrievalTelemetry` — structured per-retrieval log
  (query, text, confidence, latency, kind), exposed at `GET /api/retrieval_telemetry`.

### Phase 2 — Sidecar generalization
- Two real `CognitiveService` implementers beyond RAG: `tools/calculator.py`
  (genuine `CRITICAL`-urgency arithmetic via a whitelisted AST walk, **not**
  `eval()`) and `rag_service.py::RAGCognitiveService` (adapts `RAGManager` to the
  interface — built as an on-ramp, deliberately **not** wired into `channel.py`
  live, since RAG's live path has bespoke behavior — wait-steps, speculative reuse,
  multi-shot — the generic sidecar doesn't replicate).
- All three sidecar acceptance criteria pinned down with explicit tests: critical
  deadline enforcement, background non-blocking dispatch, full circuit-breaker
  lifecycle (open → stays-open-in-cooldown → half-open → closed).

### Phase 3 — Long-term memory (**live-wired**)
- `memory/working_memory.py::WorkingMemory` — bounded by turn count *and*
  character budget.
- `memory/store.py::MemoryStore` — **real SQLite** (stdlib, no new dependency),
  verified with an actual write-close-reopen-read persistence test.
- `memory/semantic_extraction.py::extract_facts` — regex-based fact extraction
  (name/occupation/location/preference). **A real bug was caught and fixed here**:
  the initial pattern was greedy past clause boundaries ("my name is Nidhi and I
  work as an engineer" extracted "Nidhi and I work as an engineer" as the name) —
  fixed with a word-count cap + conjunction cutoff, regression-tested.
- `memory/salience.py::score_salience` + `memory/memory_service.py::MemoryCognitiveService`
  — memory as a second `CognitiveService`, structurally identical to RAG's.
- `merge.py::merge_candidates` / `resolve_reference_conditioning` — arbitrates
  between RAG and memory candidates by `ConfidenceScore.strength()`, greedily
  concatenating within a character budget (they're usually complementary, not
  conflicting); merged confidence is the strongest candidate's own score, not an
  average, so a weak scrap can't dilute a strong hit.
- **Live wiring in `channel.py`**: both RAG trigger sites (confirmed + speculative)
  dispatch a memory lookup **concurrently** with RAG (`task_group.create_task`,
  never awaited inline) and merge results once RAG completes. User utterances are
  committed to memory (facts + salience-gated episodes) when the model's next
  response starts. **A real latency-regression mistake was caught and fixed here**:
  the first version awaited the memory lookup inline *before* triggering RAG, which
  would have blocked subsequent audio-frame processing in `_output_loop` for up to
  200ms on every RAG trigger — directly against the project's own "never destroy
  Moshi's low-latency architecture" thesis. Fixed to dispatch as an independent task
  and only await it inside the handler that already runs once RAG's own (slower)
  retrieval completes.
- `user_id` is a known placeholder throughout: this repo has no authentication
  layer, so there's no real "same user across sessions" concept — a per-connection
  UUID is what's used, documented as a caller limitation, not an oversight.

### Phase 4 — Full-duplex interruption (**live-wired**)
- `conversation_state.py::ConversationState` — `turn_id`, `speaking_state`
  (`IDLE`/`USER_SPEAKING`/`MODEL_SPEAKING`/`OVERLAPPING` — `OVERLAPPING` is the
  state a single `active_speaker` string structurally can't represent),
  `interruption_state`, `user_confidence`. `decide_interruption_response(state,
  kind)` maps a classification to `CONTINUE`/`YIELD`/`WAIT`.
- `interruption.py::InterruptionClassifier` — heuristic barge-in vs. backchannel
  ("mhm", "right") vs. hesitation, from VAD history (matching
  `turn_manager.TurnManager`'s convention: lower value = voice present) + partial
  ASR words. **A real bug was caught here too**: the initial version defaulted
  every ambiguous overlap to `HESITATION`, mislabeling a genuinely substantive
  multi-word utterance whose VAD just hadn't confirmed "sustained" yet — fixed to
  fall through to `NONE` in that case.
- **Phase 4.3 (the "cut generation" requirement) is implemented at the output
  level, deliberately not the generation level.** On a `YIELD` decision,
  `channel.py` immediately stops forwarding that channel's Opus audio bytes to the
  client (`_model_audio_muted`, gated in `_output_loop`) and sends an
  `[INTERRUPTED]` marker. This is a real, justified mechanism — not a shortcut —
  because Moshi's full-duplex design means the model is *always* processing the
  user's live audio regardless of whose turn `TurnManager` thinks it is; muting
  what the user hears is exactly what "yielding the floor" means from their actual
  experience, and it touches zero shared batch state (unlike a step-loop change
  would). **What it explicitly does not do**: stop the transformer from computing
  more tokens internally (no compute saved), or correct the model's internal sense
  of "what it said." That would require a hook into the shared batched step loop
  (`BatchRunner`/`LMGen` in `server.py`/`inference_utils/batch_runner.py`), which is
  GPU-only to build and verify safely and was **not attempted** rather than guessed
  at blind. This is the most concrete "needs a GPU to finish properly" item in the
  whole roadmap — see §6.

### Phase 5.1 (pulled forward, done alongside Phase 4)
- `prosody.py::ProsodyTracker`/`estimate_pitch`/`rms_energy` — **real acoustic
  signal**, not another heuristic: autocorrelation-based F0 estimation and RMS
  energy over raw PCM, classical DSP (no trained model). Genuinely tested against
  synthetic audio — numpy is installed in this sandbox even though torch isn't, so
  a 150Hz sine wave is verified to be detected as 150Hz, silence/noise correctly
  return `None`, and a declining-amplitude sequence is verified as "trailing off."
  **A real bug was caught here**: `.mean()` was called on a possibly-empty array
  before the size check, throwing a `RuntimeWarning` — reordered, fixed.
  `InterruptionClassifier` now takes this as a one-way override: declining energy
  can downgrade a lexical barge-in call to hesitation, never the reverse.
- Fed frame-by-frame in `_recv_loop` from the same raw PCM already being decoded,
  before the torch conversion — no new audio path needed.
- **Phase 5.2 (feeding this signal into retrieval urgency/confidence scoring) is
  not started** — natural next small step, see §6.

---

## 4. A pattern you should keep following

Almost every phase above used the same discipline, and it's worth preserving:

1. **Build the interesting logic as a pure, torch-free module first.** Put it in
   `cognitive/`, write real unit tests (not mocks-all-the-way-down where avoidable —
   e.g. `MemoryStore` tests hit real SQLite, `prosody.py` tests use real generated
   sine waves).
2. **Only then wire it into the live, torch-dependent code**, with the smallest,
   most defensible diff you can manage. Read the surrounding code carefully first —
   several real bugs (see above) were caught specifically because the pure-logic
   layer was tested in isolation *before* being trusted in the wiring.
3. **Be explicit about what's live-wired vs. what's "proven correct but not yet
   connected."** `RAGCognitiveService` and `MemoryCognitiveService`-via-sidecar
   pattern were deliberately built but *not* used to replace the live RAG path,
   because that path has tuned, tested, bespoke behavior the generic sidecar
   doesn't replicate — rewiring it "just because the sidecar exists" would be churn
   without a behavior change. Don't do integration for its own sake.
4. **When something genuinely can't be built/verified in this environment (GPU-only
   work), don't guess at it blind.** State exactly what's missing and why, rather
   than writing untested code that touches shared, high-blast-radius state (like
   the batched generation loop) and hoping it's right. `PHASES.md`'s status notes
   are full of exactly this kind of honest boundary-drawing — read them for the
   tone to match.
5. **When you catch a bug while testing, say so explicitly** (in commit messages,
   in `PHASES.md` status notes) rather than silently fixing it. Several real
   defects were caught this way this session and are documented as such — that
   record has value for anyone auditing the work later.

---

## 5. Git / repo housekeeping context

- `origin` was originally `kyutai-labs/moshi-rag` (the upstream this repo forked
  from) — it was repointed to `https://github.com/SomehowLiving/MoshiMind.git`
  early in this session because the user doesn't have write access upstream. If
  you clone fresh on a new machine, you should already be pointed at the right
  remote — but if you ever see a push get rejected with a permissions error
  mentioning `kyutai-labs`, check `git remote -v` and fix it the same way:
  `git remote set-url origin https://github.com/SomehowLiving/MoshiMind.git`.
- **Commit messages are short, one-line, and have no AI co-author trailer** — the
  user explicitly asked for this early on (history was rewritten once to strip
  `Co-Authored-By: Claude` lines and shorten messages). Keep following that
  convention: no multi-paragraph commit bodies, no co-author trailers.
- An architecture/execution audit of the *original* upstream repo (before any of
  this session's changes) was done earlier and published as a Claude Artifact
  titled "MoshiRAG Field Audit" — it's not in the repo itself, just mentioned here
  in case it's useful context for understanding the baseline the fixes/PHASES work
  was reacting to (reliability bugs like ARC-encoder failures killing the whole
  session, a bare `KeyError` on missing `LLM_BASE_URL`, inconsistent dependency
  pins between `requirements.txt` and `pyproject.toml` — all fixed in the first two
  commits of this session's history, before `PHASES.md` existed).

---

## 6. Concrete next steps, in priority order

1. **Validate the live wiring for real, first — this machine has the GPU that
   makes it possible.** Nothing in Phases 1/3/4 has ever actually been executed —
   it's been reasoned through and unit-tested at the decision-logic level only.
   Concretely:
   - Install real dependencies (`pip install -e moshi/` or per `moshi/pyproject.toml`;
     note the `requirements.txt`/`pyproject.toml` pin-drift was already fixed once
     this session — check they're still in sync before trusting either blindly).
   - Start the server (`moshi-server-py`, see `moshi/pyproject.toml`
     `[project.scripts]`) with a real `LLM_BASE_URL`/`REFERENCE_ENCODER_URL`.
   - Have a real conversation and check, one by one: does confidence-weighted
     conditioning behave sanely (§3 Phase 1.1)? Does speculative retrieval actually
     fire and get reused instead of double-retrieving (1.2)? Does the multishot
     gate actually cap repeated triggers (1.3)? Does memory actually get recalled
     and merged with RAG (Phase 3)? Does a real interruption actually mute audio
     within a perceptible beat (Phase 4.3)? Does the prosody-based override ever
     actually fire on real speech (5.1)?
   - Fix whatever the real hardware surfaces. Treat every claim in `PHASES.md`'s
     "implemented, live-wired" status notes as a well-reasoned hypothesis until
     you've watched it happen against real generation — the unit tests prove the
     decision logic is internally consistent, not that the integration is bug-free.
2. **Phase 4.3's remaining half, now actually buildable**: hook the batched step
   loop (`BatchRunner`/`LMGen` — see `moshi/moshi/inference_utils/batch_runner.py`
   and `moshi/moshi/models/lm.py`) so a `YIELD` decision actually stops the
   transformer from computing more tokens for that slot, not just stops forwarding
   audio. This was explicitly *not* attempted without a GPU — you have one now, so
   this is a real, unblocked next step, not a "someday." Go carefully: this loop is
   shared across every concurrently connected channel's batch slot, so a bug here
   has a much larger blast radius than anything built so far.
3. **Phase 5.2**: thread `ProsodyTracker.energy_trend()`/`pitch_trend()` into
   `classify_urgency` (Phase 2.2) and `ConfidenceScore` (Phase 1.1) — e.g. a
   hesitant, declining-energy question gets a lower-confidence grounding pass.
   Small, natural extension of what's already built; doesn't need a GPU.
4. **Phase 6 (context management)**: bound `TurnManager.conversation_context`'s
   unbounded growth — `cognitive/memory/working_memory.py::WorkingMemory` was
   already built with exactly this in mind (Phase 3.1) but not yet swapped in to
   replace the live accumulator; that swap is Phase 6's job.
5. Everything else follows the priority order in §1 / `PHASES.md`.

---

## 7. If the other Claude asks "what should I tell the user / what should I do first"

Point it at `PHASES.md` for the authoritative status table, tell it to run the
existing test suite (§3 command) to confirm the baseline still holds, and — if this
new machine actually has a GPU — treat live validation (§6 item 1) as the single
highest-value thing it could do, since it's the one category of work that was
*impossible* in the environment this was built in.
