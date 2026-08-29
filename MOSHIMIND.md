# MoshiMind

**MoshiMind is a full-duplex, speech-native cognitive layer built on top of Kyutai's
Moshi** — not another STT → LLM → TTS voice-agent framework. The bet this project
makes is narrow and deliberate: Moshi already has a working mechanism for injecting
external signal into its live generation loop without blocking it (a `rag_token_id`
text token the model can emit mid-response, which triggers an async retrieval and
gets summed back into the transformer's input stream as a conditioning bias, one
frame at a time). Everything here either **widens** that one mechanism — more
producers of conditioning, better-scored injection, predictive triggering — or
**protects** it — task identity, fault isolation, benchmarking. Nothing in this repo
reaches for a generic agent framework, LiveKit, or a conventional blocking retrieval
pipeline unless a piece of work is explicitly infrastructure, not cognition.

This file describes MoshiMind as a product: what it does today, what's proven versus
still assumed, and where it's headed. For exact per-feature status and engineering
detail, see [`PHASES.md`](PHASES.md) (the living roadmap with a status table and a
detailed write-up under every item) and [`RECAP.md`](RECAP.md) (narrative handoff
context for a new engineer or a fresh session picking this up). For the original
upstream MoshiRAG paper/architecture description, see [`README.md`](README.md).

---

## What it does today

A live conversation with MoshiMind, right now, genuinely does the following — each of
these has been exercised end-to-end against real audio on real GPU hardware, not just
unit-tested against fakes:

- **Talks in real time, full-duplex**, like base Moshi — it listens and speaks
  simultaneously rather than turn-by-turn.
- **Notices when it needs outside knowledge mid-response**, predictively (from the
  user's partial speech, before it's even finished asking) as well as reactively (via
  its own `rag_token_id` emission), and races the two so a good guess is never wasted.
- **Grounds its answer** by sending the conversation context to a retrieval LLM (any
  OpenAI-compatible endpoint — OpenAI, Groq, a local vLLM/Ollama server), scoring the
  answer's confidence, and only conditioning generation on it if that confidence
  clears a threshold — a bad or empty retrieval doesn't get to distort the response.
- **Remembers within and across turns**: a bounded working-memory window, a real
  SQLite-backed episodic/semantic store for durable facts, and a merge policy that
  combines memory and fresh retrieval by whichever is more confident rather than
  picking one and discarding the other.
- **Notices when it's being interrupted**, using VAD history, the partial transcript,
  and a real acoustic signal (pitch/energy trend from raw PCM, not a trained model) to
  tell a genuine barge-in apart from a backchannel ("mhm") or a hesitation — and mutes
  its own audio output immediately on a real interruption.
- **Degrades gracefully**: a dead retrieval LLM, an unreachable reference encoder, or
  a full-on `CUDA out of memory` during setup all fail safely — logged, contained,
  never taking the whole conversation (or the whole server) down with them.

## What's proven versus what's still assumed

Until recently, none of the above had ever actually been executed — it was built and
unit-tested in an environment with no GPU and no torch, then reasoned through
carefully against the real tensor shapes and APIs. That changed this session: on a
consumer 8GB GPU (RTX 5050), the real checkpoint doesn't fit in bf16, so it's now
loaded with int8 quantization (a dormant utility already in the codebase, wired up
and verified), and a full real conversation — real synthesized speech in, real
retrieval, real interruption signal, real generation out — has been run against the
live server. That run is what confirmed the bullets above are real, not aspirational.

**Two things are still genuinely open, not by design choice but by external
constraint or hardware limit**, and are worth knowing about before treating this as
production-ready:

- **Grounding quality depends on a gated model.** The reference-encoder step that
  turns retrieved text into a conditioning tensor needs `meta-llama/Llama-3.2-3B-Instruct`,
  which requires requesting and being granted access on Hugging Face. Without it,
  retrieval still runs and produces a real answer, but it can't be encoded into the
  model's conditioning stream — the conversation degrades gracefully rather than
  using the reference, exactly as designed, but grounding isn't fully realized.
- **Quantized inference is slow relative to real-time on modest hardware** — measured
  at roughly 7-30x slower than the ~80ms/frame real-time budget on the 8GB card this
  was validated on. The cognitive logic runs correctly at this speed; it just isn't
  yet a real-time conversational experience on this class of hardware. Getting there
  needs either a bigger/faster GPU, a better quantization scheme, or (the biggest,
  most delicate remaining piece of engineering here) making the shared batched
  generation step loop itself faster and non-blocking — see Phase 4.3/Phase 8 in
  `PHASES.md`.

## Where this is headed

The priority order hasn't changed since the project's founding direction-setting —
what's changed is how much of it is now provable, not just designed:

1. 🟢 **Native Moshi + RAG integration** — done and live-validated: confidence-weighted
   conditioning, predictive/speculative retrieval, multi-shot bounding, telemetry.
2. 🟢 **Streaming/asynchronous cognition (the sidecar)** — done: a generic
   urgency-scheduled, circuit-broken dispatcher, with RAG and memory as its two real
   producers.
3. 🟡 **Long-term memory** — built and live-wired into the conversation path; not yet
   independently verified end-to-end in a live multi-turn conversation (this session
   validated RAG and interruption live; memory recall specifically is next).
4. 🟡 **Full-duplex interruption/anticipation** — the perceptible half (muting audio
   the instant a real barge-in is detected) is done and live-validated. The deeper
   half — actually stopping the shared transformer from computing more tokens for an
   interrupted response, not just muting its output — is the highest-blast-radius
   piece of remaining work (it touches the batch loop shared across every connected
   channel) and is now buildable, since a GPU is finally available to verify it
   safely.
5. 🟠 **Prosody, emotion, conversational behavior** — real acoustic signal (pitch/
   energy trend) is extracted and already feeds the interruption classifier; feeding
   that same signal into retrieval urgency and confidence scoring is the natural next
   small step.
6. 🔲 **Context management** — bounding the raw conversation-context accumulator with
   the already-built working-memory window is designed but not yet swapped in.
7. 🟡 **Reliability / fault isolation** — the pattern (contain a failure, log it,
   keep the conversation going) is established and has now been proven against a real
   failure (a genuinely unreachable reference-encoder service), not just a simulated
   one.
8. 🔲 **Performance optimization** — was entirely blocked without a GPU; now the
   single most concrete, well-measured piece of unblocked work (see "quantized
   inference is slow," above).
9. 🔲 **Developer architecture / API** — documenting the cognitive-sidecar interface
   and building the intended benchmark suite, once the above stabilizes.
10. ⚪ **Generic voice-agent plumbing** — explicitly deprioritized; only touched if it
    blocks something above.

**The single highest-leverage next step**, in short: get real, sustained multi-turn
conversations running (ideally on better GPU hardware, and with gated model access
sorted) to validate memory recall and interruption timing the way RAG grounding was
just validated — everything downstream of that (performance work, the deeper
interruption hook, context management) is easier to prioritize correctly once that's
in hand.
