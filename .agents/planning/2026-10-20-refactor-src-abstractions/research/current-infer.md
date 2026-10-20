# current-infer.md

Walk of `src/strands_decider/infer.py` (575 LOC) — what it does today and where each piece lands in the target architecture.

## File-level responsibilities

The module mixes four concerns:

1. **KV cache forking** — module-level, backend-coupled (transformers 5 layered cache layout, handles Gated DeltaNet).
2. **`EngineConfig`** — serving-time knobs.
3. **`SystemOneEngine`** — one request → one response, with shared-prefix and batched paths.
4. **Factory + MLX dispatch** — `load_engine`, `mlx_available`, `load_mlx`.

## `EngineConfig`

Serving-only knobs:

```
device, use_prefix_cache, max_batch, model_name,
max_question_fraction, strict_window, max_batch_tokens
```

Nothing architectural (dtype/attention/etc. live on `StrandsDeciderConfig`). **Target home:** stays an `EngineConfig` dataclass in `engine.py`; cross-request batching flags (`batching`, `batch_wait_ms`, `batch_tune`, `max_batch_tokens` as a batcher cap) added by PR #88 join it.

## KV cache forking (`_fork_layered_cache`, `_expand_cache`, `UnforkableCache`, `_ROW_STATES`, `_SHARED_TENSORS`)

- Operates on transformers-5 layered cache (`cache.layers` is a list of per-layer objects).
- Row-dim tensors expanded by `n`: `keys`, `values`, `conv_states`, `recurrent_states` (DeltaNet).
- Shared per-layer tensors passed through unchanged: `_sliding_window_tensor` (Gemma 4 sliding window).
- Any other tensor on a layer → `UnforkableCache` (caller falls back to batched encoding).
- `.contiguous()` is non-negotiable — the suffix forward writes into these tensors; a stride-0 view would error.
- DeltaNet layers mutate their states and `has_previous_state` flag in place, so each fork must get its own layer object + dict + tensors (not shared refs).

**Target home:** `TorchServingModel` internals. The MLX backend has its own cache representation and never calls these. **Not** on the Scheduler — this is backend-specific cache logic.

## `SystemOneEngine.__init__`

Four things wired here:

1. **Kernel install** (`mps_kernels.install()` or `cpu_kernels.install_cpu()`) when device is MPS/CPU. Target: `TorchServingModel.load` picks a kernel install at construction.
2. **Model to device + eval** (`.to(device).eval()`). Target: `TorchServingModel.load`.
3. **CPU torso upcast** (bf16/fp16 → fp32 under `inference_mode`). Target: `TorchServingModel.load`.
4. **`_prefix_cache_unforkable`** — sticky-false flag so later requests skip the shared-prefix path after the first `UnforkableCache`. Must be preserved; today's monotonic-false is race-safe without a lock.
5. **`_tok_lock`** — HF fast tokenizer reconfigures truncation/padding state per call; raising "Already borrowed" on two concurrent threads. Target: belongs on whatever holds the tokenizer (`TorchServingModel`).

## `_fit` (window fitting)

**Rule:** question gets first claim on the window; state truncates from the front with `truncation=True, max_length=state_budget`; question front-truncates by `max(0, len(x) - reserve)` so the tail (options + `<answer>` marker) is preserved; `strict_window` raises instead of truncating.

**Why it matters structurally:** this is the one piece of prep that depends on the model's `max_length` and the tokenizer. Can live cleanly inside `Engine.prep(req, model)`. Returns `(state_ids, question_ids, offsets)`. The offsets are returned (not stored on the engine) explicitly so concurrent requests on one engine don't share state — this is already the right discipline for a Scheduler-friendly refactor.

## `_option_idx`, `_pad`

Both pure tensor ops on `self.device`. Build `opt_idx [B,K]` with `-1` padding past each row's option count; `_pad` right-pads token rows + attention mask. **Target home:** `strands_decider.ops` or inline on `TorchServingModel`; MLX backend has its own equivalents.

## `_slot_probs_batched` (the fallback / sole path for 1-question requests)

- Pads `[s + qi for qi in q]` to a rectangle.
- Builds `opt_idx` **offset by `len(s)`** (whole prompt, so option positions are state-relative).
- Calls `self.model(input_ids, attention_mask, n_slots, temperature, opt_idx)` — i.e. the **training `forward`** of `StrandsDeciderModel` (!), which runs encode → pool_last → head → temperature → masked_log_softmax.
- Returns `log_probs.exp()` and summed real tokens.

**Observation:** this path routes serving through `StrandsDeciderModel.forward` — the same forward training uses. That is the single-implementation win the refactor is after; it already exists for this path. The shared-prefix path below breaks that discipline.

## `_slot_probs_shared_prefix` (N>1 questions, prefix path)

- Encodes state through `self.model.torso` directly (bypasses `StrandsDeciderModel.forward`).
- Forks cache `m` times with `_expand_cache`.
- Encodes suffix through `self.model.encode(...)` (torso only; no head).
- Pools last of suffix with mask-aware `pool_last_token` (mask is longer than hidden — covers the cached prefix positions).
- Head applied manually: `gather_options`, pointer vs slot branch, `self.model.head(...)`, then `apply_temperature`, `masked_log_softmax`.

**Divergence from the single-forward discipline:** this path **does not** go through `StrandsDeciderModel.forward`. It inlines the head application. If `forward` ever grows a step the serving path misses, this path drifts. Noted for the design doc — the refactor should route this through the model's own head application too (either as `TorchServingModel.read(hidden, opt_idx, kinds, temperatures)` or by routing through `forward`).

**opt_idx base:** 0 here — `hidden` is suffix-only.

## `evaluate` (the public method)

Pipeline:
1. Reject images (`request.images` → ValueError). Target: Engine checks `model.handles_images` instead.
2. Render each question (`render_question(q)`), check slot ceiling (pointer head skips this check — unbounded options).
3. Render state (`render_state(request.state)`).
4. For each `_chunk` (slice, state_ids, question_ids, offsets):
   - If `use_prefix_cache` and not sticky-unforkable and `>1` questions → try `_slot_probs_shared_prefix`; on `UnforkableCache`, set sticky flag and fall through to batched.
   - Else `_slot_probs_batched`.
   - Each row's probs → `_to_answer` (dispatched by `rq.kind`).
5. Return `SystemOneResponse(answers, Usage(input_tokens=total, output_tokens=len(names)))`.

Target split:
- **Rendering + slot check**: `Engine.prep`.
- **Chunking**: `Engine.prep` or `Engine.infer` (pure Python, no device).
- **Shared-prefix branch**: `TorchServingModel.ask(state, questions)` — the model owns this.
- **Batched single-chunk**: `scheduler.submit(prepared_rows)` → `model.forward_rows(...)` → read.
- **`_to_answer`**: shared, backend-neutral; lives on the Engine or a response helper.

## `_chunks`

Honours `max_batch` AND `max_batch_tokens`. For each group of `max_batch` questions, fits them once, then cuts into sub-chunks where `(rows) × (len(s) + max_question_len)` would exceed `max_batch_tokens`. Target: pure Python; moves into `Engine.prep` or a chunking helper. Scheduler never sees this — it's single-request-level chunking, orthogonal to cross-request batching.

## `ask`

Convenience wrapper for `evaluate(SystemOneRequest(state, questions))`. Keep on the model as `TorchServingModel.ask`.

## `_option_token_index` (module-level)

Picks the **last** token overlapping an option's character span (the one that has read the whole option under causal attention). Raises if truncation ate all tokens of an option. Target: shared helper in `ops.py` or `prompting.py`.

## `_to_answer` (module-level)

Dispatches by `rq.kind`:
- `noul` → `NoulAnswer(noul=read_noul(...))`
- `choice` → `ChoiceAnswer(choice, probabilities, confidence=derive_confidence(...))`
- `score` → `ScoreAnswer(score, legend=score_legend(...), probabilities, confidence=derive_score_confidence(..., ordinal_smoothing))`

Target: backend-neutral; stays either here or moves into `prompting.py` with the readouts (PR #77 already moved readouts into `prompting.py` — this fits there).

## `load_engine`, `mlx_available`, `load_mlx`

- `load_engine` dispatches `device == "mlx"` to `load_mlx`, else constructs `SystemOneEngine(StrandsDeciderModel.load(ckpt), EngineConfig(...))`.
- `mlx_available` checks Darwin+arm64 + `mlx.core` + `mlx_lm` importability.
- `load_mlx` imports `.mlx_engine.load_mlx_engine` deferred.

Target: `make_engine(checkpoint, device, config)` factory in `engine.py` picks:
- device == "mlx" → `MlxServingModel.load(ckpt)` + NoScheduler/Serialise
- device.startswith("mps") → `TorchServingModel.load(ckpt, device="mps")` + SerialiseScheduler
- device in {"cuda","cpu"} → `TorchServingModel.load(ckpt, device=...)` + MicroBatcher (if cuda+config.batching) else NoScheduler

Lambda + `hidden_engine` later: `LlamaCppServingModel.load(ckpt)` + NoScheduler.

## Target module split

| Current in `infer.py` | New home |
|---|---|
| `_fork_layered_cache`, `_expand_cache`, `UnforkableCache`, `_ROW_STATES`, `_SHARED_TENSORS` | `TorchServingModel` internals (private module helpers or `strands_decider.torch_cache`) |
| `EngineConfig` | `strands_decider.engine` |
| `SystemOneEngine.__init__` (kernel install, device, upcast, lock) | `TorchServingModel.load` + `TorchServingModel.__init__` |
| `_upcast_torso_for_cpu` | `TorchServingModel.load` |
| `_temperatures` | `TorchServingModel` method |
| `_fit`, `_option_idx`, `_pad` | `Engine.prep` (orchestrates) + shared helpers in `ops.py` |
| `_slot_probs_batched` | `scheduler.submit` → `model.forward_rows` + model.read |
| `_slot_probs_shared_prefix` | `TorchServingModel.ask` (model owns shared-prefix) |
| `evaluate` | `Engine.infer` |
| `_chunks` | `Engine.prep` helper |
| `ask` | `TorchServingModel.ask` |
| `_option_token_index`, `_to_answer` | `prompting.py` (PR #77 direction) |
| `load_engine`, `mlx_available`, `load_mlx` | `strands_decider.engine.make_engine` factory |

## Surprises / gotchas for the refactor

1. **The batched path already goes through `StrandsDeciderModel.forward`.** The training forward is used for serving. Keep this. The shared-prefix path diverges — the refactor should either unify (route both through a `TorchServingModel.forward_rows` + `model.read`) or carefully preserve the second code path.
2. **Sticky `_prefix_cache_unforkable` flag.** Must survive the refactor as scheduler-independent model state.
3. **`_tok_lock` is per-engine-instance.** When ServingModel owns the tokenizer, the lock moves with it.
4. **CPU bf16 upcast.** `inference_mode()` matters here — if load code moves, keep the context.
5. **Serving-time attn impl swap (`serving_attn_implementation`).** Gemma 4 E2B needs eager attention on the shared-prefix path. This is already in `modeling.py` (`load`), and must be called after torso load but before serving — training skips it.
