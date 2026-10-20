# current-mlx.md

Walk of `src/strands_decider/mlx_engine.py` (278 LOC). The MLX backend as a `SystemOneEngine` subclass that bypasses the parent's `__init__`.

## The trick

`MLXEngine.__init__` **does not** call `super().__init__`, because the parent moves a torch torso to a torch device. Instead it sets fields the inherited helpers (`_fit`, `_option_idx`, `_pad`, `_temperatures`, `evaluate`, `_chunks`) will read:

```
self.cfg = config
self.model = readout            # NOT a StrandsDeciderModel — a _Readout
self.tok = readout.tokenizer
self.device = "cpu"             # where the head and small index tensors live
self._decoder = decoder         # the MLX decoder
self._cache_owner = cache_owner # module that makes the MLX prompt cache
self._lock = threading.Lock()   # one evaluate at a time
self._prefix_cache_unforkable = False
self._tok_lock = threading.Lock()
```

**`_Readout`** is a plain object with `config`, `tokenizer`, `head` — the smallest surface the inherited code actually calls on `self.model`. It is already, in miniature, exactly what the `ServingModel` Protocol captures.

In the target shape, `MlxServingModel` replaces `_Readout` + `MLXEngine` and implements the Protocol directly. The subclassing-with-manual-`__init__` dance disappears.

## `evaluate` (overridden)

```python
with self._lock:
    return super().evaluate(request)
```

MLX's lazy arrays belong to the thread that created them; `_lock` serialises requests on the server's thread pool.

**Target:** `SerialiseScheduler` provides this discipline externally — one lock, one forward at a time. `MlxServingModel.supports_batch_rows = False`; factory picks `SerialiseScheduler`.

## `_hidden(rows, cache=None)`

- Right-pad with `pad_id`.
- `ids = mx.array([...], dtype=mx.int32)`.
- `hidden = self._decoder(ids, cache=cache)`.
- `mx.eval(hidden)` — forces evaluation on this thread.
- **No attention mask** — both layer kinds in Qwen3.5 are causal; right-padding means no real position reads a pad.

Target: `MlxServingModel.forward_rows(ids, mask, cache=None)` runs the same code; the mask kwarg is accepted for interface uniformity but ignored (document this).

## `_probs(hidden, last, opt_idx, n_slots, kinds)` — the backend boundary

This is where the design's claim ("head stays torch everywhere") is realised in the current code:

1. Compute flat positions for each row's `last` token + each row's `opt_idx` entries.
2. Reshape `hidden` to `[rows*width, dim]` and gather at those positions on MLX.
3. Convert the gathered `[len(positions), dim]` MLX fp32 array to a torch CPU tensor via `_to_torch` (through the buffer protocol — a `memoryview` copy).
4. On torch, under `inference_mode`: run `self.model.head(...)`, apply temperature, masked log-softmax, exponentiate.

The MLX side only extracts the specific hidden rows the head needs; the heavy `[B, L, d]` tensor never crosses the boundary. The head is torch fp32 on CPU — **the same `PointerHead` object** a torch backend would use.

Target: this is the **structural anti-drift anchor from Q7**. `MlxServingModel.read(hidden_mlx, option_positions)` gathers on MLX, converts to torch via `_to_torch`, calls the shared `apply_pointer_head(hidden_torch, positions, self.pointer_head)` — same code the torch backend calls.

## `_slot_probs_batched`, `_slot_probs_shared_prefix`

Both override `SystemOneEngine`'s equivalents and share the same shape with their torch siblings, with two MLX specifics:

- **Shared prefix:** `make_prompt_cache(self._cache_owner)` instead of torch cache; encode state; **`type(layer).merge([layer] * len(questions))`** replicates the cache across the batch. This is MLX's cache-fork equivalent — a cleaner mechanism than torch's `_fork_layered_cache` (which has to re-copy fields manually because transformers 5 doesn't expose a `merge`).
- **Batched (fallback):** `ids = state + question` per row, forward, read `last = len(row) - 1`.

Target: `MlxServingModel.ask(state, questions)` runs the shared-prefix path (model owns per-request cache). `MlxServingModel.forward_rows` is the batched primitive the (irrelevant for MLX) Scheduler would call — but `supports_batch_rows=False` means it only ever gets one row at a time.

## `_to_torch(array)`

```python
return torch.frombuffer(bytearray(memoryview(array)), dtype=torch.float32)
```

Buffer-protocol copy. Zero-copy would require shared ownership, which `torch.frombuffer` + `memoryview(mlx_array)` doesn't give; the `bytearray` makes it a real copy. Cheap (a memcpy of fp32 bytes for a few rows × `dim`).

Target: shared `strands_decider.torch_internals.mlx_to_torch` helper.

## `merge_lora(lm, adapter_dir, prefix)` — the LoRA merge-at-load trick

Torch keeps LoRA unmerged (PEFT applies the delta per forward). MLX folds it into the base weights once at load:

- Only plain linear `lora_A/lora_B` pairs are supported. Refuses:
  - DoRA (`fan_in_fan_out`)
  - rank/alpha patterns
  - embedding/bias LoRAs
  - saved modules, trainable tokens
- PEFT names `base_model.model.<path>` → mlx-lm names `<prefix><path>.weight`. Prefix is `"model."` for text-only, `"language_model.model."` for multimodal.
- Merge formula: `W + scale * B @ A`, where `scale = alpha / rank` (or `alpha / sqrt(rank)` for rslora).
- fp32 on CPU (`mx.stream(mx.cpu)`), rounded once back to the base weight's dtype.
- Weights swapped in one at a time to bound peak memory.

**Measured fidelity (from the module docstring):** v19 on fp32 CPU, 54 answers, no answer changes, max probability difference 0.0138 (vs 0.0051 on MPS). Most of the gap is the merge's rounding; MLX with merged LoRA matches MPS with merged LoRA within 0.0036.

Target: `strands_decider.torch_internals` doesn't apply here — this is MLX-internal. Lives in `MlxServingModel._merge_lora` or `strands_decider.mlx_internals.merge_lora`. **Keep the strict refusal list** — silently skipping an unsupported adapter type would ship wrong weights.

## `_load_torso(base, dtype)` — MLX torso construction

- Reads `base/config.json` for `model_type`.
- Overrides `qwen3_5_text` → `qwen3_5` (mlx-lm names the text-only type after the multimodal one).
- `load_model(base, model_config=overrides)` from mlx-lm.
- For multimodal checkpoints: `owner = lm.language_model`, prefix `"language_model.model."`; else `owner = lm`, prefix `"model."`.
- Casts floating weights to the target dtype, except those `cast_predicate` keeps in fp32 (e.g. `A_log` in Gated DeltaNet).
- `mx.eval(lm.parameters())` to materialise lazy arrays on this thread.

Target: `MlxServingModel._load_torso` (private, moves to `mlx_model.py`).

## `load_mlx_engine`

- Reads config/head/tokenizer/base via the shared `checkpoint_dir`, `config_path`, `base_revision`, `load_head_state`.
- Sets `mx.set_cache_limit(DEFAULT_CACHE_LIMIT)` — 1 GiB; prevents MLX from retaining up to 95% of RAM.
- Loads base torso via `_load_torso` with `checkpoint_dir(base_model, base_revision)`.
- Merges LoRA if `use_lora`.
- `mx.clear_cache()` — drops the merge's fp32 transients.
- Builds torch head via `build_head(config, hidden_size)`; loads head state; `.to(fp32).eval()`.
- Wraps in `MLXEngine(decoder, owner, _Readout(config, tokenizer, head), EngineConfig(device="mlx"))`.

Target: `MlxServingModel.load(checkpoint)` runs this; `make_engine` picks `SerialiseScheduler` for MLX.

## Target module split for mlx_engine.py

| Current piece | New home |
|---|---|
| `_Readout` class | Deleted — replaced by `MlxServingModel` holding `config`, `tokenizer`, `head` directly |
| `MLXEngine.__init__` field setup | `MlxServingModel.__init__` |
| `MLXEngine.evaluate` + `_lock` | Deleted — `SerialiseScheduler` provides the discipline |
| `_hidden` | `MlxServingModel.forward_rows` |
| `_probs` (gather + convert + head) | `MlxServingModel.read` (and shared `apply_pointer_head` on the torch side) |
| `_option_positions` | Shared `_option_idx` on the base ServingModel helper (or inline) |
| `_slot_probs_batched` | `MlxServingModel.forward_rows` + `read` composition |
| `_slot_probs_shared_prefix` | `MlxServingModel.ask` (model owns shared-prefix) |
| `_to_torch` | `strands_decider.torch_internals.mlx_to_torch` |
| `merge_lora` | `MlxServingModel._merge_lora` (private) or `mlx_internals.merge_lora` |
| `_load_torso` | `MlxServingModel._load_torso` (private) |
| `load_mlx_engine` | `MlxServingModel.load` + factory `make_engine` wires `SerialiseScheduler` |
| `DEFAULT_CACHE_LIMIT`, `_DTYPES`, `_MODEL_TYPES` | Private module constants of `mlx_model.py` |

## Surprises / gotchas for the refactor

1. **The subclass-with-no-super init is already a workaround.** It exists because `_Readout` isn't a `StrandsDeciderModel`. The refactor's `MlxServingModel` is a first-class peer of `TorchServingModel` under the Protocol — the hack vanishes.
2. **`_Readout` is proto-`ServingModel`.** The attributes it exposes (`config`, `tokenizer`, `head`) are already the shared read surface. The refactor formalises this.
3. **MLX's own cache fork (`type(layer).merge`) is cleaner than torch's.** Each backend keeps its own cache mechanism in its own class — don't try to unify.
4. **Head is torch, on CPU, fp32.** The one `PointerHead` object is shared with torch backend. The refactor keeps this.
5. **MLX serialises forwards (`_lock`).** Translates to `SerialiseScheduler`, not `NoScheduler` — the device thread is a shared resource across requests even when there's no batching.
6. **`mx.set_cache_limit` is process-wide and critical.** 1 GiB cap prevents MLX from eating most of RAM. The refactor must preserve this at `MlxServingModel.load` time, with the same escape hatch (`cache_limit_bytes=None`).
7. **LoRA merge strictness.** The refusal list exists for correctness. Preserve it exactly.
8. **`load_model` dtype casting respects `cast_predicate`.** Keeps `A_log` in fp32. Don't collapse to a plain "cast everything" when porting.
9. **Positions on prefix for multimodal MLX**: not yet implemented — this file is text-only on MLX today. User already stated MLX+Vision is a future target; the Protocol fix (drop torch-tensor return type) accommodates it.
