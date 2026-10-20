# current-modeling.md

Walk of `src/strands_decider/modeling.py` (746 LOC). Five concerns bundled; this is the file the refactor splits most.

## Constants

- `DEFAULT_NUM_SLOTS = 24` — must match the corpus's largest option count (training data pipeline uses this).
- `MASK_VALUE = -1e4` — not `-inf`, to keep softmax finite under autocast/bf16 when a row is fully masked. Shared across backends via `masked_log_softmax`.
- `GEMMA4_CLASSES` — model_type → transformers class for the multimodal wrapper whose loader maps the text decoder's weight names (`gemma4` and `gemma4_unified` both need their vision+audio encoders discarded after load).

## `serving_attn_implementation(torso_cfg)`

Gemma 4 E2B (1 kv head) + torch 2.7 CUDA SDPA + shared-prefix path with 64k+1 query tokens = **wrong** attention output (a yes/no answer measures 0.726 vs 0.860 reference). Fix: `"eager"` attn. Every other torso: `None` (default).

Called by `load` on serving (not training). Target home: shared helper, called by `TorchServingModel.load`.

## `StrandsDeciderConfig` (dataclass, 23 fields)

Grouped by concern:

| Group | Fields |
|---|---|
| **Base model** | `base_model`, `base_revision` |
| **Head architecture** | `num_slots`, `head_hidden`, `head_dropout`, `head_type` (slot|pointer), `pointer_dim`, `head_init` ("random" | "lm_head") |
| **Window** | `max_length` |
| **Precision** | `torch_dtype` |
| **Calibration** (post-training) | `temperature`, `temperature_by_kind`, `ordinal_smoothing` |
| **Training-only knobs** | `kl_frozen_weight` (KL regularizer), `use_lora`, `lora_r/alpha/dropout/targets` |
| **Model-specific** | `force_bos` (Gemma 4-it), `host_embeddings` (Gemma 4 E2B) |

**Legacy guard:** rejects `full_weight_targets` from older Gemma 4 research builds.

**Target split:** this is the architecture + training recipe. Stays as one `StrandsDeciderConfig` in `strands_decider.config` since both serving and training read it. SlotHead-only fields (`num_slots`, `head_init` == "lm_head") become dead with the slot head removal — the design doc should flag them for deletion.

**Scope cleanup implied by Q6 (A6):** when `SlotHead` is deleted, `head_type` has one value (`pointer`); either drop the field entirely or keep it for legacy checkpoint loading with a `"slot"` → error path.

## `HostEmbedding(nn.Module)`

- Wraps a frozen `nn.Embedding`.
- Keeps weights in CPU memory, pinned when CUDA is available.
- Forward: ids → CPU → embed → pin → `.to(device, non_blocking=True)`.
- `_apply`: dtype casts applied on CPU; device moves dropped so the table stays off the GPU.

**Why:** Gemma 4 E2B's per-layer embedding table is 262,144 × 8,960 (2.35B params, 4.4 GiB bf16). Read-only, never trained; moving to CPU frees GPU memory at no measured speed cost.

**Scope:** torch-only. Target home: shared tool for the torch backend, used by `TorchServingModel._load_torso` when `config.host_embeddings` is set.

## `ensure_bos(tokenizer)`

Rewrites a fast tokenizer's post-processor to prepend BOS whenever `add_special_tokens=True`. For gemma-4-E2B-it, whose tokenizer leaves BOS to the chat template; without BOS the frozen torso reads JevBench at chance.

**Target:** shared helper used at both train-time and serve-time; works with the HF tokenizer either backend holds. Lives in a shared `tokenization.py` or stays with the model that constructs the tokenizer.

## `SlotHead` (to be deleted)

v1-v6 readout. LayerNorm → optional dropout → Linear or 2-layer MLP → `num_slots` logits.

Confirmed deletable per Q5/A5. Still referenced by `training/tests/test_variance.py`.

## `PointerHead`

v19 readout, **kept**. Single attention score:
- `self.q(norm(decide))` → `[B, dim, 1]`
- `self.k(norm(options))` → `[B, K, dim]`
- `(o @ d).squeeze(-1) * dim**-0.5` → `[B, K]`

LayerNorm **before** both query and key — not in the reference impl, needed because the torso's final hidden states aren't scaled for a bare dot product.

**Measured vs slot head:** +0.063 at dp=256, +0.058 at dp=16; a 2.1M-param MLP slot head reaches only +0.032. The win is addressing, not capacity.

Target: `strands_decider.heads.PointerHead` — one torch `nn.Module`, imported by `TorchServingModel`, `MlxServingModel` (via `_Readout`), and `TrainingModel`. The one-shared-class anti-drift anchor.

## `build_head(config, hidden_size)`

Dispatch SlotHead vs PointerHead; always `.to(torch.float32)` ("tiny head, low-precision classifier is a needless source of calibration error").

Post-SlotHead-removal: becomes a one-liner constructing `PointerHead`. Or inline into `TorchServingModel.__init__`.

## Shared tensor ops

| Op | Shape in → out | Target home |
|---|---|---|
| `gather_options(hidden, opt_idx)` | `[B,L,d]` + `[B,K]` → `[B,K,d]` | `ops.py` |
| `pool_last_token(hidden, mask)` | `[B,L,d]` + `[B,M]` where `M >= L` → `[B,d]` | `ops.py` |
| `masked_log_softmax(logits, n_slots)` | `[B,K]` + `[B]` → `[B,K]` log-probs with `-inf` past n_slots | `ops.py` |
| `apply_temperature(logits, t)` | scalar or `[B]` tensor | `ops.py` |

All torch. Four pure functions. The MLX backend doesn't use these — it would have numpy/MLX equivalents OR convert hidden to torch and call these. Current `MLXEngine` does the latter via `_Readout`.

## `StrandsDeciderModel(nn.Module)` — the 325-LOC class

This is the thing that becomes `TorchServingModel` + `TrainingModel`.

### Composition
```
.config: StrandsDeciderConfig
.torso:  nn.Module          # PEFT-wrapped if use_lora
.tokenizer: HF AutoTokenizer
.head:    nn.Module         # SlotHead | PointerHead, fp32
```

### `hidden_size(torso)` (staticmethod)

Reads `torso.config.hidden_size | n_embd | d_model`. Target: shared helper.

### `from_pretrained_base(config, device_map, attn_implementation)` (classmethod)

Load tokenizer → pad token fallback → maybe `ensure_bos` → `_load_torso` → attach LoRA (if use_lora). **Training-only entry point.** Target: moves to `TrainingModel`. Serving-side `TorchServingModel` never calls this — it calls `load`.

### `_load_torso(config, device_map, attn_implementation)` (staticmethod)

Three branches:

1. **Qwen3.5 / Qwen3.5-text** → `transformers.Qwen3_5ForCausalLM.from_pretrained(... text_config ...)`, keep `.model` (decoder). AutoModel would hand back a wrapper with a vision tower.
2. **Gemma 4** (`gemma4` or `gemma4_unified`) → `getattr(transformers, GEMMA4_CLASSES[...]).from_pretrained(...)`, keep `.model.language_model`, discard vision+audio (0.47B params). Gemma4ForCausalLM alone leaves weights randomly initialised with only a warning.
3. **Else** → `AutoModel.from_pretrained(...)`.

Then: swap `embed_tokens_per_layer` for `HostEmbedding` if `config.host_embeddings`. Set `torso.config.use_cache = True`.

**Target:** `TorchServingModel._load_torso` (same code, same branches). The multimodal unwrap logic is architecture-specific and belongs with the torch backend. Vision support: when the user wants images, the full-model load keeps the vision tower too (today's `VisionDeciderModel._load_torso` is a separate variant — see current-vision.md).

### `is_hybrid(torso)` (staticmethod)

Detects linear-attention layers (Qwen3.5 Gated DeltaNet). Called by `infer.py`'s cache fork (actually, not directly — `_fork_layered_cache` handles it generically via `_ROW_STATES`). Target: still useful helper for logging/debug; keep in a shared place.

### Training-only methods

| Method | What it does |
|---|---|
| `attach_lora()` | PEFT `LoraConfig(..., task_type=FEATURE_EXTRACTION)` on torso |
| `freeze_torso()` | `requires_grad_(False)` on all torso params |
| `trainable_parameters()` | `(trainable, total)` param counts |
| `init_head_from_lm_head()` | Seeds slot head rows from LM output rows for digits "1"-"9", scaled to match head init std; **slot-head only** (will be dead code post-cleanup) |
| `_output_embedding()` | Returns the tied-embedding matrix (Qwen3-style); asserts `tie_word_embeddings`; serves as the LM head for `init_head_from_lm_head` and `frozen_slot_log_probs` |
| `frozen_slot_log_probs(...)` | Adapter-disabled torso forward, reads option-number tokens via `_output_embedding` with optional Gemma `final_logit_softcapping`; returns `(log_probs, eligibility_mask)` for rows with single-token option numbers |
| `forward(...)` | **Training forward** (loss computation); see below |

**Target:** all these move onto `TrainingModel` (which extends `TorchServingModel`). `init_head_from_lm_head` and `frozen_slot_log_probs` are slot-head-only; after SlotHead removal, they likely delete or lose most of their body.

### `encode(input_ids, attention_mask, past_key_values=None)`

Thin wrapper over `self.torso(..., use_cache=past_key_values is not None, return_dict=True)` → `last_hidden_state`. Used by both the shared-prefix path (passing `past_key_values`) and `frozen_slot_log_probs` (no cache).

**Target:** `TorchServingModel.encode` — serving-side method, shared by `TorchServingModel.forward_rows` and `TorchServingModel.ask` (shared-prefix).

### `forward(input_ids, attention_mask, n_slots, labels?, label_dist?, weights?, past_key_values?, temperature?, opt_idx?)`

Does serving AND training in one method:
1. `hidden = self.encode(...)`
2. `pooled = pool_last_token(hidden, mask).to(float32)`
3. Pointer: `gather_options` → `self.head(pooled, options)`; Slot: `self.head(pooled)`
4. `apply_temperature`, `masked_log_softmax`
5. If labels/label_dist/weights given → add loss (NLL or soft-target cross-entropy with weighted-mean reduction)

**Target split:**
- **Serving path (no loss):** becomes `TorchServingModel.forward_rows(ids, mask, cache?) → hidden` and `TorchServingModel.read(hidden, opt_idx) → log_probs` (plus temperature + mask). Pure-model, no scheduler.
- **Training path (loss):** `TrainingModel.forward(batch) → loss` composes those two + loss computation.

### `save_pretrained(path)`

Writes: `strands_decider_config.json`, `slot_head.pt` (the head's state dict — misnamed; works for both heads), LoRA adapter under `lora/`, tokenizer files.

**Observation:** the filename `slot_head.pt` is a legacy name even for pointer heads. Not blocking, but a design-doc note (either rename or keep legacy for back-compat on existing checkpoints).

**Target:** moves to `TrainingModel`. `TorchServingModel.load` reads this file layout.

### `load(path, device_map?, attn_implementation?, trainable=False, host_embeddings?)` (classmethod)

Serving-time entry point:
1. `checkpoint_dir(path)` — local dir or Hub download via `snapshot_download`.
2. Config from JSON (`config_path` picks `strands_decider_config.json` or legacy `hobson_config.json`), `base_revision` from provenance.json fallback.
3. Override `host_embeddings` if caller passed one.
4. Guard: if `use_lora` and `lora/` dir missing → `FileNotFoundError` (prevents silently loading the base and giving different probabilities).
5. Load head state (`load_head_state` — accepts `head.safetensors` from HF export OR `slot_head.pt`; refuses both present).
6. Tokenizer + pad token + maybe `ensure_bos`.
7. `_load_torso` + optional `set_attn_implementation(serving_attn_implementation(...))` on serving path.
8. `PeftModel.from_pretrained(torso, lora_dir, is_trainable=trainable)` if use_lora.
9. **Bypass `__init__`** (`cls.__new__ + nn.Module.__init__`) because __init__ attaches LoRA again. Build fields manually, build head, load head state, cast head to fp32.

**Target:** `TorchServingModel.load(path, device, attn_implementation?, host_embeddings?)`. The `trainable=True` branch moves to `TrainingModel.load_for_resume(path)`.

**Observation:** the __init__-bypass hack is coupled to the fact that `__init__` has training-only side effects (attach_lora). Splitting Training/Serving removes the hack naturally: `TorchServingModel.__init__` has no LoRA attach; the `load` can use normal construction.

## Module-level file I/O helpers

| Function | What it does | Target home |
|---|---|---|
| `CONFIG_NAME`, `LEGACY_CONFIG_NAME` | File name constants | `strands_decider.config` |
| `config_path(path)` | Picks new or legacy config file | `strands_decider.config` |
| `base_revision(path, config)` | Falls back to provenance.json | `strands_decider.config` |
| `checkpoint_dir(path)` | Local dir or Hub snapshot download | `strands_decider.config` |
| `load_head_state(path)` | Loads safetensors or .pt head state | `strands_decider.config` |

All backend-neutral. These are the on-disk contract: both `TorchServingModel.load` and `MlxServingModel.load` and `LlamaCppServingModel.load` call these.

## Target module split for `modeling.py`

| Current piece | New home |
|---|---|
| `StrandsDeciderConfig`, `CONFIG_NAME`, `LEGACY_CONFIG_NAME`, `config_path`, `base_revision`, `checkpoint_dir`, `load_head_state` | `strands_decider/config.py` |
| `MASK_VALUE`, `GEMMA4_CLASSES`, `serving_attn_implementation` | `strands_decider/config.py` (constants) + stays here or `torch_internals.py` |
| `PointerHead`, `build_head` | `strands_decider/heads.py` (post-SlotHead-removal, `build_head` is just `PointerHead(...)`) |
| `gather_options`, `pool_last_token`, `masked_log_softmax`, `apply_temperature` | `strands_decider/ops.py` |
| `HostEmbedding`, `ensure_bos` | `strands_decider/torch_internals.py` (torch-only) |
| `StrandsDeciderModel` serving surface (`__init__`, `hidden_size`, `_load_torso`, `is_hybrid`, `encode`, serving read logic from `forward`, `load`) | `strands_decider/model.py` → `TorchServingModel` + the `ServingModel` Protocol |
| `StrandsDeciderModel` training surface (`from_pretrained_base`, `attach_lora`, `freeze_torso`, `trainable_parameters`, loss branch of `forward`, `save_pretrained`) | `training/model.py` → `TrainingModel(TorchServingModel)` |
| `init_head_from_lm_head`, `_output_embedding`, `frozen_slot_log_probs`, `SlotHead` | DELETE (slot-head only; A5 confirmed slot head is out) — but check `frozen_slot_log_probs` usage first; it's slot-head-specific too but is more fundamentally a "frozen torso readout" tool |

## Surprises / gotchas for the refactor

1. **Training `forward` is used for serving** (the batched path in `infer.py` calls `self.model(...)` which routes to `.forward(...)`). The split into `TorchServingModel.forward_rows` + `TorchServingModel.read` must either preserve this (TrainingModel.forward calls self.forward_rows + self.read) or carefully update both callers.
2. **`load` bypasses `__init__`** because `__init__` attaches LoRA. The split removes this constraint: `TorchServingModel.__init__` has no LoRA, so `load` constructs normally.
3. **`slot_head.pt` is the on-disk filename even for pointer heads.** Design-doc note; keep for back-compat.
4. **Legacy `hobson_config.json`.** `config_path` falls back to it. Preserve.
5. **Head is fp32, torso may be bf16/fp16/fp32.** `build_head` always `.to(torch.float32)`. The design must keep this discipline through the split.
6. **`frozen_slot_log_probs` depends on tied embeddings.** Pre-asserted (`tie_word_embeddings`). Qwen3 has them; some models do not. If the method is kept for a frozen pointer baseline (seems unlikely), handle the pointer case differently; otherwise delete with SlotHead.
7. **`is_hybrid` is a public staticmethod** — training code (`training/train.py`, `training/soup.py`, test files) imports or might import it. Check training imports before moving.
