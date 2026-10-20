# migration.md

Candidate migration sequence. Goal: each step keeps the full test suite green (`pytest -q` + GPU/MPS/MLX smoke tests on the relevant hosts), each step is reviewable as its own PR, and no step requires rewriting a step before it.

## Design constraints

- **Test suite stays green between steps.** No step broadens an API break that persists across step boundaries.
- **In-tree checkpoints keep loading.** Head state filename stays `slot_head.pt`; `strands_decider_config.json` + legacy `hobson_config.json` both still read.
- **Training stays runnable throughout.** Training code imports stable public surfaces; private moves are shimmed as needed.
- **Public re-exports preserved during transition.** `strands_decider.modeling` keeps re-exporting symbols that moved; a final step cleans up the shims.
- **No big-bang.** Each step is 100-500 LOC net, mergeable alone.

## Suggested order

### Step A — Delete dead code (0 structural change)

1. Delete `SlotHead` and all slot-head-only methods (`init_head_from_lm_head`, `frozen_slot_log_probs` iff unused for pointer baseline, `head_init == "lm_head"` branch, `build_head` slot branch).
2. Clean `StrandsDeciderConfig.head_type` to pointer-only (keep the field for legacy load; raise on `"slot"`).
3. Clean references in training tests (`test_variance.py` imports SlotHead).

**Risk:** low. Entirely subtractive. Rebuild test suite before and after.
**Deliverable:** `modeling.py` shrinks ~50-80 LOC. Everything else unchanged.

### Step B — Extract shared primitives (pure moves)

Move with re-exports from `strands_decider.modeling`:

1. Create `strands_decider/config.py` — `StrandsDeciderConfig`, `MASK_VALUE`, `GEMMA4_CLASSES`, `CONFIG_NAME`, `LEGACY_CONFIG_NAME`, `config_path`, `base_revision`, `checkpoint_dir`, `load_head_state`.
2. Create `strands_decider/heads.py` — `PointerHead`, `build_head` (one-line post-cleanup).
3. Create `strands_decider/ops.py` — `pool_last_token`, `masked_log_softmax`, `apply_temperature`, `gather_options`.
4. Create `strands_decider/torch_internals.py` — `HostEmbedding`, `ensure_bos`, `serving_attn_implementation`, `mm_token_type_ids`, `qwen_base`.
5. `strands_decider/modeling.py` becomes a shim: `from .config import *; from .heads import *; from .ops import *; from .torch_internals import *; from .model import ...` once `model.py` exists; for now, re-export only.

**Risk:** low. Imports work. Preserves `strands_decider.modeling.X` for every external consumer.
**Deliverable:** new modules, `modeling.py` reduced to re-exports + the two classes (`StrandsDeciderModel`, which stays).

### Step C — Introduce `strands_decider.model` with `ServingModel` + `TorchServingModel`

1. Create `strands_decider/model.py`:
   - `ServingModel(Protocol)` — `config`, `tokenizer`, `handles_images`, `supports_batch_rows`, `encode`, `forward_rows`, `read`, `ask`.
   - `TorchServingModel(nn.Module)` — the serving surface. Loads a checkpoint, runs forward, reads with pointer head.
   - `TorchServingModel.load(path, device, attn_implementation=None, vision=False, host_embeddings=None)` — factory. Reads `strands_decider_config.json`, loads torso (reusing `_load_torso` logic from `modeling.py` and `vision.py`), optional vision tower.
   - `TorchServingModel.forward_rows(ids, mask, cache=None)` — the shared forward (same code as today's `_slot_probs_batched`'s `self.model(...)` call, extracted).
   - `TorchServingModel.read(hidden, option_positions)` — applies pointer head via shared `apply_pointer_head(hidden, positions, self.pointer_head)` from `ops.py`.
   - `TorchServingModel.ask(state, questions, images=None)` — shared-prefix path (extracted from today's `_slot_probs_shared_prefix` for text, `_image_probs` for vision).
2. `StrandsDeciderModel` subclasses `TorchServingModel` and adds training methods (temporary — moves out in Step F).

**Risk:** medium — this is where the shared-prefix path needs careful preservation (cache fork, pointer head on suffix). Pin with the existing `test_prefix_cache.py` tests.
**Deliverable:** `TorchServingModel` exists and passes its own serving tests. `StrandsDeciderModel` still exists as a training convenience.

### Step D — Introduce `strands_decider.engine` + `strands_decider.scheduler`

1. Create `strands_decider/scheduler.py`:
   - `Scheduler(Protocol)`.
   - `NoScheduler(Scheduler)` — synchronous one-at-a-time.
   - `SerialiseScheduler(Scheduler)` — one lock, one at a time (MPS).
2. Create `strands_decider/engine.py`:
   - `EngineConfig` (move from `infer.py`).
   - `Engine(model, scheduler, config)` with `infer`, `health`, `shutdown`.
   - `make_engine(ckpt, device, config, vision=False)` factory.
   - `prep(req, model, config) → PreparedRequest` (window-fit, option positions, kinds, temperatures).

Shim `strands_decider.infer`:
```python
from .engine import EngineConfig, Engine, make_engine

# Compat alias for existing callers
from .engine import Engine as SystemOneEngine

load_engine = make_engine
```

Also keep the cache-fork functions here as deprecated shims pointing at `strands_decider.torch_internals` (where they move).

**Risk:** medium — ensure `Engine.infer` is a drop-in for `SystemOneEngine.evaluate`. Pin with existing engine tests (`test_batch_tokens.py`, `test_engine_concurrency.py`, `test_hybrid.py`).
**Deliverable:** entry layers (CLI, server) begin using `Engine`. Scheduler path is `NoScheduler` for now (behaviourally identical to today).

### Step E — Introduce `strands_decider.mlx_model`

1. Create `strands_decider/mlx_model.py`:
   - `MlxServingModel(ServingModel)` — holds config, tokenizer, head (shared `PointerHead`), decoder (MLX), cache owner.
   - `MlxServingModel.load(path, cache_limit_bytes=DEFAULT_CACHE_LIMIT)`.
   - Implements `encode`, `forward_rows`, `read`, `ask`. Reuses `apply_pointer_head` with converted hidden.
2. Update `make_engine(..., device="mlx")` to construct `MlxServingModel` + `SerialiseScheduler`.
3. Shim `strands_decider.mlx_engine`:
```python
from .mlx_model import MlxServingModel, merge_lora  # noqa: F401
# Compat alias — MLXEngine → the composition
# External users: use strands_decider.engine.make_engine(ckpt, device="mlx") instead
```

**Risk:** medium — the subclass-with-no-super init goes away; carefully preserve the `_lock`, `_prefix_cache_unforkable`, and `mx.set_cache_limit` behaviours. Pin with `test_mlx_engine.py` fidelity tests (merged LoRA, 54-answer parity).
**Deliverable:** MLX serves through the new Engine + Scheduler shape.

### Step F — Move `TrainingModel` to `training/model.py`; strip serving from `StrandsDeciderModel`

1. Create `training/model.py` containing `TrainingModel(TorchServingModel)` with all training-only methods: `from_pretrained_base`, `attach_lora`, `freeze_torso`, `trainable_parameters`, loss branch of `forward`, `save_pretrained`.
2. Delete `StrandsDeciderModel` from `strands_decider/modeling.py`.
3. Keep `strands_decider.modeling` as a thin re-export shim (now excluding `StrandsDeciderModel`).
4. Rewrite every `training/*.py` import that references `strands_decider.modeling.StrandsDeciderModel` to `training.model.TrainingModel`.
5. Rewrite training tests similarly.

**Risk:** medium-high — the broad training-code import rewrite. Mitigation: do it mechanically with search/replace; run `pytest training/tests/`.
**Deliverable:** `src/strands_decider/` no longer carries training-only code. `TrainingModel` is the training class. The `StrandsDeciderConfig` + checkpoint format is the contract between training and serving.

### Step G — Vision as composition (not subclass)

1. In `TorchServingModel.load`, when `vision=True`, keep the vision tower (`Qwen3_5ForConditionalGeneration`); otherwise discard. Set `handles_images = True`.
2. Pass `pixel_values`, `image_grid_thw`, `position_ids` through `forward_rows` kwargs — delete the `_mm` side channel.
3. Delete `VisionDeciderModel`, `VisionEngine`, `VisionEngineConfig`; promote their logic into `TorchServingModel.ask(state, questions, images)` and `TorchServingModel.forward_rows(...)` with explicit position args.
4. Shim `strands_decider.vision` to re-export `images.py` symbols for compat.
5. Delete `_reset_positions` — always-explicit positions make it unnecessary.
6. Update `make_engine(..., vision=True)` to construct `TorchServingModel(vision=True)` + chosen scheduler.

**Risk:** medium-high — vision M-RoPE paths are subtle. Pin with `test_vision.py` end-to-end tests.
**Deliverable:** vision is a backend capability, not a parallel subclass tree.

### Step H — Add `MicroBatcher` (lands PR #88's logic under the new structure)

1. Add `MicroBatcher(Scheduler)` to `scheduler.py` with: queue, `plan_rows`, `tune_batching`, OOM split, lost-context stop.
2. Update `make_engine` to pick `MicroBatcher` for `cuda + batching=True + model.supports_batch_rows`.
3. Server handler calls `await run_in_threadpool(engine.infer, req)` (already set up in Step D).
4. CLI/server surface: `--no-batching`, `--batch-wait-ms`, `--no-batch-tune`, `/health` fields.
5. Preserve PR #88's test file (`test_batching.py`) with import updates.

**Risk:** medium — this is a large feature. If PR #88 is merged before this step, it becomes a code move + rename.
**Deliverable:** cross-request batching under the new architecture.

### Step I — `LlamaCppServingModel` + Lambda example (PR #77 under new structure)

1. If PR #77 is merged before this: port `hidden_engine.py` → `strands_decider/llamacpp_model.py` as `LlamaCppServingModel(ServingModel)`; move Lambda deployment artefacts stay in `examples/lambda/`.
2. If PR #77 is still open: author it on the new shape directly.
3. Lambda handler uses `make_engine(ckpt, device="llamacpp") + NoScheduler`.

**Risk:** medium — numpy pointer head is a separate implementation from the shared torch `PointerHead`. Pin parity tests (PR #77 already has these).
**Deliverable:** three-backend ServingModel composition under one Protocol.

### Step J — Clean up shims

1. Delete the `strands_decider.modeling` re-export shim.
2. Delete the `strands_decider.infer` compat shim.
3. Delete the `strands_decider.vision` re-export shim.
4. Delete the `strands_decider.mlx_engine` compat shim.
5. Update any external docs that reference the old import paths.

**Risk:** low — purely subtractive after all external callers have been updated.
**Deliverable:** clean public surface. New users see the new layout.

### Step K — Optional: `Trainer` protocol (deferred per A6)

Future refactor: convert `TrainingModel` into `TorchTrainer` composing a `TorchServingModel`. Noted in design, out of scope for this effort.

## Parity testing discipline

Each step should preserve or add tests that pin:

| What | Test | Across steps |
|---|---|---|
| Checkpoint load (every head/legacy name) | `test_checkpoint_load.py` | A-J |
| Shared-prefix path matches batched (torch) | `test_prefix_cache.py` | C, D, G |
| MLX LoRA merge + answer parity | `test_mlx_engine.py` | E, J |
| CPU/MPS kernels still install and dispatch | `test_cpu_kernels.py`, `test_mps_kernels.py` | B (torch_internals move), D |
| Vision text-path + image-path answers | `test_vision.py` | G |
| Engine concurrency, dtype, batch tokens | `test_engine_concurrency.py`, `test_engine_dtype.py`, `test_batch_tokens.py` | C, D, H |
| Server end-to-end | `test_server.py` | D, G, H |
| Hybrid torso (Qwen3.5 Gated DeltaNet) | `test_hybrid.py` | C, D |
| Gemma 4 (serving_attn, HostEmbedding, BOS) | `test_gemma4.py`, `test_serving_attn.py` | B, C |

Across the whole migration, add one new parity test:

- `tests/test_training_serving_parity.py`: build a `TrainingModel` on a tiny config, `save_pretrained`, load as `TorchServingModel`, assert bit-identical logits on the same inputs. Introduced in Step F.

## Ordering decisions

- **A before B** so B doesn't carry dead-code baggage.
- **B before C** so new `TorchServingModel` can import from `heads/ops/config` cleanly.
- **D after C** so Engine can depend on the `ServingModel` Protocol.
- **E after D** so MLX uses the Scheduler shape and goes through `make_engine`.
- **F after E** because MLX code historically referenced `StrandsDeciderModel.config`; cleaner to remove `StrandsDeciderModel` after MLX has moved.
- **G after F** because vision composition code needs the clean `TorchServingModel` without training coupling.
- **H after G** so `MicroBatcher`'s compat flag `supports_batch_rows=False` is already defined for vision.
- **I any time after D** — depends only on the Engine/Scheduler/ServingModel shape being in place; can parallelise with any step after D.
- **J last** — removes the compat shims.

## Rough LOC per step

- A: -50 to -80 LOC.
- B: ~+500 (new files) / -0 (re-exports keep shape). Net ~+500.
- C: +500 (new file) / -200 (removed from modeling). Net +300.
- D: +400 / -300. Net +100.
- E: +300 / -278 (mlx_engine.py). Net +25.
- F: +250 / -500 (StrandsDeciderModel). Net -250. **Peak training-code churn here.**
- G: +100 / -575 (vision.py). Net -475.
- H: +325 or so (PR #88).
- I: +322 (hidden_engine as llamacpp_model) + whatever PR #77 brings.
- J: -200 (shim deletions).

Net: `src/strands_decider/` moves from ~3,225 LOC today to roughly 3,000-3,500 LOC after all steps, with the content redistributed and better-organised.
