# Implementation plan: refactor `src/strands_decider/` abstractions

Each step is a reviewable PR that keeps the full test suite green. Steps 1-10 correspond to migration steps A-J from the design doc, written as implementation objectives with tests folded into each step.

## Checklist

- [ ] **Step 1 — Delete `SlotHead` and slot-head-only hooks**
- [ ] **Step 2 — Extract shared primitives into `config`, `heads`, `ops`, `torch_internals`**
- [ ] **Step 3 — Introduce `ServingModel` Protocol and `TorchServingModel`**
- [ ] **Step 4 — Introduce `Engine`, `Scheduler`, `NoScheduler`; wire the entry layers**
- [ ] **Step 5 — Introduce `MlxServingModel`; delete the subclass-bypass-init hack**
- [ ] **Step 6 — Move `TrainingModel` to `training/`; delete `StrandsDeciderModel` from `src/`**
- [ ] **Step 7 — Vision as composition; delete `VisionDeciderModel` and `VisionEngine`**
- [ ] **Step 8 — Add `MicroBatcher`; land PR #88's logic under the new shape**
- [ ] **Step 9 — Add `LlamaCppServingModel`; land PR #77's hidden-engine under the new shape**
- [ ] **Step 10 — Delete migration shims; clean public surface**

---

## Step 1: Delete `SlotHead` and slot-head-only hooks

**Objective.** Remove dead code before any reshaping. SlotHead is no longer used; its removal shrinks `modeling.py` by ~50-80 LOC and simplifies every later step.

**Implementation guidance.**
- Delete the `SlotHead` class from `src/strands_decider/modeling.py`.
- Delete `init_head_from_lm_head`, `_output_embedding` (used only by SlotHead init), and the frozen-slot path (`frozen_slot_log_probs` and its vision subclass override) iff grep confirms no caller remains after SlotHead deletion.
- Simplify `build_head` to construct `PointerHead` only; retain the function signature for one release to minimise churn.
- In `StrandsDeciderConfig`, keep the `head_type` field for legacy JSON compatibility but raise `ValueError("slot head is no longer supported; use strands-decider <= <release>")` when `head_type == "slot"` in `StrandsDeciderConfig.from_json`.
- Delete slot-only fields' defaults where sensible (`head_init`'s `"lm_head"` branch).
- Remove `SlotHead` imports and references from `training/tests/test_variance.py` and any other callers; keep the tests they contain by porting them to `PointerHead` or delete them if the slot-head-specific test is no longer meaningful.

**Test requirements.**
- Full `pytest -q` green across `tests/` and `training/tests/`.
- Add one regression test: loading a checkpoint whose `strands_decider_config.json` sets `head_type: "slot"` raises a clear error.
- Preserve `tests/test_checkpoint_load.py` coverage of pointer-head checkpoints unchanged.

**Integration.** Baseline step. No new abstractions introduced; subsequent steps assume slot-head is gone.

**Demo.** `pytest -q` passes with the slot-head code removed. Loading a legacy slot-head checkpoint raises a message that tells the user which release still supports it.

---

## Step 2: Extract shared primitives into `config`, `heads`, `ops`, `torch_internals`

**Objective.** Split `modeling.py` horizontally into pure-concern modules. No semantic changes — this is a move with re-exports so every external import continues to work.

**Implementation guidance.**
- Create `src/strands_decider/config.py`: move `StrandsDeciderConfig`, `MASK_VALUE`, `GEMMA4_CLASSES`, `CONFIG_NAME`, `LEGACY_CONFIG_NAME`, `config_path`, `base_revision`, `checkpoint_dir`, `load_head_state`.
- Create `src/strands_decider/heads.py`: move `PointerHead` and the simplified `build_head`.
- Create `src/strands_decider/ops.py`: move `pool_last_token`, `gather_options`, `masked_log_softmax`, `apply_temperature`. Add `apply_pointer_head(hidden, option_positions, pointer_head) -> logits` that composes `gather_options` + pointer-head application, so backends share one implementation.
- Create `src/strands_decider/torch_internals.py`: move `HostEmbedding`, `ensure_bos`, `serving_attn_implementation`, the cache-fork helpers (`_fork_layered_cache`, `_expand_cache`, `UnforkableCache`, `_ROW_STATES`, `_SHARED_TENSORS`) from `infer.py`, and `mm_token_type_ids`, `qwen_base` from `vision.py`. Add `mlx_to_torch(mlx_array) -> torch.Tensor` for future step 5.
- Convert `src/strands_decider/modeling.py` to a re-export shim: `from .config import *; from .heads import *; from .ops import *; from .torch_internals import *; from .modeling_impl import StrandsDeciderModel`. The remaining `StrandsDeciderModel` class moves to `src/strands_decider/modeling_impl.py` or stays in `modeling.py` alongside the re-exports — either works; pick whichever keeps the diff smaller.
- Create a `kernels/` subpackage under `strands_decider/`: `kernels/__init__.py` with `install_for(device: str)`, `kernels/cpu.py` from today's `cpu_kernels.py`, `kernels/mps.py` from today's `mps_kernels.py`. Convert `cpu_kernels.py` and `mps_kernels.py` to re-export shims pointing at `kernels/cpu.py` and `kernels/mps.py`.

**Test requirements.**
- Full `pytest -q` green with zero test modifications (re-exports preserve import paths).
- Add `tests/test_module_layout.py` that imports each new module and asserts the moved symbols are accessible both from the new location and the legacy `strands_decider.modeling` path.

**Integration.** `modeling.py` becomes a thin shim; every subsequent step imports from the new leaf modules. `kernels/` subpackage will grow (PR #36 adds a third file; MLX / llama.cpp never import from it).

**Demo.** `python -c "from strands_decider.heads import PointerHead; from strands_decider.ops import pool_last_token; from strands_decider.config import StrandsDeciderConfig; print('ok')"` and `python -c "from strands_decider.modeling import PointerHead, pool_last_token, StrandsDeciderConfig; print('ok')"` both succeed. Full pytest green.

---

## Step 3: Introduce `ServingModel` Protocol and `TorchServingModel`

**Objective.** Carve the serving surface out of `StrandsDeciderModel` into a dedicated class. `StrandsDeciderModel` subclasses the new class temporarily so training continues to work.

**Implementation guidance.**
- Create `src/strands_decider/model.py`:
  - `ServingModel(Protocol)` with `config`, `tokenizer`, `device`, `handles_images`, `supports_batch_rows`, `encode`, `forward_rows`, `read`, `ask` — exactly as the design pins.
  - `TorchServingModel(nn.Module)` implementing the Protocol:
    - `__init__(config, torso, pointer_head, tokenizer, vision_tower=None, device="cuda")`.
    - `supports_batch_rows = True`; `handles_images = bool(vision_tower)`.
    - `_lock = threading.RLock() if device == "mps" else contextlib.nullcontext()` — the model-owned lock pattern from the design.
    - `encode(text, images=None)` — tokenise via `self.tokenizer`.
    - `forward_rows(ids, mask, cache=None, pixel_values=None, image_grid_thw=None, position_ids=None)` — runs the torso under `self._lock`. Keep vision kwargs on the signature; `TorchServingModel.vision_tower is None` makes them no-ops in the forward.
    - `read(hidden, option_positions)` — calls `apply_pointer_head` from `ops.py` with `self.pointer_head`.
    - `ask(state, questions, images=None)` — the shared-prefix path. Reuses `prep` logic (extracted from today's `_fit` and `_slot_probs_shared_prefix` in `infer.py` + the vision counterparts from `vision.py`'s `_image_probs`). Under `self._lock`.
    - `@classmethod load(path, *, device="cuda", attn_implementation=None, vision=False, host_embeddings=None)` — the serving entry point. Pulls torso-load logic out of today's `StrandsDeciderModel.load` + `VisionDeciderModel.load` (vision loads the Qwen3.5 multimodal class and keeps the vision tower). Calls `kernels.install_for(device)` at the right moment. Applies serving `attn_implementation` via `serving_attn_implementation` from `torch_internals`.
- Change `StrandsDeciderModel`'s class declaration to `class StrandsDeciderModel(TorchServingModel)` — inherit the new serving surface. Keep the training-only methods on `StrandsDeciderModel` (they move to `training/` in Step 6).
- Fix the `__init__`-bypass hack: `TorchServingModel.__init__` does not attach LoRA, so `TorchServingModel.load` can use normal construction. `StrandsDeciderModel.load` continues to need the bypass only while LoRA attach lives on `StrandsDeciderModel.__init__`.

**Test requirements.**
- Preserve `tests/test_prefix_cache.py` pins across the extraction — the shared-prefix path moves from `infer.py` to `TorchServingModel.ask`. Both plain and hybrid torsos must still agree with batched encoding to float tolerance.
- Preserve `tests/test_hybrid.py`, `tests/test_batch_tokens.py` without changes.
- Add `tests/test_torch_serving_model.py` with four assertions: `TorchServingModel.load` yields a working model; `forward_rows` + `read` on a known fixture match `StrandsDeciderModel.forward` on the same inputs (same-implementation guarantee); `handles_images = True` when `vision=True`; `supports_batch_rows = True` always.

**Integration.** Serving-side callers can start using `TorchServingModel` directly. `StrandsDeciderModel` still exists and still works as before because it inherits.

**Demo.** `python -c "from strands_decider.model import TorchServingModel; m = TorchServingModel.load('<checkpoint>', device='cpu'); print(m.encode('hello world'))"` works. CLI's `strands-decider ask` and `serve` still work unchanged.

---

## Step 4: Introduce `Engine`, `Scheduler`, `NoScheduler`; wire the entry layers

**Objective.** Replace `SystemOneEngine` orchestration with `Engine` + `Scheduler`. CLI, server, and tests move to the new API; a compat alias keeps the old import working.

**Implementation guidance.**
- Create `src/strands_decider/scheduler.py`:
  - `Scheduler(Protocol)` with `submit`, `health`, `shutdown`.
  - `PreparedRequest` NamedTuple with the fields the design pins.
  - `NoScheduler(Scheduler)` — runs `model.forward_rows` + `model.read` on the caller's thread.
- Create `src/strands_decider/engine.py`:
  - `EngineConfig` dataclass — moved from today's `infer.py`, extended with the fields the design calls out (`batching`, `batch_wait_ms`, `batch_tune`, `max_batch_tokens`, image-related). Keep legacy field names stable.
  - `prep(request, model, config) -> PreparedRequest | list[PreparedRequest]` — extracted from today's `_fit`, `_option_idx`, `_pad`, `_temperatures`, `_chunks`, and the chunking logic in `evaluate`. Pure CPU; runs on the caller's thread; raises on over-window with `strict_window`.
  - `Engine(model, scheduler, config)` with `infer`, `health`, `shutdown`. `infer` short-circuits multi-question requests to `model.ask` and routes single-question to `scheduler.submit`.
  - `make_engine(checkpoint, *, device, config, deployment)` factory that picks `MlxServingModel` / `TorchServingModel` / (future) `LlamaCppServingModel` based on `device`, and picks `MicroBatcher` (not yet implemented — falls to `NoScheduler` in this step) or `NoScheduler`.
  - Move `_to_answer` and `_option_token_index` from `infer.py` to `engine.py` (or into `prompting.py` for a cleaner home — pick one; design doc notes either works).
- Convert `src/strands_decider/infer.py` into a compat shim:
  - `from .engine import EngineConfig, Engine, make_engine as load_engine`
  - `SystemOneEngine = Engine` (type alias).
  - Re-export `UnforkableCache`, `_expand_cache` from `torch_internals` for the few tests that import them.
- Update `src/strands_decider/server.py` to call `Engine.infer` via `await run_in_threadpool(engine.infer, request)`. Keep the `/health` endpoint; its body is now `engine.health()`.
- Update `src/strands_decider/cli.py` to construct `Engine` via `make_engine(...)` with `deployment="cli"`.

**Test requirements.**
- Every existing test that imports from `strands_decider.infer` must pass with the compat shim.
- Add `tests/test_engine.py`: construct an `Engine` with a stub model + `NoScheduler`, assert `infer` routes single-question → `scheduler.submit`, multi-question → `model.ask`, and image + text-only-model → `ValueError`.
- Add `tests/test_engine_concurrency.py` coverage (or extend existing) to assert `Engine.infer` is safe across threads under `NoScheduler` on a torch CPU model.
- Preserve `tests/test_server.py` and `tests/test_devices.py` unchanged.
- Preserve `tests/test_batch_tokens.py` — the chunking logic in `prep` must match today's `_chunks` behaviour byte-for-byte on the fixture requests.

**Integration.** The entry layer is now using the new Engine. All backends go through `make_engine`. Scheduler path is `NoScheduler` for every configuration — behaviourally identical to today's non-batching serving.

**Demo.** `strands-decider ask <ckpt>` and `strands-decider serve <ckpt>` work. `curl POST /v1/systemone` returns the same answers as before. `/health` reports the new structured fields (`model`, `device`, `handles_images`, `supports_batch_rows`).

---

## Step 5: Introduce `MlxServingModel`; delete the subclass-bypass-init hack

**Objective.** `MLXEngine` becomes `MlxServingModel`, a sibling of `TorchServingModel` under the Protocol. The subclass-with-no-`super().__init__` dance disappears.

**Implementation guidance.**
- Create `src/strands_decider/mlx_model.py`:
  - `MlxServingModel` holding `config`, `tokenizer`, `decoder`, `_lm` (for cache), `pointer_head` (the same torch `PointerHead` class on CPU fp32), `device = "mlx"`, `handles_images = False`, `supports_batch_rows = False`, `_lock = threading.RLock()`.
  - `encode`, `forward_rows`, `read`, `ask` — mirror today's `MLXEngine._hidden`, `_probs`, `_slot_probs_batched`, `_slot_probs_shared_prefix` with the lock applied inside `forward_rows` and `ask` (replacing today's `_lock` on the engine).
  - `read(hidden_mlx, option_positions)`: gather pooled + per-option rows on MLX, convert the small slice to torch via `torch_internals.mlx_to_torch`, call shared `ops.apply_pointer_head(hidden_torch, positions, self.pointer_head)` — same code the torch backend calls.
  - `@classmethod load(path, *, cache_limit_bytes=1 << 30)` — mirror today's `load_mlx_engine`. Preserve `mx.set_cache_limit(cache_limit_bytes)` behaviour and the strict `merge_lora` refusal list. Both files (`merge_lora`, `_load_torso` from mlx-lm) live in `mlx_model.py` as private helpers.
- Delete `src/strands_decider/mlx_engine.py` content; replace with a compat shim: `from .mlx_model import MlxServingModel, merge_lora` and a `MLXEngine = MlxServingModel` alias for any external callers.
- Update `engine.make_engine` to construct `MlxServingModel.load(...)` when `device == "mlx"` and compose it with `NoScheduler` (the lock inside the model handles serialisation).

**Test requirements.**
- Preserve every test in `tests/test_mlx_engine.py`: LoRA merge refusal list, 54-answer parity against torch fp32 (0.0138 max probability delta, no answer changes), MLX torso load, MLX shared-prefix path.
- Add `tests/test_mlx_lock.py` on Mac hosts only: two concurrent `Engine.infer` calls on an `MlxServingModel` do not crash (today's `_lock` would prevent crashes; the new model-owned lock must do the same).

**Integration.** MLX now serves through `make_engine(device="mlx")`. The subclass-with-no-super-init pattern is gone. `strands_decider.mlx_engine` keeps working via the compat shim.

**Demo.** `strands-decider serve <ckpt> --device mlx` on an Apple-silicon host serves the same answers as before. `/health` reports `{"device": "mlx", "supports_batch_rows": false}`.

---

## Step 6: Move `TrainingModel` to `training/`; delete `StrandsDeciderModel` from `src/`

**Objective.** The `src/strands_decider/` package becomes serving-only. `TrainingModel` lives in `training/` and imports from the shared primitives in `src/`.

**Implementation guidance.**
- Create `training/model.py`:
  - `TrainingModel(TorchServingModel)` with every training-only method today's `StrandsDeciderModel` carries: `from_pretrained_base`, `attach_lora`, `freeze_torso`, `trainable_parameters`, training `forward(batch) -> loss` (uses inherited `forward_rows` + `read` and composes with cross-entropy / soft-target loss / weighted mean), `save_pretrained`.
  - `TrainingModel.load(path, trainable=True)` for resume-training, mirroring today's `StrandsDeciderModel.load(trainable=True)` branch.
  - Keep the `__init__`-bypass hack only on `TrainingModel.load` because training needs `attach_lora` to happen on construction.
- Rewrite every `training/*.py` import that references `strands_decider.modeling.StrandsDeciderModel` to `training.model.TrainingModel`. The grep in `training-coupling.md` is the full list.
- Delete `StrandsDeciderModel` from `src/strands_decider/modeling.py` (or `modeling_impl.py` from Step 2). Keep the re-export shim pointing at the moved primitives but **not** at `StrandsDeciderModel` — external code with `from strands_decider.modeling import StrandsDeciderModel` gets an `ImportError` with a message pointing at `training.model.TrainingModel`. The compat shim can carry a `_deprecated` tombstone that raises on `__getattr__("StrandsDeciderModel")`.
- Delete `VisionDeciderModel` is **not** in this step — vision stays as it is until Step 7. For now, `VisionDeciderModel` becomes `VisionDeciderModel(TorchServingModel)` with vision-specific overrides, mirroring today's inheritance.

**Test requirements.**
- Add `tests/test_training_serving_parity.py`: construct a `TrainingModel` on a tiny config, `save_pretrained(tmpdir)`, load with `TorchServingModel.load(tmpdir)`, assert bit-identical logits on the same inputs. Pins the single-implementation guarantee across the checkpoint round-trip.
- Preserve `training/tests/test_configs.py`, `training/tests/test_variance.py` (SlotHead-free since Step 1), `training/tests/test_core.py`, `training/tests/test_ddp.py`, `training/tests/test_gemma4_train.py`, `tests/test_hf_export.py`, `tests/test_checkpoint_load.py`.
- Verify `python -m training.train` kicks off a run on a tiny config.

**Integration.** The `strands_decider` package no longer carries training-only code. Training imports from `strands_decider.{config, heads, ops, torch_internals, model}` and from `training.model`. Vision still inherits `TorchServingModel` because Step 7 hasn't run yet — but the inheritance chain is `VisionDeciderModel(TorchServingModel)`, cleaner than today's `(StrandsDeciderModel)`.

**Demo.** `python -m training.train --config <tiny.json>` runs a training step and saves a checkpoint. `strands-decider ask <checkpoint>` loads that checkpoint and answers questions. `test_training_serving_parity.py` passes.

---

## Step 7: Vision as composition; delete `VisionDeciderModel` and `VisionEngine`

**Objective.** The vision tower is an optional submodule on `TorchServingModel`, not a subclass. Delete the `_mm` side channel and the `_reset_positions` workaround.

**Implementation guidance.**
- Extend `TorchServingModel`:
  - `__init__` accepts an optional `vision_tower` nn.Module.
  - `handles_images` reflects `bool(vision_tower)`.
  - `forward_rows` accepts `pixel_values`, `image_grid_thw`, `position_ids` as kwargs (not through a hidden attribute). When `vision_tower is None` and any of these is set, raise. When present, pass them to the torso forward alongside the ids.
  - `ask(state, questions, images=None)` grows an image path: if `images` is provided and `handles_images`, run the image prep (reusing `strands_decider.images.*` helpers), compute explicit `position_ids` with the torso's `get_rope_index`, encode the image-aware prefix, fork the KV cache, forward suffixes with explicit positions, apply the pointer head via `read`. The entire path is explicit about positions — no `rope_deltas` state is left on the torso.
  - `TorchServingModel.load(path, *, vision=False, ...)` constructs `vision_tower` when `vision=True` by using the `Qwen3_5ForConditionalGeneration` loader and keeping the ViT + merger alongside the decoder. LoRA key remap for multimodal checkpoints lives in `load`.
- Delete `VisionDeciderModel`, `VisionEngine`, `VisionEngineConfig`, `load_vision_engine` from `src/strands_decider/vision.py`. Keep `src/strands_decider/images.py` as the shared image prep module (already extracted by PR #77 or extracted here if PR #77 hasn't merged).
- Convert `src/strands_decider/vision.py` to a thin compat shim re-exporting `strands_decider.images.*` for one release.
- Update `make_engine(..., config.vision=True)` to call `TorchServingModel.load(..., vision=True)` + the chosen scheduler (NoScheduler because vision does not batch — set `TorchServingModel.supports_batch_rows = False` when `vision_tower is not None`).

**Test requirements.**
- Preserve `tests/test_vision.py` end-to-end: text request on a vision-capable server returns identical answers (same path as before via inherited text forward); image request returns answers within the published tolerances; multi-image request works; `strict_window` refusal still fires.
- Add `tests/test_vision_composition.py`: `TorchServingModel.load(ckpt, vision=False)` has `handles_images = False`; same checkpoint with `vision=True` has `handles_images = True`; the forward signature accepts image kwargs; a vision model's `supports_batch_rows` is False.
- Add a regression test confirming `rope_deltas` is **never** stored on the torso between requests (any attempt to set it is overwritten by explicit `position_ids` on the next request).

**Integration.** Vision is now a `TorchServingModel` configuration, not a subclass. The `_mm` side channel is gone. The `_reset_positions` workaround is gone — a correctness win from the refactor.

**Demo.** `strands-decider serve <ckpt> --vision` on CUDA/CPU/MPS answers text requests and image requests through the same `/v1/systemone` endpoint. `strands-decider ask <ckpt> --image <path>` returns an answer. All vision tests pass.

---

## Step 8: Add `MicroBatcher`; land PR #88's logic under the new shape

**Objective.** Cross-request batching for CUDA serving. If PR #88 has merged by this point, this step is largely a code move; if not, it ports the feature.

**Implementation guidance.**
- Add `MicroBatcher(Scheduler)` to `src/strands_decider/scheduler.py`:
  - Owns a device thread via `threading.Thread(target=self._run, daemon=True)`.
  - Queue of `PreparedRequest`s.
  - `plan_rows(queued) -> list[list[PreparedRequest]]` — sort by length, cut at `max_batch`, `max_batch_tokens`, the padding-waste limit (padded tokens ≤ 1.25× real past 4,096), and the quadratic cap for eager-attention Gemma (`rows × longest² ≤ window²`).
  - OOM-split retry, row-by-row retry on other errors, lost-context stop. Each error path is wired to a `response_slot` so only affected requests fail.
  - `tune_batching(model, config)` runs at startup when `config.batch_tune`: times `model.forward_rows` at widths 1, 2, 4, ... at token sizes 256 / 1k / 4k; stops at the first OOM; records fastest rows-per-forward per width.
  - `health()` returns tuned caps, requests-per-cycle, max cycle size.
- Update `_pick_scheduler` in `engine.py`: `MicroBatcher(model, config)` when `deployment == "server"` and `model.supports_batch_rows` and `config.batching`.
- CLI flags: `--batching / --no-batching`, `--batch-wait-ms N`, `--batch-tune / --no-batch-tune`, `--max-batch`, `--max-batch-tokens`.

**Test requirements.**
- Preserve PR #88's `tests/test_batching.py` (23 tests): batched = unbatched to float tolerance, planner budgets + padding limit + quadratic cap, tuned caps, refusal isolation, OOM splitting, lost-context stop, shared-prefix path, server wiring, `--no-batching` restores bit-identical answers.
- Add `tests/test_scheduler_selection.py`: factory picks `MicroBatcher` only for CUDA + `supports_batch_rows=True` + `batching=True`; `NoScheduler` otherwise.

**Integration.** Server throughput on CUDA matches PR #88's measurements. Other configurations are unaffected (NoScheduler fast path). `/health` reports batcher state.

**Demo.** `strands-decider serve <ckpt> --device cuda` saturates a GPU at the throughput PR #88 published. `--no-batching` restores the pre-PR behaviour bit-identically. `/health` reports batcher tuned caps.

---

## Step 9: Add `LlamaCppServingModel`; land PR #77's hidden-engine under the new shape

**Objective.** Third backend: llama.cpp torso + numpy pointer head, torch-free. Enables the Lambda deployment.

**Implementation guidance.**
- Create `src/strands_decider/llamacpp_model.py`:
  - `LlamaCppServingModel` holding `config`, `tokenizer`, `_lm` (llama.cpp handle via the C shim), `pointer_head_numpy` (numpy weights), `device = "llamacpp"`, `supports_batch_rows = False`, optionally `handles_images = True` when the mmproj GGUF is loaded.
  - `encode`, `forward_rows`, `read` (numpy pointer-head application), `ask` — all pure numpy; no torch anywhere.
  - `@classmethod load(path, *, vision=False)` — loads the Q8_0 GGUF torso via the C shim; loads pointer head weights from the shared `slot_head.pt` safetensors / torch file into numpy; parity-tested against `TorchServingModel` on reference prompts at CI.
- The C shim (`decider_hidden.c`) stays with the Lambda example at `examples/lambda/shim/`.
- Update `engine.make_engine` to pick `LlamaCppServingModel` when `device == "llamacpp"`. Compose with `NoScheduler`.
- Lambda example (`examples/lambda/app/handler.py`) uses `make_engine(ckpt, device="llamacpp", deployment="lambda")`.

**Test requirements.**
- Preserve PR #77's 21 CPU tests from `tests/test_hidden_engine.py` (or new `tests/test_llamacpp_model.py` if PR #77 names it differently): trimming, strict mode, BOS, padding, budget, image pixels, image fit, image answers.
- Add `tests/test_backend_parity.py`: on a shared reference prompt set, `TorchServingModel` and `LlamaCppServingModel` produce answers within documented tolerances (PR #77 published: probabilities within 0.017, no choice changed over 28 requests).
- CI runs the Lambda handler locally against a mock event to confirm the deployment path still works.

**Integration.** Three concrete `ServingModel`s under one Protocol. The Lambda deployment loads the same checkpoint directory as torch and MLX.

**Demo.** `strands-decider serve <ckpt> --device llamacpp` serves answers on a CPU host with no torch installed. `aws lambda invoke --function-name strands-decider ...` on the deployed stack returns the same answers at the published cost.

---

## Step 10: Delete migration shims; clean public surface

**Objective.** Remove the backward-compatibility layers introduced in Steps 2-5. Users who followed the deprecation warnings have moved to the new paths; the final shape is what remains.

**Implementation guidance.**
- Delete `src/strands_decider/modeling.py` entirely (the re-export shim). External consumers import from `strands_decider.config`, `strands_decider.heads`, `strands_decider.ops`, `strands_decider.torch_internals`, `strands_decider.model`.
- Delete `src/strands_decider/infer.py` (the Engine compat shim). External consumers import from `strands_decider.engine`.
- Delete `src/strands_decider/vision.py` (now a re-export shim). External consumers import from `strands_decider.images`.
- Delete `src/strands_decider/mlx_engine.py` (the MlxServingModel compat shim). External consumers import from `strands_decider.mlx_model`.
- Delete the flat `cpu_kernels.py` and `mps_kernels.py` re-exports (Step 2 added `kernels/`). External consumers import from `strands_decider.kernels.cpu` and `strands_decider.kernels.mps`.
- Update `docs/*.md`, `README.md`, and `training/README.md` to reference the new import paths.
- Update `CLI --help` text and the HF export's suggested-usage string to reference the new paths.

**Test requirements.**
- Full `pytest -q` green with no re-export shims available (any test still importing from old paths is updated or fails).
- Add `tests/test_public_api.py` that imports every documented public symbol from its final location and asserts it's accessible.

**Integration.** Final layout matches the design doc exactly.

**Demo.** A fresh checkout + `pip install -e .` lets a user import `from strands_decider.engine import make_engine` and `from strands_decider.model import TorchServingModel` without any legacy re-export working. Docs match code.

---

## Notes for anyone picking up a step

- Each step's PR should link back to this plan and the detailed design doc.
- Keep step PRs under ~1500 LOC net where possible; Step 2 and Step 8 may be larger because they move or add a lot of code.
- If PR #36 (MPS Metal inverse kernel), PR #77 (Lambda), or PR #88 (batching) merges mid-sequence, the step that lands it becomes a code move + rename. The sequence still runs end-to-end; adjust the step's "Implementation guidance" to describe the move rather than the port.
- Target branch for every PR: `main` on `strands-labs/strands-decider`.
- Each step adds exactly the tests it needs; no "testing-only" step exists later to catch up.
