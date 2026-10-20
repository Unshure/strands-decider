# Idea Honing

## Q1: Which abstractions feel hardest to follow today?

The `src/strands_decider/` package has 13 modules (4191 total LOC). The largest are:
- `hf_export.py` (921)
- `modeling.py` (746)
- `infer.py` (575)
- `vision.py` (575)
- `mlx_engine.py` (278)
- `cli.py` (229)

Which of these feel confusing, tangled, or at the wrong altitude? Any specific function/class names that come to mind? Or is the pain more about cross-module boundaries (e.g., `infer` ↔ `modeling` ↔ kernels)?

### A1

The abstractions aren't legible to begin with. Target architecture:

- **Entry (CLI + Server)**: take requests, format them into the shape the engine accepts. Server can batch concurrent requests.
- **Engine**: takes N requests, batches them with caching for efficient execution.
- **Kernels**: installed to help improve engine inference.

Also check whether this abstraction accommodates in-flight PRs:
- strands-labs/strands-decider#36
- strands-labs/strands-decider#88
- strands-labs/strands-decider#77 (Lambda hosting)

## Q2: What's today's actual layering, as you see it?

Before researching the PRs, let me confirm the current state vs. the target:

**Current (as I read it):**
- `cli.py` — entry (typer app, calls `infer.SystemOneEngine` directly)
- `server.py` — entry (FastAPI; constructs one global `SystemOneEngine`; appears to serve one request at a time, no cross-request batching)
- `infer.py` (575 LOC) — `SystemOneEngine` + `EngineConfig`; prefix-cache logic, batching *within* one request's N questions
- `modeling.py` (746) — model class, kernels glue (`apply_temperature`, `pool_last_token`, etc.)
- `cpu_kernels.py` / `mps_kernels.py` — device-specific kernels
- `mlx_engine.py` (278) — a parallel engine for Apple MLX
- `vision.py` (575) — vision adapter
- `prompting.py`, `schema.py`, `naming.py`, `hf_export.py` — supporting

**Two things the target implies that aren't there today:**
1. Server-side **cross-request batching** (continuous batching / request queue).
2. One `Engine` abstraction with `mlx_engine` as a backend behind it (rather than a parallel module).

**Question:** Is that read correct? Specifically — are (1) cross-request batching and (2) unifying `infer.SystemOneEngine` + `mlx_engine` behind one interface **in scope** for this refactor, or do you want to keep the refactor to re-slicing existing code and documenting the layering, deferring new capability to follow-ups?

### A2

1. Cross-request batching should be **considered for the abstraction** (the design must accommodate it, even if the first implementation defers).
2. Yes — unify behind one Engine interface with torch and MLX as backends.

## Q3: What's the Engine's public interface — one-request or batch?

The target "Engine takes N requests, batches with caching" could land as:

- **(a) Single-request API with internal queue.** `engine.infer(request) -> response` is the only call. The Server posts requests; the Engine's internal scheduler batches them across callers (continuous batching). Entry layers never see batches. Natural fit for FastAPI's async handlers and for Lambda (PR #77) where each invocation is one request.
- **(b) Explicit batch API.** `engine.infer_batch([req1, req2, ...]) -> [resp1, resp2, ...]`. The Server does its own small-window batching before calling. Simpler engine; batching policy visible to the entry layer.
- **(c) Both.** Engine exposes `infer_batch`; a thin `RequestScheduler` wraps it with a queue to offer `infer`. Entry layers can pick which to use (Server uses `infer`, CLI uses `infer_batch` directly for `--batch` CLI mode).

Which shape do you want? (c) is the most flexible but adds a layer; (a) is cleanest for the stated goal but hides the batch boundary from tests/CLI; (b) is the minimum rework from today.

### A3

**(a) Engine owns batching; public API is single-request.** Rationale:

- Batching strategy is a property of the compute backend (CUDA wants micro-batched padded forwards; MLX runs on its own scheduler; CPU serialises). The engine has that info; the server doesn't.
- Two kinds of batching exist in the engine today (shared-prefix within a request; cross-request micro-batching in PR #88). Keeping both under the engine means one place reasons about "fork KV cache vs. pack rows vs. serialise."
- Entry layers become uniform: Server, CLI, Lambda all do `request -> engine.infer(request) -> response`.
- The engine picks a `BatchingPolicy` at construction based on backend + deployment mode: `none` (Lambda/CLI), `micro_batch` (CUDA server), `serialize` (MPS). Lambda (PR #77) degrades to queue-depth-1 without special-casing entry.

## Q4: Where do the Engine's internal pieces split?

The engine's job is "take requests, run them efficiently on this hardware." Inside the engine, there are at least these concerns today:

1. **Prepare** (CPU, no device) — render prompt, tokenise, fit to window, locate option positions. Can run on the request's thread; PR #88 already does.
2. **Schedule / batch** — the `MicroBatcher` from PR #88: queue, planner, OOM retry, lost-context stop.
3. **Model forward** — `_forward_rows` / `_slot_probs_batched`: run the torso + heads, read answers.
4. **Backend** — torch (CUDA/MPS/CPU) with `{cpu,mps}_kernels.py`, or MLX (today a parallel engine), or vision (today a parallel engine too).
5. **Model object** — `StrandsDeciderModel` in `modeling.py` (746 LOC): the torso + choice/noul/score heads + utilities (`apply_temperature`, `pool_last_token`, `masked_log_softmax`).

Options for where to draw lines:

- **(α) Engine = scheduler; Backend = model + forward + kernels + batching policy.** One `Backend` interface with `TorchBackend`, `MlxBackend`, `VisionBackend` implementations. Engine is a thin facade: it owns the queue (or not), delegates every forward to the backend. Model stays inside the backend. Kernels are a Torch-backend internal detail. Clean compute-boundary story (your framing).
- **(β) Engine = model + scheduler; Backend = just a forward primitive.** Engine owns the shared-prefix logic and heads; backend only implements "run this packed forward and return logits." Smaller backend surface, but more work in the common engine, and vision has to inject itself at the forward level.
- **(γ) Keep roughly today's shape, just rename.** `SystemOneEngine` grows a `batching_policy`, `mlx_engine.py` becomes `MlxEngine` under the same interface, vision becomes `VisionEngine`. Minimal rework; the "Backend" layer is implicit (= the engine subclass).

Which split do you want? I lean (α) — it most closely matches your framing ("engine knows the underlying compute architecture") and gives vision + MLX + torch a shared interface at the right altitude. (γ) is cheaper now but kicks the can on vision/MLX duplication.

### A4

**Shape 2: two axes — Backend × Scheduler.** Modality lives *inside* the backend because the vision tower is a weight-bearing module tied to the array lib. Shared pure-Python prep helpers (image decode, token expansion) are called by both backends. MLX must be able to grow vision support under the same backend.

```
Engine(backend, scheduler)
  Backend
    TorchBackend  — forward, cache, kernels (cpu/mps/cuda), text + vision
    MlxBackend    — forward, cache, text + vision (future)
  Scheduler
    NoScheduler, MicroBatcher, Serialise
```

Vision is a backend capability (`backend.handles_images`) + shared `vision_prep` helpers.

## Q5: Where does the Model live, and what's the Backend interface?

`StrandsDeciderModel` (modeling.py, 746 LOC) today holds the torso + choice/noul/score heads + small utilities (`apply_temperature`, `pool_last_token`, `masked_log_softmax`). `VisionDeciderModel` subclasses it with a vision tower.

Two sub-questions, since they couple:

**(i) Does Model stay a first-class thing, or get folded into Backend?**

- **(A) Backend owns the model.** `TorchBackend` loads weights, holds the torso and heads, exposes `backend.forward(rows, cache)`. There is no shared `StrandsDeciderModel` class across backends — each backend has its own because the weights and tower implementations are backend-specific anyway.
- **(B) Model stays, Backend is thin.** `StrandsDeciderModel` remains the first-class torch module (what training produces and HF export reads). `TorchBackend` wraps it and adds cache/forward orchestration. `MlxBackend` has its own `MlxModel` (today's `_Readout` + MLX decoder, cleaned up).

(A) is simpler internally but makes "the model" an implementation detail — awkward for training and HF export, which legitimately need the torch nn.Module. (B) keeps a public Model concept, which training already depends on.

I lean **(B)**: Model stays, because training and HF export need a torch nn.Module and today's `StrandsDeciderModel` is already that interface. Backend becomes a serving-side wrapper.

**(ii) Backend interface sketch — what does `Backend` expose?**

```python
class Backend(Protocol):
    config: BackendConfig  # device, dtype, window, etc.
    handles_images: bool  # modality capability flag
    supports_batch_rows: bool  # scheduler compatibility flag

    def make_cache(self) -> KVCache: ...
    def encode_state(self, tokens, images=None) -> KVCache: ...  # for shared-prefix path
    def forward_rows(self, rows: PaddedBatch, cache: KVCache | None) -> Logits: ...
    def read_answers(self, logits, option_positions, kinds, temperatures) -> list[Answer]: ...
```

Plus one factory: `load_backend(checkpoint, device, dtype, vision=False) -> Backend`.

Is this the right surface, or missing something? Specifically:
- Should `read_answers` live on the Backend (where logits are produced) or on a shared `AnswerReader` that both backends call? (Today the logic is tangled into `_slot_probs_*`.)
- Should `encode_state` be separate from `forward_rows`, or is "forward with a cache" enough? (The shared-prefix path currently has its own method.)

### A5

- **Mental model of Model:** the underlying LLM (Qwen / Gemma) **plus the pointer head glued on top**.
- **Scope cleanup:** `SlotHead` is no longer used — remove it from this codebase.
- Open sub-question: how does Model stay a coherent cross-backend abstraction while still carrying torch-only training features (LoRA, freeze, training forward)?

## Q6: Model vs. Backend — one concept, or two?

Given your framing ("Model = torso + pointer head"), two options:

**(P1) Keep Model and Backend as separate concepts.**

- `Model` = weights (torso + pointer head) + tokenizer + encode. Pure architecture.
- `Backend` = serving wrapper around a Model: owns the KV cache, device, batching readiness, exposes `forward_rows`, `encode_state`, `read_answers`.
- Training imports `StrandsDeciderModel` (torch) directly — it never touches Backend.
- MLX has its own `MlxModel` + `MlxBackend`.

Pro: training-only concerns (LoRA, freeze) live on the torch Model and never leak into the serving-side Backend protocol. Clean separation of "what the weights are" from "how we run them."
Con: four classes to track (TorchModel, TorchBackend, MlxModel, MlxBackend). Boundary between Model and Backend is thin — Backend mostly delegates to Model.

**(P2) Collapse — Model *is* the backend.**

- `Model` is the one concept. It holds weights, knows its device/array lib, exposes cache ops and forward.
- `StrandsDeciderModel` (torch): implements the full serving surface + inherits `nn.Module` + carries training hooks (`attach_lora`, `freeze_torso`, `forward` for training). Training code uses it directly as today.
- `MlxModel`: implements the same serving surface, no training.
- No separate `Backend` class. Engine composes a `Model` with a `Scheduler`.

Pro: fewer concepts. Matches your framing ("Model = underlying LLM + pointer head") exactly; "Model" is the single word for the abstraction. Training is a torch Model's extra surface, not a cross-cutting layer.
Con: the torch Model class carries both serving and training responsibilities. The serving-side protocol (`forward_rows`, `read_answers`, etc.) is defined on something that also has `attach_lora` on it. Mitigation: a serving `Protocol` (structural interface) sits above both torch and MLX models and is what the Engine depends on, while training depends on the concrete torch class.

**How training stays clean under (P2):**

```
StrandsDeciderModel(nn.Module)         # torch concrete; the public training class
  - config, tokenizer, torso, pointer_head
  - encode(text) -> ids
  - forward_rows(ids, mask, cache?) -> hidden     # serving use
  - read(pooled, option_positions) -> logits      # pointer-head apply
  - forward(ids, mask, options) -> loss           # training only
  - attach_lora(), freeze_torso(), trainable_parameters()  # training only
  - save_pretrained(), load()

MlxModel                                # mlx concrete; serving-only
  - config, tokenizer, mlx_torso, pointer_head (torch)
  - encode, forward_rows, read

ServingModel(Protocol)                  # structural interface the Engine uses
  - config, tokenizer, encode, forward_rows, read, handles_images, supports_batch_rows
```

Engine depends on `ServingModel`. Training depends on the concrete `StrandsDeciderModel`. HF export depends on the concrete `StrandsDeciderModel`.

My call: **(P2)** matches your framing and keeps the concept count low. The `ServingModel` Protocol gives the Engine a clean serving-only surface without inventing a parallel `Backend` class.

Does (P2) land, and does the Model↔training separation above feel right (training hooks on the concrete torch class, serving on a Protocol)?

### A6

**(P2) with a crisper split: two separate concrete classes, `ServingModel` and `TrainingModel`.**

- `ServingModel`: Protocol (encode, forward_rows, read, config, tokenizer, capability flags). Concretes: `TorchServingModel`, `MlxServingModel`.
- `TrainingModel`: nn.Module. Carries torso + pointer head + LoRA + training forward + checkpoint I/O.
- The connection between them is the **on-disk checkpoint format**, not a class hierarchy.
- `SlotHead` is deleted (not used anymore).

## Q7: Where does `TrainingModel` live, and what moves out of `src/strands_decider/modeling.py`?

Today, `src/strands_decider/modeling.py` (746 LOC) contains everything: config, heads (`SlotHead` + `PointerHead`), tensor ops, `HostEmbedding`, `StrandsDeciderModel` (training+serving mix), checkpoint path helpers. The `training/` directory sits outside `src/` (per the CLI docstring: "Training, evaluation, calibration and corpus-building live outside this package.") and imports heavily from `modeling.py` (confirmed in `training/train.py`, `training/evaluate.py`, `training/soup.py`, and the training tests).

Two layouts:

**(L1) `src/` is serving-only. `TrainingModel` moves to `training/`.**

```
src/strands_decider/
  config.py          # StrandsDeciderConfig, config path helpers
  heads.py           # PointerHead (SlotHead deleted)
  ops.py             # pool_last_token, masked_log_softmax, apply_temperature,
                     # gather_options, HostEmbedding, MASK_VALUE
  model.py           # ServingModel Protocol + TorchServingModel
  mlx_model.py       # MlxServingModel
  vision/            # vision prep helpers; VisionServingModel
  kernels/           # cpu_kernels, mps_kernels (torch backend internals)
  engine.py          # Engine facade + scheduler composition
  scheduler.py       # MicroBatcher, NoScheduler, Serialise
  prompting.py       # unchanged
  schema.py          # unchanged
  server.py          # FastAPI
  cli.py             # typer
training/
  model.py           # TrainingModel (today's StrandsDeciderModel), imports heads/ops/config
  train.py, evaluate.py, soup.py, hf_export.py  # imports from training.model
```

Pro: the `strands_decider` package becomes a serving library. "Model" in the package means `ServingModel`. Matches the CLI docstring's intent. HF export already lives in `training/` after #96.

Con: training now has one more import path (`training.model` vs `strands_decider.modeling`); every training file needs an import rewrite. External users who have `strands_decider.modeling.StrandsDeciderModel` in their code break.

**(L2) Everything stays in `src/`, just reorganised.**

```
src/strands_decider/
  config.py
  heads.py
  ops.py
  training_model.py  # TrainingModel (today's StrandsDeciderModel), nn.Module
  model.py           # ServingModel Protocol + TorchServingModel
  mlx_model.py
  vision/
  kernels/
  engine.py, scheduler.py, server.py, cli.py, prompting.py, schema.py
```

Pro: smaller blast radius for imports. External `strands_decider.training_model` is one rename.
Con: the package carries training-only code, blurring the "src is serving" story the CLI docstring already set up.

My call: **(L1)**. The CLI docstring already says training lives outside the package; completing that move is a one-time import rewrite in `training/` files, and the `strands_decider` package becomes clean (serving + shared primitives only).

Which layout?

### A7

- **Layout: (L1).** `strands_decider` package is serving + shared primitives; `training/` holds `TrainingModel` and all training scripts.
- **Model/Training shape — for now:** `TrainingModel(TorchServingModel)` — class-based extension. Defer the `Trainer`-Protocol/composition refactor to a later pass.

**Caveat noted for the design doc:** the `ServingModel.forward_rows -> torch.Tensor` contract bakes in "head stays torch." If an `MlxTrainer` ever ships, the Protocol will need to loosen to backend-native tensors (and `PointerHead` ported to MLX). Not a problem today.

## Q8: Engine + Scheduler — final interface shape

```python
# strands_decider/engine.py
class Engine:
    def __init__(self, model: ServingModel, scheduler: Scheduler, config: EngineConfig): ...
    def infer(self, request: SystemOneRequest) -> SystemOneResponse: ...
    def health(self) -> dict: ...
    def shutdown(self) -> None: ...


# strands_decider/scheduler.py
class Scheduler(Protocol):
    def submit(self, prepared: PreparedRequest) -> SystemOneResponse: ...
    def health(self) -> dict: ...
    def shutdown(self) -> None: ...


class NoScheduler(Scheduler):  # CLI, Lambda, --no-batching
    """Serve each request alone."""


class MicroBatcher(Scheduler):  # CUDA server (PR #88)
    """Cross-request queue + planner + OOM retry + tune. Owns device thread.
    Requires model.supports_batch_rows."""


class SerialiseScheduler(Scheduler):  # MPS
    """One forward at a time, no cross-request packing (issue #9)."""
```

Engine construction picks the scheduler from (model, deployment mode, config).

Three sub-questions:

**(a) Prep location.** `PreparedRequest` (rendered prompt + ids + option positions + kinds + temperatures) — produced by the Engine before `scheduler.submit`, or by the Scheduler on the request thread (as PR #88 does)?
I'd put it on the **Engine**: prep is pure CPU and moving it off the device thread is a win for every scheduler.

**(b) Shared-prefix cache ownership.** One state, N questions — Engine or ServingModel owns the prefix path?
I'd put it on the **ServingModel**: KV cache layout is backend-specific. Engine calls `model.ask(state, questions)`; internally does shared-prefix if N>1. Scheduler just sees "run these rows."

**(c) Sync vs async.** `Engine.infer` fully sync (behind a thread for FastAPI), or returns a future?
I'd keep **Engine.infer sync**. Server wraps it in `run_in_threadpool`. CLI and Lambda call it directly.

OK with (a), (b), (c)?

### A8

OK with all three.










