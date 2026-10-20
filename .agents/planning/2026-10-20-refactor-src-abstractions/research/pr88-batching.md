# pr88-batching.md

Summary of PR #88 (`feat(serve): batch concurrent requests`) and its fit under the target architecture. Based on the PR description + today's file state.

## Scope

Files PR #88 touches today:

| File | Delta |
|---|---|
| `docs/inference.md` | +30 |
| `src/strands_decider/cli.py` | +18 / -1 |
| `src/strands_decider/infer.py` | +325 / -6 |
| `src/strands_decider/server.py` | +141 / -3 |
| `tests/test_batching.py` | +368 (new) |

The feature is almost entirely in `infer.py` + `server.py`.

## What PR #88 adds

1. **`MicroBatcher`** — a scheduler that owns the device thread, takes queued requests, packs their rows into shared forwards, scatters answers back.
2. **Handler-thread prep** — rendering, tokenisation, window-fit, option positions, kinds happen on the FastAPI handler thread before enqueue. A refusal (422 for too-long prompts) raises here, never reaches the queue.
3. **`plan_rows` planner** — sorts rows by length so similar lengths pad together; cuts forwards at `--max-batch`, `--max-batch-tokens`, a padding-waste limit (padded tokens ≤ 1.25× real past 4,096), and a quadratic cap for eager-attention Gemma (rows × longest² ≤ window²).
4. **One shared forward: `_forward_rows`** — same forward as `_slot_probs_batched` today, with rows from different requests.
5. **Error isolation.** On CUDA OOM, split the batch in halves and retry (only a row that fails alone gets the OOM). On other errors in a multi-row forward, retry row-by-row. On a CUDA device-side assert, stop launching more work.
6. **Startup tune (`tune_batching`)** — before serving, times forwards at increasing widths (1, 2, 4, ...) at 256 / 1k / 4k tokens; records the fastest rows-per-forward per token width; planner caps at those. Analogous to vLLM's profiling run.
7. **Opt-out capability flag.** `MLXEngine` and `VisionEngine` declare `can_batch_rows() = False`; the batcher never gets them. Multi-question requests take the shared-prefix path and run alone.
8. **CLI/server surface.**
   - `--no-batching` → restores one forward per request (bit-identical to main).
   - `--batch-wait-ms N` → first-request hold window (default 0, no wait).
   - `--no-batch-tune` → skips the startup tune.
   - `--max-batch` (reused), `--max-batch-tokens` (reused) → cap shared forwards.
   - `/health` reports batcher state (tuned caps, requests per cycle, max cycle size).
9. **Defaults.** Batching **on** for `--device cuda`. Tune on. No wait window.

## Headline measurements (L40S, 2B-qwen3.5)

| 256 tok | 1k tok | 4k tok |
|---|---|---|
| 49.1 dec/s (+7.7× main) | 7.77 dec/s (+2.6×) | 3.20 dec/s (+3.0×) |

Lone-request p50 unchanged (83.5 ms vs main's 85.5 ms). Burst p95 falls from 8.6 s to 0.66 s. Argmax agreement with unbatched 99.47-99.58% (flips are near ties).

## Mapping to target architecture

```
┌─────────────── Entry ────────────────┐
│ server.py (+141 LOC): handler calls  │
│   await run_in_threadpool(            │
│       engine.infer, req)              │
│ cli.py (+18 LOC): --no-batching etc.  │
└───────────────────────────────────────┘
┌─────────────── Engine ───────────────┐
│ engine.infer(req):                    │
│   prep(req, model) → PreparedRequest  │  ← PR #88's handler-thread prep lands here
│   if req multi-question:              │
│       return model.ask(state, qs)     │  ← PR #88's "runs alone" rule
│   return scheduler.submit(prepared)   │
└───────────────────────────────────────┘
┌───────────── Scheduler ──────────────┐
│ MicroBatcher.submit(prepared):        │  ← ALL of PR #88's scheduler code lives here
│   enqueue; wake device thread;        │
│   plan_rows(queued) → forwards;       │
│   for forward in forwards:            │
│       logits = model.forward_rows(f)  │
│       scatter answers                 │
│   OOM split, row-by-row retry,        │
│   lost-context stop                   │
│                                       │
│   On startup:                         │
│     tune_batching(model, config)      │
│     stores caps per token width       │
│                                       │
│   .health() → {tuned_caps, requests_  │
│       per_cycle, max_cycle_size, ...} │
└───────────────────────────────────────┘
┌─────────── ServingModel ─────────────┐
│ model.forward_rows(padded) → logits   │  ← the primitive the batcher calls
│ model.ask(state, questions) → resp    │  ← shared-prefix path, model owns KV cache
│ model.supports_batch_rows: bool       │  ← PR #88's can_batch_rows()
└───────────────────────────────────────┘
```

**Piece-by-piece mapping**

| PR #88 piece | Target module |
|---|---|
| `MicroBatcher` class | `scheduler.py` → `MicroBatcher(Scheduler)` |
| `plan_rows` helper | `scheduler.py` → private helper of `MicroBatcher` |
| OOM split / lost-context stop / row-by-row retry | `scheduler.py` → scheduler policy |
| `tune_batching` | `scheduler.py` → called by `MicroBatcher.__init__` |
| Handler-thread prep | `engine.py` → `Engine.infer` prep step |
| `_forward_rows` (padded-batch forward) | `model.py` → `TorchServingModel.forward_rows` |
| `can_batch_rows() = False` on MLX/vision | `ServingModel.supports_batch_rows: bool` |
| Multi-question "runs alone" | `engine.py` short-circuits to `model.ask(...)` |
| `/health` batcher fields | `Engine.health()` delegates to `scheduler.health()` |
| CLI flags | `EngineConfig` fields; `make_engine()` factory picks `MicroBatcher` vs `NoScheduler` |

## Points of care when porting PR #88 under the refactor

1. **The `_forward_rows` extraction.** PR #88 already factors a shared forward out of the batched path. The refactor renames it to `TorchServingModel.forward_rows` — a one-line move once the engine ownership is clean.
2. **Tune's black-box forward measurement.** `tune_batching` times `model.forward_rows(rows, cache=None)` at target widths. Preserve: don't let it depend on scheduler internals (so a future MicroBatcher2 or an alternative scheduler can reuse it).
3. **Handler-thread prep must raise 422 cleanly.** Over-window + `strict_window` → `ValueError` on the handler thread. Keep this discipline in `Engine.prep` — the scheduler must only see tensor-ready rows.
4. **Shared-prefix path still runs alone.** Preserve: multi-question requests do not enter `MicroBatcher`. The Engine short-circuits them to `TorchServingModel.ask`.
5. **`/health` is a public contract.** Tests and ops dashboards read specific field names. Preserve them when `Engine.health()` composes `model` + `scheduler` info.
6. **Tune skips on OOM.** `tune_batching` stops at the first forward that runs out of memory — doesn't fail startup. Preserve.
7. **`can_batch_rows() = False` on vision in the current PR = `TorchServingModel.supports_batch_rows = False` when `vision_tower is not None`.** The composition-with-vision shape in the refactor means vision is still unbatched by default, same as PR #88.

## Branch state

- PR #88 branch head commit (per PR description): `65fd7bb`. Later commits change default flags and names, nothing in batching code.
- Not checked out locally. Per user: "description is good enough."

## Risk of ordering vs. refactor

Two cases:

- **If PR #88 merges before the refactor lands:** the refactor is a code move + rename, with PR #88's logic preserved. Lower-risk individual steps; parity tests pin agreement.
- **If the refactor lands first:** PR #88 is rewritten as a `MicroBatcher` on the new `Scheduler` Protocol — a cleaner diff, no `infer.py` to grow by 325 LOC. Preferred, if the PR author is OK rebasing.

See migration.md for the ordering recommendation.
