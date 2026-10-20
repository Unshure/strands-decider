# training-coupling.md

Every import crossing the `strands_decider` ↔ `training` / `tests` boundary, from a `grep` sweep. This is the rename/import-rewrite surface for layout (L1).

## Imports from `training/*.py`

| File | Imports from `strands_decider` |
|---|---|
| `training/train.py` | `strands_decider.modeling` (via earlier grep: multiple symbols) |
| `training/soup.py` | `strands_decider.modeling` (multiple) |
| `training/evaluate.py` | `strands_decider.infer` (EngineConfig, SystemOneEngine); `strands_decider.modeling` (multiple); `strands_decider.schema` (derive_confidence, derive_score_confidence); `strands_decider.cli.resolve_device` (lazy) |
| `training/hf_export.py` | `strands_decider.modeling` (CONFIG_NAME, LEGACY_CONFIG_NAME, config_path) |
| `training/data/__main__.py` | `strands_decider.prompting.build_prompt` |
| `training/data/collate.py` | `strands_decider.prompting.build_prompt` |
| `training/data/format.py` | `strands_decider.schema` (ChoiceQuestion, Content, NoulQuestion, Question, ScoreQuestion) |
| `training/data/multistep.py` | `strands_decider.prompting.build_prompt` (lazy) |
| `training/data/question_transforms.py` | `strands_decider.prompting` (NOUL_DEFAULT_CRITERIA, NOUL_SLOT_LABELS) |
| `training/data/replay.py` | `strands_decider.modeling` (StrandsDeciderModel, masked_log_softmax) (lazy) |
| `training/data/sampling.py` | `strands_decider.prompting.build_prompt` |
| `training/tests/test_configs.py` | `strands_decider.modeling.StrandsDeciderConfig` |
| `training/tests/test_core.py` | `strands_decider.modeling` (masked_log_softmax, pool_last_token, apply_temperature, PointerHead, gather_options); `strands_decider.schema` (various + derive_score_confidence) |
| `training/tests/test_ddp.py` | `strands_decider.modeling` (StrandsDeciderConfig, StrandsDeciderModel, MASK_VALUE) |
| `training/tests/test_variance.py` | `strands_decider.modeling` (StrandsDeciderConfig, StrandsDeciderModel, SlotHead) |
| `training/tests/test_gemma4_train.py` | `strands_decider.modeling` (HostEmbedding, StrandsDeciderModel, StrandsDeciderConfig) |

### Symbols training imports from `strands_decider.modeling`

Flattened list:

- Classes: `StrandsDeciderConfig`, `StrandsDeciderModel`, `SlotHead`, `PointerHead`, `HostEmbedding`
- Functions: `masked_log_softmax`, `pool_last_token`, `apply_temperature`, `gather_options`, `ensure_bos`
- Constants: `MASK_VALUE`, `CONFIG_NAME`, `LEGACY_CONFIG_NAME`
- Helpers: `config_path`, `base_revision`, `checkpoint_dir`, `load_head_state`

Under layout (L1), every one of these moves to a new location. The import rewrite map:

| Old import | New location |
|---|---|
| `strands_decider.modeling.StrandsDeciderConfig` | `strands_decider.config.StrandsDeciderConfig` |
| `strands_decider.modeling.CONFIG_NAME` / `LEGACY_CONFIG_NAME` | `strands_decider.config` |
| `strands_decider.modeling.config_path` / `base_revision` / `checkpoint_dir` / `load_head_state` | `strands_decider.config` |
| `strands_decider.modeling.MASK_VALUE` | `strands_decider.config` (constants) |
| `strands_decider.modeling.PointerHead` | `strands_decider.heads.PointerHead` |
| `strands_decider.modeling.SlotHead` | DELETED (post-cleanup) |
| `strands_decider.modeling.masked_log_softmax` / `pool_last_token` / `apply_temperature` / `gather_options` | `strands_decider.ops` |
| `strands_decider.modeling.HostEmbedding` | `strands_decider.torch_internals.HostEmbedding` |
| `strands_decider.modeling.ensure_bos` | `strands_decider.torch_internals.ensure_bos` (or `strands_decider.tokenization`) |
| `strands_decider.modeling.StrandsDeciderModel` | `training.model.TrainingModel` **for training code**, `strands_decider.model.TorchServingModel` **for serving code** |

### Symbols training imports from `strands_decider.infer`

- `training/evaluate.py` imports `EngineConfig`, `SystemOneEngine`.

New location: `strands_decider.engine.EngineConfig`, `strands_decider.engine.Engine` (and the construction via `make_engine(ckpt, device, config)` factory).

### Symbols training imports from `strands_decider.schema` / `strands_decider.prompting`

Unchanged — these two modules keep their locations and public surfaces (see current-prompting-schema.md). Zero-change for these imports.

## Imports from `tests/*.py`

| File | Depends on |
|---|---|
| `test_batch_tokens.py` | `infer.EngineConfig`, `infer.SystemOneEngine`, `modeling.*`, `prompting.*`, `schema.*` |
| `test_checkpoint_load.py` | `modeling` (StrandsDeciderConfig, StrandsDeciderModel), `cli.app` |
| `test_cli_eval_devices.py`, `test_cli_exit.py` | `cli`, `modeling` (StrandsDeciderConfig, StrandsDeciderModel, config_path) |
| `test_cpu_kernels.py` | `cpu_kernels.causal_conv1d_cpu`, `infer.load_engine`, `schema.NoulQuestion`, `modeling.*` |
| `test_devices.py` | `cli`, `infer`, `server` |
| `test_engine_concurrency.py` | `infer.EngineConfig`, `infer.SystemOneEngine`, `modeling.*`, `schema.*` |
| `test_engine_dtype.py` | `infer.EngineConfig`, `infer.SystemOneEngine` |
| `test_gemma4.py` | `infer._expand_cache`, `modeling.ensure_bos`, more modeling |
| `test_hf_export.py` | `modeling.load_head_state`, `modeling.MASK_VALUE`, `modeling.StrandsDeciderModel` |
| `test_hybrid.py` | `infer.EngineConfig`, `infer.SystemOneEngine`, `modeling.StrandsDeciderModel` |
| `test_llm_client.py` | (unknown — likely client-side) |
| `test_mlx_engine.py` | `mlx_engine.merge_lora`, `infer.load_engine`, `modeling.*`, `prompting.*`, `schema.*` |
| `test_mps_kernels.py` | `mps_kernels.*` |
| `test_naming.py` | (`naming` moved out in #96 — likely stale, or stub) |
| `test_prefix_cache.py` | `infer.UnforkableCache`, `infer._expand_cache` |
| `test_server.py` | `server`, `infer._to_answer`, `prompting.render_question`, `schema.*`, `infer` |
| `test_serving_attn.py` | `modeling.serving_attn_implementation` |
| `test_vision.py` | `infer.EngineConfig`, `infer.SystemOneEngine`, `infer._option_token_index`, `modeling.*`, `prompting.render_question`, `schema.*`, `vision.*` |

Tests that import module-private helpers (`_expand_cache`, `_option_token_index`, `_to_answer`, `UnforkableCache`) are **structural tests** of exactly the pieces the refactor is moving. They must be updated when those symbols move; alternatively, keep deprecation re-exports in `strands_decider.infer` / `strands_decider.modeling` for a release.

## Package-wide re-exports to consider

To minimise external breakage, consider shim modules during migration:

```python
# strands_decider/modeling.py  (shim after the refactor lands)
from .config import (
    StrandsDeciderConfig,
    CONFIG_NAME,
    LEGACY_CONFIG_NAME,
    config_path,
    base_revision,
    checkpoint_dir,
    load_head_state,
    MASK_VALUE,
)
from .heads import PointerHead
from .ops import masked_log_softmax, pool_last_token, apply_temperature, gather_options
from .torch_internals import HostEmbedding, ensure_bos
# StrandsDeciderModel is intentionally NOT re-exported — it moved to training/.
# Internal callers should migrate to TorchServingModel or TrainingModel explicitly.
```

```python
# strands_decider/infer.py  (shim after the refactor lands)
from .engine import EngineConfig, Engine

# SystemOneEngine is deprecated; existing code gets a renamed object via a shim.
from .engine import Engine as SystemOneEngine  # last-rev compat; remove after migration
```

This bounds the churn to:
- `training/` files that use `StrandsDeciderModel` — these must switch to `training.model.TrainingModel`.
- Tests that reach into private symbols — update as the private symbols move; keep public re-exports stable for one release.

## Count the churn

- **Public symbols that stay put:** schema (everything), prompting (everything), cli (likely wraps engine differently but the module exists), server (ditto).
- **Public symbols that move:** modeling → config / heads / ops / torch_internals / model / (TrainingModel to training/). ~15 symbols.
- **Public symbols that disappear:** SlotHead, `init_head_from_lm_head`, `frozen_slot_log_probs` (unless kept for pointer-head frozen baseline), `_mm` side-channel on VisionDeciderModel, `_Readout` (replaced by MlxServingModel).
- **Tests needing updates:** every test file that imports from `infer` or `modeling` — ~15 of 19 tests. Many just update the import; a handful (prefix_cache, batch_tokens, concurrency, vision structural tests) check internals that are being reshaped and need matching updates.

## Risk flagged for the migration order (see migration.md)

1. **Training uses `StrandsDeciderModel` and `_load_torso` details** — moving training-only methods off the serving model while training code is still running means training breaks until it migrates. Mitigate by moving `TrainingModel` first (as a subclass that still holds everything today's `StrandsDeciderModel` does), then peeling serving-side methods out into `TorchServingModel`.
2. **Head state filename `slot_head.pt`** — in-tree checkpoints and published Hub checkpoints use this name. Keep for back-compat regardless of class rename.
3. **The MLX LoRA merge** has strict refusal list — easy to break silently. Pin fidelity tests on it (`test_mlx_engine.py` has these — preserve).
