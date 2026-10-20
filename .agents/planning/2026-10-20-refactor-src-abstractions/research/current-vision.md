# current-vision.md

Walk of `src/strands_decider/vision.py` (575 LOC). Mixes image preprocessing, a `StrandsDeciderModel` subclass with multimodal torso, and a `SystemOneEngine` subclass with the image forward.

## Three sections in one file

1. **Prompt layout** — image placeholders in `<state>`, token-expansion helpers.
2. **Image I/O** — base64 decode, format allowlist, EXIF normalisation, size cap, resize.
3. **Model + Engine** — `VisionDeciderModel(StrandsDeciderModel)` + `VisionEngine(SystemOneEngine)`.

## Prompt layout (torch-free, pure Python)

- `render_image_state(state, n_images)` — wraps `render_state(state)` and inserts `<|vision_start|><|image_pad|><|vision_end|>` placeholders at the top of `<state>`. Images come first so the vision tower + image-token portion of the prefix is identical across questions.
- `expand_image_tokens(prompt, tokens_per_image)` — expands each `<|image_pad|>` to N copies, matching the processor's expansion so the tokenizer's offsets map onto the text the model sees.
- `image_tokens_for_grid(grid_thw, merge_size)` — `t*h*w // merge_size**2`.

PR #77 already extracted these to `src/strands_decider/images.py`. Target: stays as backend-neutral image prep, callable from any ServingModel that handles images.

## Image I/O

- `decode_image(data, max_pixels=4096*4096)` — base64 → PIL. Format allowlist (`PNG|JPEG|WEBP|GIF`); pre-decode pixel-count check (header-only); EXIF transpose for phone-rotated photos; alpha → white composite.
- `fit_image(img, long_side)` — scale **down** only; `BICUBIC`. `long_side=0` is a no-op.
- `_pil()` — raises a friendly ImportError if Pillow is missing.

Target: `images.py` (already extracted in PR #77).

## `VisionDeciderModel(StrandsDeciderModel)` — the model subclass

Overrides:

| Method | What it changes |
|---|---|
| `hidden_size` (staticmethod) | Reads `torso.config.get_text_config().hidden_size` (composite config) |
| `is_hybrid` (staticmethod) | Reads layer_types from the text config |
| `_load_torso` (staticmethod) | `Qwen3_5ForConditionalGeneration` (kept whole: ViT + merger + decoder); refuses non-qwen3_5 model_type; `visual.requires_grad_(False)` |
| `attach_lora` | LoRA targets regex-scoped to `^language_model\.`, so the ViT's own projections aren't adapted |
| `encode` | Adds `pixel_values`, `image_grid_thw`, `position_ids` kwargs; computes `mm_token_type_ids` |
| `forward` | Sets `self._mm = (pixel_values, image_grid_thw)` **as a side channel**, calls `super().forward(...)`, resets `_mm` in `finally` |
| `frozen_slot_log_probs` | Same side-channel trick to inject image tensors |
| `load` (classmethod) | Serves-only (`trainable=True` raises); refuses non-pointer heads; LoRA key remap `base_model.model.X` → `base_model.model.language_model.X`; uses `set_peft_model_state_dict` instead of `PeftModel.from_pretrained` |

**Red flag — the `_mm` side-channel:** the parent `StrandsDeciderModel.forward` doesn't know about images, so the subclass stuffs `(pixel_values, image_grid_thw)` into a class attribute and reads them back inside the overridden `encode`. This is a code smell born of trying to inherit a forward whose signature wasn't designed for multimodal inputs.

**In the composition target shape:** the vision tower is an **optional submodule** on `TorchServingModel`. `forward_rows(ids, mask, cache?, images=None, image_grid_thw=None)` carries images through its own signature. No side channel, no subclass, no `_mm` state. `TorchServingModel.handles_images = bool(self.vision_tower)`.

**LoRA key remap for vision:** when a text-trained adapter is loaded onto the multimodal torso, keys shift. Preserve this mapping in `TorchServingModel.load(..., vision=True)`.

**Non-trivial constraint:** vision needs **transformers ≥ 5.18** (`_check_transformers()` enforces). Preserve as an import-time check.

## `VisionEngineConfig(EngineConfig)` — extra knobs

- `image_long_side: int = 448` — fixed per deployment so the token budget is predictable (448 px square ≈ 196 tokens at patch=16, merge=2).
- `max_images: int = 4` — request-level cap.
- `max_image_pixels: int = 4096*4096` — pre-decode cap.

Target: these roll into `EngineConfig` as optional fields `image_long_side`, `max_images`, `max_image_pixels` (ignored when `model.handles_images` is False).

## `VisionEngine(SystemOneEngine)` — the engine subclass

Overrides `__init__` (adds image processor; pins `Qwen2VLImageProcessorPil`, not the torchvision variant whose pixel values flip some answers), `evaluate` (text → super; image → `_image_probs`), and adds `_reset_positions`, `ask_images`, `_fit_images`, `_image_probs`.

### `_reset_positions()`

Clears `torso.rope_deltas` because the text path leaves positions to transformers, which would read a `rope_deltas` stored by any earlier image forward (another training/eval step on the same model object). **Cross-request state leak** that the current engine works around by clearing before every request.

**In the target shape:** `TorchServingModel.forward_rows` **always** passes `position_ids` explicitly (image path and text path alike). No stored `rope_deltas`, no reset needed. This is actually a correctness win from the refactor.

### `_fit_images(state, images, question_texts)`

Image-aware window fitting:
1. Tokenise questions, measure longest.
2. Run image processor → `pixel_values` + `image_grid_thw`. Compute token counts via `image_tokens_for_grid`.
3. Build prompt with placeholders expanded; find the vision-end token position (`keep`).
4. If `strict_window` and `len(s)+longest > max_len` → raise.
5. Else: question reserve first, then enforce image budget (`keep ≤ budget` else raise), then truncate state tail at `budget`.

Target: `Engine.prep` grows an image branch; or `TorchServingModel.ask(state, questions, images)` runs this internally. Scheduler never sees images (vision backends don't batch).

### `_image_probs(state, images, rendered)` — the inference core

M-RoPE handling makes this different from text:

- `base.get_rope_index(prefix_ids, mm_token_type_ids, image_grid_thw)` returns multimodal positions `mpos` and a `rope_delta` (negative; the text-after-image offset).
- Prefix forward passes `position_ids = cat([text_pos, mpos], dim=0)` — a `[4, 1, n]` tensor: text + 3 M-RoPE rows.
- Cache uses `_expand_cache` from `infer.py` (same fork logic).
- Suffix forward passes `position_ids = cat([st, (st + delta).expand(3, m, -1)], dim=0)`.

The torso is called with `pixel_values`, `image_grid_thw`, `mm_token_type_ids` on the prefix pass; cache is forked; suffix pass just forwards suffix ids with the explicit position tensor. Head + temperature + mask applied like the text path.

**Target:** this is a vision-specific shared-prefix implementation. In the composed shape, `TorchServingModel.ask(state, questions, images)` runs this when `images` is non-empty. The vision tower is a submodule the forward uses conditionally.

## `load_vision_engine`

Mirrors `load_engine`: constructs `VisionEngineConfig` from a plain `EngineConfig` + `image_long_side`, loads `VisionDeciderModel`, returns `VisionEngine`. Target: `make_engine(ckpt, device, config, vision=True)` constructs `TorchServingModel.load(ckpt, vision=True)` + whichever scheduler.

## Target module split for vision.py

| Current piece | New home |
|---|---|
| `VISION_START`, `IMAGE_PAD`, `VISION_END`, `MIN_TRANSFORMERS`, `_check_transformers` | `strands_decider.images` (PR #77 direction) |
| `render_image_state`, `expand_image_tokens`, `image_tokens_for_grid` | `strands_decider.images` |
| `_pil`, `IMAGE_FORMATS`, `decode_image`, `fit_image` | `strands_decider.images` |
| `mm_token_type_ids`, `qwen_base` | `strands_decider.torch_internals` (torch-only) |
| `VisionDeciderModel` (hidden_size, is_hybrid, _load_torso, attach_lora, encode, forward, frozen_slot_log_probs, load) | Merged into `TorchServingModel` as the `vision=True` branch; vision tower becomes an optional submodule, `_mm` side channel deleted |
| `VisionEngineConfig` fields | Added to `EngineConfig` as optional |
| `VisionEngine._reset_positions` | Deleted — becomes unnecessary when positions are always explicit |
| `VisionEngine._fit_images` | Merged into `Engine.prep` with an image branch (or into `TorchServingModel.ask` for the image shared-prefix path) |
| `VisionEngine._image_probs` | Becomes the image path of `TorchServingModel.ask` |
| `load_vision_engine` | Deleted — `make_engine(ckpt, device, config, vision=True)` handles it |

## Surprises / gotchas for the refactor

1. **The `_mm` side channel is removable.** The current pattern passes images via a class attribute because the parent `forward` has no image kwargs. The refactor adds image kwargs to `forward_rows` and the attribute goes away.
2. **`rope_deltas` cross-request leak.** Today's `_reset_positions` is a workaround. Always-explicit positions in the refactor removes the need.
3. **LoRA key remap for multimodal.** `base_model.model.X` → `base_model.model.language_model.X`. Preserve inside `TorchServingModel.load`.
4. **`Qwen2VLImageProcessorPil`, not `Qwen2VLImageProcessor`.** The torchvision backend flips answers; PIL is what every measurement was taken on. Keep pinned.
5. **Vision requires transformers ≥ 5.18.** Preserve the version check at import or load time.
6. **`VisionEngine` text-only requests delegate to `super().evaluate`.** Preserve: a vision-capable server must still serve plain text requests through the identical path.
7. **Vision does not batch (per PR #88).** `TorchServingModel(vision=True).supports_batch_rows = False`. The Engine factory picks `NoScheduler` (or `SerialiseScheduler` on MPS) for a vision model.
8. **`_image_probs` reuses `_expand_cache` from `infer.py`.** When cache logic moves into `TorchServingModel`, the vision image path finds it on the same model.
