# current-prompting-schema.md

Walk of `src/strands_decider/prompting.py` (187 LOC) and `src/strands_decider/schema.py` (198 LOC). Both are pure-Python, backend-neutral, no torch/MLX dependencies. **Target: both stay essentially as-is.**

## `schema.py` — the wire types

- Pydantic BaseModels defining the public API shape (matches the Jev API).
- `Content = str | dict | list` — a state or instructions.
- `NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion` → `Question` union (discriminated by `type`).
- `SystemOneRequest(state, questions, model, images)`.
- `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer` → `Answer` union.
- `SystemOneResponse(model, answers, usage)`, `Usage(input_tokens, output_tokens)`.
- Constants: `MAX_CHOICE_OPTIONS = 255`, `MIN_SCORE_LEVELS = 2`, `MAX_SCORE_LEVELS = 10`.
- Validators enforce option/level counts; `NoulQuestion.criteria` keys must subset `{"true", "false"}`.
- Two public confidence formulas:
  - `derive_confidence(probs)` — normalised max probability `(N*p_max - 1)/(N - 1)`, scale-independent.
  - `derive_score_confidence(probs, ordinal_smoothing)` — ordinal σ-based; accounts for ordinal_smoothing floor so a correctly-fitted model can reach confidence 1.0.

**No changes needed.** Backend-neutral, no churn. Imported by every entry point and every backend.

## `prompting.py` — rendering and readback

### Rendering

- `render_content(c)` — strip strings; `json.dumps(indent=2, ensure_ascii=False, sort_keys=False)` for dicts/lists. **Deterministic**: the same state must tokenise identically so the shared-prefix cache stays sound.
- `RenderedQuestion` dataclass (frozen): `text`, `slot_labels`, `slot_descriptions`, `kind`, `option_spans`.
- `_option_block(pairs)` — numbered "1. name — desc" lines, one option per line (so pointer-head char spans are unambiguous).
- `render_question(q, option_order?)` — dispatches Noul/Choice/Score, applies optional permutation, assembles `<question>...<options>...<answer>` with per-option char spans.
- `render_state(state)` — `<state>\n{content}\n</state>\n`. **The shared prefix.**
- `build_prompt(state, question)` — convenience.

### Readback (slot distribution → typed answer)

- `read_noul(probs, rq)` — `P(true)` regardless of which slot it landed in.
- `read_choice(probs, rq)` — `{name: p}`.
- `score_legend(rq)` — `{level: rubric}` in canonical ascending order (undoes permutation).
- `read_score(probs, rq)` — expected level + distribution by level.

### Why generic slots work

Head k has no fixed meaning — it scores "whatever option was listed at position k in the prompt." Two consequences on display here:
1. The prompt must number options explicitly (`_option_block`).
2. Training must shuffle option order (otherwise heads memorise "slot 0 ≈ positive class").

The readback helpers exist precisely to undo the (possibly permuted) rendering so callers see options back the way they declared them.

### PR #77 moved these readouts here

PR #77 extracted readback functions from `infer.py` into `prompting.py` (+75 LOC note from its diff). This is already the right place — the readouts are backend-neutral and the file is called `prompting`, which is where "how to interpret the slot distribution" belongs. The refactor keeps this.

## Target placement in the new layout

```
strands_decider/
  schema.py       # unchanged
  prompting.py    # unchanged (keeps PR #77's additions)
```

Both are imported by:
- Every ServingModel (`TorchServingModel`, `MlxServingModel`, `LlamaCppServingModel`) — rendering.
- The Engine — for `_to_answer`-equivalent response assembly (today in `infer.py`'s `_to_answer`; moves here or to `strands_decider.response`).
- The CLI and Server — wire types for request/response bodies.

## Candidate micro-moves for cleanliness (optional)

- `_to_answer` and `_option_token_index` from `infer.py` could land in `prompting.py` since they operate on `RenderedQuestion` and shared schema types. PR #77 already moved related readouts here; completing the migration makes `prompting.py` the "turn requests into text / turn probs into typed answers" module.
- Alternatively, split into `prompting.py` (render) + `response.py` (readback + `_to_answer`) if `prompting.py` grows unwieldy. Not needed today (187 LOC).

## Surprises / gotchas for the refactor

1. **`render_content` must stay deterministic.** The shared-prefix cache soundness depends on it. Any change to JSON encoding would invalidate caches silently.
2. **`option_spans` are character offsets, not token indices.** The engine maps them to tokens via the tokeniser's `return_offsets_mapping`. The refactor preserves this contract; `_option_token_index` reads these spans.
3. **Permutation lives in `render_question(..., option_order=...)` only.** The readbacks undo it. Training augments option order through this parameter; serving passes `None`.
4. **`<answer>` is the pooling position.** Prompt layout is load-bearing: lose `<answer>` and the head reads an arbitrary token. The window-fitting code (`_fit`) front-truncates questions precisely to preserve this.
