# current-kernels.md

Walk of `src/strands_decider/cpu_kernels.py` (93 LOC) and `src/strands_decider/mps_kernels.py` (195 LOC). Both monkey-patch `transformers.models.qwen3_5.modeling_qwen3_5` to replace reference ops with device-specialised implementations.

## Shared pattern

```
_installed = False

def install() -> bool:
    if _installed: return True
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    except ImportError:
        return False
    current = q.<reference_fn>
    def dispatch(x, *a, **kw):
        if x.device.type == <my_device>: return my_impl(x, *a, **kw)
        return current(x, *a, **kw)
    dispatch.__wrapped__ = current
    q.<reference_fn> = dispatch
    _installed = True
    return True
```

Both:
- Idempotent via module-level flag.
- No-op if transformers.qwen3_5 isn't importable.
- Preserve the reference fn via a closure dispatch for other devices.
- Mark the wrapped function with `__wrapped__` for inspection.

## `cpu_kernels.causal_conv1d_cpu`

**Target function:** `causal_conv1d_fn` in transformers' qwen3_5 (used by Gated DeltaNet q/k/v projections, kernel width 4).

**Why:** without the `causal_conv1d` CUDA extension, transformers falls back to `F.conv1d(..., groups=channels)`. On CPU this becomes one `_slow_conv2d_forward` per channel: 6,144 channels × 18 layers = 110,592 calls per forward. Profiled on M3 Pro: 2.0 of 2.7 s for a 70-token question.

**What it does:** manual width-4 depthwise causal conv as four shifted multiply-adds over `[B, L, C]` (channels-last for cache locality). 8 ms for all 18 layers vs. 2,026 ms. Accumulates in fp32 for half-precision weights (`addcmul_`). Agrees with the reference to float rounding.

**Also faster on x86 Linux** (oneDNN grouped conv), by 1.1-4× per layer.

## `mps_kernels.chunk_gated_delta_rule_mps`

**Target function:** `torch_chunk_gated_delta_rule` in transformers' qwen3_5.

**Why:** on macOS neither `flash-linear-attention` (Triton, Linux only) nor `causal_conv1d` (CUDA) installs, so the Qwen3.5 torso falls back to reference PyTorch. The reference runs two `torch.linalg.solve_triangular` per layer; on MPS each is ~20 ms for a `[1, 16, 16, 64, 64]` system = 806 ms of 1,739 ms per forward at 1,024 tokens.

**What it does:** replaces the two solves with one block inversion applied twice. The system `I + N` has strictly-lower-triangular `N`; its inverse is built by six levels of recursive block inversion (each doubling block width, two 64×64 matmuls per level). A few ms on MPS.

**Correctness detail — Neumann product does not work.** An earlier version used `(I - N)(I + N^2)(I + N^4)...`, which is algebraically exact but off by up to 5e8 on real Qwen3.5 activations where the powers of `N` grow large before cancelling. A regression test (`tests/test_mps_kernels.py`) now pins an ill-conditioned case.

**Install skips if fla+triton present:** some builds will bind fla's kernel; that path is faster and this replacement is unneeded.

**PR #36 lands here:** adds a specialised Metal kernel for `_unit_lower_inverse` (fp32, 64×64, MPS only, with a fallback). 2.78× speedup on the inverse itself, 1.15× on full-model inference. Zero conceptual change to the install pattern.

## Where kernels sit in the current call tree

```
TorchServingModel (future) construction
  └─ picks device
       ├─ cuda → nothing to install (fla or causal_conv1d bind at import)
       ├─ mps  → cpu_kernels.install() — NO; mps_kernels.install()
       └─ cpu  → cpu_kernels.install_cpu()
```

Called today in `SystemOneEngine.__init__`:
```python
if device.startswith("mps"):
    from .mps_kernels import install

    install()
elif device == "cpu":
    from .cpu_kernels import install as install_cpu

    install_cpu()
```

Deferred imports — the modules aren't imported unless the device matches.

## Target location

User's framing: "**Kernels: installed to help improve engine inference.**"

Two layout options:

**(K1) Flat (keep today's shape):**
```
strands_decider/
  cpu_kernels.py
  mps_kernels.py
```

**(K2) Subpackage:**
```
strands_decider/
  kernels/
    __init__.py
    cpu.py
    mps.py
    cuda.py   # future
```

Both work; (K2) grows better if more backend-specific kernels appear (CUDA flash variants, MPS Metal kernels like PR #36's). (K1) is the minimum churn.

**Caller:** `TorchServingModel.load(..., device=...)` installs the appropriate kernel at construction, as today's `SystemOneEngine.__init__` does. Deferred imports preserved (don't import `cpu_kernels` when loading on CUDA).

**Backend scope:** kernels are a torch-backend internal. `MlxServingModel` and `LlamaCppServingModel` never call `install()`.

## Surprises / gotchas for the refactor

1. **Monkey-patching transformers is global and process-wide.** Two engines on different devices in the same process would both install; each `install()` is idempotent and device-dispatched, so this is safe. Preserve the dispatch pattern (don't replace unconditionally).
2. **`_installed` is a module-level flag.** Preserved by module reload; not reset by creating a new engine. Fine for serving.
3. **The install predicate (fla+triton for MPS) belongs with the kernel code.** Don't move it to the engine — the engine shouldn't need to know what alternatives bind at import.
4. **PR #36's Metal kernel adds a shape/dtype predicate (`fp32, 64×64, MPS`).** Preserve the fallback path — this is the pattern for every future accelerator addition.
5. **`__wrapped__` attribute** on the dispatcher lets tests and introspection see the original fn. Keep when the pattern repeats.
