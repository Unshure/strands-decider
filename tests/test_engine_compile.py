"""`EngineConfig.compile_mode` wraps the torso in `torch.compile` and keeps answers.

CPU only, tiny random torso and byte-level tokenizer. The production gate is CUDA-only
(a one-line note on other devices); here we bypass the gate directly so CI can run the
compiled path on CPU. Numerics are compared to the uncompiled engine on the same tiny
weights and must match to float tolerance.
"""
from __future__ import annotations

import pytest
import torch

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["<state> <question> <options> <answer> choice"],
        trainers.BpeTrainer(
            vocab_size=300, special_tokens=["<pad>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


def _tiny_model():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    tok = _tokenizer()
    torso = Qwen3Model(Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    ))
    cfg = StrandsDeciderConfig(
        base_model="tiny", head_type="pointer", pointer_dim=16,
        use_lora=False, torch_dtype="float32", max_length=512,
    )
    return StrandsDeciderModel(cfg, torso, tok)


def _requests():
    """Mix of single-question and multi-question requests, so both the batched path and
    the shared-prefix path are compiled through."""
    out = []
    for i in range(6):
        opts = [f"option{k}" + "x" * (k * 3 % 7) for k in range(2 + i % 3)]
        out.append(SystemOneRequest(
            state="s" * (5 + (i * 13) % 20),
            questions={"q": ChoiceQuestion(
                type="choice", instructions="pick " * (1 + i % 3),
                criteria={o: "" for o in opts},
            )},
        ))
    for i in range(4):
        out.append(SystemOneRequest(
            state="t" * (5 + i * 7),
            questions={
                f"n{j}": NoulQuestion(type="noul", instructions=f"ask {i} {j}")
                for j in range(3)
            },
        ))
    return out


def _dump(response):
    return {n: a.model_dump() for n, a in response.answers.items()}


def test_compile_mode_none_leaves_torso_eager():
    engine = SystemOneEngine(_tiny_model(), EngineConfig(device="cpu"))
    assert not hasattr(engine.model.torso, "_orig_mod")


def test_compile_mode_on_non_cuda_device_is_a_no_op(capsys):
    """The production gate: setting compile_mode on cpu/mps/mlx prints a note and skips."""
    engine = SystemOneEngine(
        _tiny_model(), EngineConfig(device="cpu", compile_mode="default"),
    )
    out = capsys.readouterr().out
    assert "compile_mode='default' ignored on device 'cpu'" in out
    assert not hasattr(engine.model.torso, "_orig_mod")


def test_compiled_torso_matches_eager(monkeypatch):
    """Bypass the CUDA gate so CPU CI can prove numerics equivalence. The compiled torso
    produces the same answers as the eager one on both the batched and shared-prefix paths.
    """
    model_a = _tiny_model()
    model_b = _tiny_model()  # seeded identically; same weights
    eager = SystemOneEngine(model_a, EngineConfig(device="cpu"))

    # Force the compile path on CPU by neutralising the device gate.
    def force_compile(self):
        mode = self.cfg.compile_mode
        if mode is None:
            return
        self.model.torso = torch.compile(self.model.torso, mode=mode)

    monkeypatch.setattr(SystemOneEngine, "_maybe_compile_torso", force_compile)
    compiled = SystemOneEngine(model_b, EngineConfig(device="cpu", compile_mode="default"))
    assert hasattr(compiled.model.torso, "_orig_mod")

    reqs = _requests()
    with torch.inference_mode():
        for r in reqs:
            a = _dump(eager.evaluate(r))
            b = _dump(compiled.evaluate(r))
            assert a.keys() == b.keys()
            for name in a:
                for field, va in a[name].items():
                    vb = b[name][field]
                    if isinstance(va, float):
                        assert va == pytest.approx(vb, abs=1e-4), (
                            f"{name}.{field}: eager {va} vs compiled {vb}"
                        )
                    elif isinstance(va, dict):
                        for k, vak in va.items():
                            assert vak == pytest.approx(vb[k], abs=1e-4)
                    else:
                        assert va == vb
