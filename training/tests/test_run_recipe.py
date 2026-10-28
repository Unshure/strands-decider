"""training/run_recipe.sh keeps the configs it ran and never overwrites that record.

The runner is driven here with a stub recipe.sh that exits 0, no S3 and one GPU.
A RUN_DIR holds one run: after `SEED=1 ... train` a plain `eval` in the same RUN_DIR
must stop, or the edited copy that trained the checkpoint would be replaced with the
unedited config and stages.jsonl, hf_export and the S3 sync would carry the wrong file.
"""

import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "training", "run_recipe.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def repo(tmp_path):
    """A repo root as the runner checks it: recipe.sh, src/strands_decider/, the two configs."""
    (tmp_path / "src" / "strands_decider").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "training").mkdir()
    stub = tmp_path / "training" / "recipe.sh"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    for name in ("train.yaml", "train-parent.yaml"):
        shutil.copy(os.path.join(ROOT, "configs", name), tmp_path / "configs" / name)
    return tmp_path


def run(repo, *stages, **env):
    e = {
        **os.environ,
        "PY": "python",
        "S3_PREFIX": "none",
        "NGPU": "1",
        "RUN_DIR": str(repo / "run"),
        **env,
    }
    return subprocess.run(
        ["bash", RUNNER, *stages], cwd=repo, env=e, capture_output=True, text=True
    )


def seed_line(repo):
    return [
        ln
        for ln in (repo / "run" / "configs" / "train.yaml").read_text().splitlines()
        if ln.startswith("seed:")
    ]


def test_plain_run_keeps_unedited_copies_and_records_the_names(repo):
    assert run(repo, "train", "eval").returncode == 0
    for name in ("train.yaml", "train-parent.yaml"):
        assert (repo / "run" / "configs" / name).read_bytes() == (
            repo / "configs" / name
        ).read_bytes()
    rows = [json.loads(ln) for ln in (repo / "run" / "stages.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["train", "eval"]
    for r in rows:
        assert (r["parent_config"], r["train_config"], r["ckpt"]) == (
            "configs/train-parent.yaml",
            "configs/train.yaml",
            "checkpoints/hobson-2b-recipe",
        )


def test_plain_run_after_seed_run_is_refused(repo):
    assert run(repo, "train", SEED="1").returncode == 0
    assert seed_line(repo) == ["seed: 1  # SEED override from training/run_recipe.sh"]
    # The same SEED again is the same run and goes on.
    assert run(repo, "eval", SEED="1").returncode == 0
    plain = run(repo, "eval")
    assert plain.returncode != 0
    assert "use another RUN_DIR" in plain.stderr
    assert seed_line(repo) == ["seed: 1  # SEED override from training/run_recipe.sh"]
    rows = [json.loads(ln) for ln in (repo / "run" / "stages.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["train", "eval"]  # the refused stage wrote no line
    assert all(r["train_config"] == "configs/train.yaml" for r in rows)


def test_config_path_with_quote_is_rejected(repo):
    shutil.copy(repo / "configs" / "train.yaml", repo / "configs" / 'bad".yaml')
    r = run(repo, "eval", TRAIN_CONFIG='configs/bad".yaml')
    assert r.returncode == 2
    assert "[A-Za-z0-9._/-]" in r.stderr
    assert not (repo / "run" / "stages.jsonl").exists()
