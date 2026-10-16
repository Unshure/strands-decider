"""Download a strands-decider model and the base model it loads, laid out for SageMaker.

    python fetch_model.py MODEL REVISION OUT_DIR
    python fetch_model.py --commit MODEL REVISION     # print the commit REVISION names

MODEL is a Hugging Face Hub id. The script writes:

    OUT_DIR/<model name>/  the model, at REVISION, with its config pointing at BASE_DIR
    OUT_DIR/base/          the base model, at the revision the model records

SageMaker copies OUT_DIR to /opt/ml/model (the stack's ModelDataSource), so the container loads
both without network access. The last line printed is the model's commit.

It needs only huggingface_hub. It repeats the base-revision rule of
strands_decider.modeling.base_revision so it can run without torch.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

BASE_DIR = "/opt/ml/model/base"
CONFIG_NAMES = ("strands_decider_config.json", "hobson_config.json")


def base_revision(checkpoint: str, config: dict) -> str | None:
    """The config's base revision, else the one provenance.json records for the same base."""
    if config.get("base_revision"):
        return config["base_revision"]
    prov = os.path.join(checkpoint, "provenance.json")
    if not os.path.exists(prov):
        return None
    with open(prov, encoding="utf-8") as fh:
        meta = json.load(fh)
    return (
        meta.get("base_model_revision") if meta.get("base_model") == config["base_model"] else None
    )


def main(model: str, revision: str, out_dir: str) -> None:
    commit = HfApi().model_info(model, revision=revision).sha
    checkpoint = os.path.join(out_dir, model.split("/")[-1])
    snapshot_download(model, revision=commit, local_dir=checkpoint)
    config_path = next(p for n in CONFIG_NAMES if os.path.exists(p := os.path.join(checkpoint, n)))
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    base, base_rev = config["base_model"], base_revision(checkpoint, config)
    print(f"model {model}@{commit}; base model {base}@{base_rev or 'main'}", file=sys.stderr)
    snapshot_download(base, revision=base_rev, local_dir=os.path.join(out_dir, "base"))

    # The loader reads the base model from BASE_DIR. With base_model no longer the Hub id in
    # provenance.json, base_revision() returns None, so nothing asks for a Hub revision offline.
    config["base_model"], config["base_revision"] = BASE_DIR, None
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    # snapshot_download keeps its metadata in .cache/; SageMaker does not need it.
    for sub in (checkpoint, os.path.join(out_dir, "base")):
        shutil.rmtree(os.path.join(sub, ".cache"), ignore_errors=True)
    print(commit)


if __name__ == "__main__":
    if sys.argv[1] == "--commit":
        print(HfApi().model_info(sys.argv[2], revision=sys.argv[3]).sha)
    else:
        main(*sys.argv[1:4])
