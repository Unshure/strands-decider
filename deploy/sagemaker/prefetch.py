"""Download a checkpoint and the base model it loads into the image, at build time.

    python prefetch.py REPO_ID REVISION OUT_DIR

The checkpoint goes to OUT_DIR. The base model goes to the Hugging Face cache (HF_HOME) at
the revision that `StrandsDeciderModel.load` asks for, so the container then loads with
HF_HUB_OFFLINE=1 and needs no network access.
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from strands_decider.modeling import StrandsDeciderConfig, base_revision, config_path


def main(repo_id: str, revision: str, out_dir: str) -> None:
    path = snapshot_download(repo_id, revision=revision, local_dir=out_dir)
    config = StrandsDeciderConfig.from_json(config_path(path))
    base_rev = base_revision(path, config)
    print(f"checkpoint {repo_id}@{revision} -> {path}")
    print(f"base model {config.base_model}@{base_rev or 'main'}")
    snapshot_download(config.base_model, revision=base_rev)


if __name__ == "__main__":
    main(*sys.argv[1:4])
