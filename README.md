# WorldJudge reward inference

This repository is the reproducible, inference-only release for the WorldJudge
reward/progress head used with three GE-Sim V2 demonstration episodes:

| Task | Instruction | Frames |
| --- | --- | ---: |
| 5638 | Pour water 8 | 1,063 |
| 6005 | Grasp and release objects | 153 |
| 5645 | Clean mirror stains | 411 |

The runtime reads the bundled LeRobot-format examples, applies the released
WorldJudge checkpoint on a local Qwen3-VL-4B-Instruct base model, and evaluates
each episode in independent 16-frame chunks. It does not require a checkout of
GE-Sim, Robometer, or the unified-dataset repositories.

The base model and WorldJudge checkpoint are **external release assets**. They
are not stored in Git because each is about 8.3 GB. The released checkpoint is
hosted at [qiukingballball/worldjudge-ckpt-21500 on Hugging Face](https://huggingface.co/qiukingballball/worldjudge-ckpt-21500).
See [`MODEL_ASSETS.md`](MODEL_ASSETS.md) for the exact download procedure,
directory layout, and information that must be recorded for a citable release.

## Repository layout

```text
worldjudge_runtime/       model, preprocessing, and LeRobot reader
infer.py                  inference entry point
run_all.sh                offline end-to-end runner
demo_data/                small test episodes used by the dataset check
demos/                    one JSONL annotation per demo task
assets/demos/             rendered MP4 walkthroughs
tests/                    dataset and numerical-parity checks
```

## Environment

Use Python 3.10 and the pinned dependencies:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

The inference code is designed for a local/offline model cache. It needs a
CUDA-capable GPU for practical Qwen3-VL inference; CPU execution is supported by
the code but is not expected to be fast.

## Run the released demos

Set `BASE_MODEL` to the local Qwen3-VL-4B-Instruct directory and `CHECKPOINT` to
the local copy of the `qiukingballball/worldjudge-ckpt-21500` checkpoint:

```bash
BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct \
CHECKPOINT=/path/to/checkpoint/21500 \
VENV=$PWD/.venv \
bash run_all.sh
```

To download the checkpoint with the Hugging Face CLI, install
`huggingface_hub` and use the model repository ID (not a browser cache path):

```bash
python -m pip install --upgrade huggingface_hub
hf download qiukingballball/worldjudge-ckpt-21500 \
  --repo-type model \
  --local-dir /path/to/checkpoint/21500
```

For a manuscript or archival release, resolve the Hugging Face repository to a
specific commit and record that revision plus the SHA-256 checksum of every
downloaded shard in [`MODEL_ASSETS.md`](MODEL_ASSETS.md). Do not rely on the
mutable `main` branch when reporting a result.

The runner enables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so that the
reported result cannot silently depend on an unrecorded network download.
Results are written to `outputs/ckpt_21500/` as `summary.json`, NumPy arrays,
and PNG curves.

## Verification

The dataset check decodes the three bundled episodes and verifies their expected
shape and SHA-256 digest:

```bash
.venv/bin/python -I tests/check_dataset.py
```

To compare a candidate run with a released run, compare the progress arrays,
success probabilities, and both binary timelines:

```bash
.venv/bin/python -I tests/compare_outputs.py \
  outputs/ckpt_21500 outputs/candidate
```

`verify_outputs.py` also checks the expected episode lengths and 16-frame chunk
counts. The `tests/golden_manifest.json` file is the machine-readable record of
the dataset and output digests used for release QA.

## Demo videos

- [Task 5638 — pour water](assets/demos/task_5638.mp4)
- [Task 6005 — grasp and release objects](assets/demos/task_6005.mp4)
- [Task 5645 — clean mirror stains](assets/demos/task_5645.mp4)

## Reproducibility and NMI reporting

This repository is organized to satisfy the reproducibility expectations used
by Nature Machine Intelligence (NMI): source code, a test dataset, installation
and run instructions, pinned dependencies, deterministic offline settings, and
automated integrity/parity checks are all versioned together. The NMI-oriented
release checklist and manuscript-ready statements are in
[`NMI_REPRODUCIBILITY.md`](NMI_REPRODUCIBILITY.md).

For publication, archive the exact tagged Git commit and the external model and
checkpoint metadata in a DOI-minting repository such as Zenodo or Code Ocean,
then cite that DOI in the manuscript. A GitHub URL alone is not a permanent
version identifier.

## Data and model access

The three demo episodes are included only as the small verification set for this
release. Their provenance and redistribution terms must be kept with the release
and are documented in [`DATA_LICENSE.md`](DATA_LICENSE.md). Do not add private,
identifying, or unauthorized data to this repository.

The model/checkpoint are not included. The exact source, revision, SHA-256
checksums, hardware/software environment, and any access restrictions should be
recorded before making a public release; use [`MODEL_ASSETS.md`](MODEL_ASSETS.md)
as the metadata template.

## Citation and license

Please cite the software using [`CITATION.cff`](CITATION.cff). The source code is
released under the MIT License; see [`LICENSE`](LICENSE). The demo data are not
automatically covered by the software license; see [`DATA_LICENSE.md`](DATA_LICENSE.md).
