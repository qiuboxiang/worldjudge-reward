# External model and checkpoint metadata

The inference repository intentionally excludes the Qwen3-VL-4B-Instruct base
model and the WorldJudge checkpoint. They are too large for a normal GitHub
source repository and may have separate licenses or access conditions.

## WorldJudge checkpoint (released asset)

The checkpoint used by the released demo is:

- **Hugging Face repository:** [`qiukingballball/worldjudge-ckpt-21500`](https://huggingface.co/qiukingballball/worldjudge-ckpt-21500)
- **Expected local directory:** `/path/to/checkpoint/21500`
- **Expected role:** local `CHECKPOINT` passed to `run_all.sh`

Download it with:

```bash
python -m pip install --upgrade huggingface_hub
hf download qiukingballball/worldjudge-ckpt-21500 \
  --repo-type model \
  --local-dir /path/to/checkpoint/21500
```

Before a citable release, pin `--revision` to the exact commit shown by the
Hugging Face repository and record it below. The repository must contain either
`model.safetensors` or `model.safetensors.index.json` plus all referenced shard
files; the loader validates this structure before inference.

| Field | Release value |
| --- | --- |
| Hugging Face model ID | `qiukingballball/worldjudge-ckpt-21500` |
| Revision/commit | `[record immutable commit here]` |
| Checkpoint step | `21500` |
| License | `[copy the checkpoint repository license here]` |
| SHA-256 | `[record every shard checksum here]` |

## Base model

The released runtime expects
[`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
as the base model. The model card currently identifies it as Apache-2.0; retain
the upstream license and attribution when redistributing or citing the runtime.

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --repo-type model \
  --local-dir /path/to/Qwen3-VL-4B-Instruct
```

Pin the base model to an immutable Hugging Face commit for each reported result
and record that commit and all file checksums in the release archive.

For every citable release, record the following information in the release notes
or a DOI-minting archive (Zenodo/Code Ocean):

| Asset | Required metadata |
| --- | --- |
| Base model | `Qwen/Qwen3-VL-4B-Instruct`, provider URL, revision/commit, Apache-2.0, SHA-256 of files |
| WorldJudge checkpoint | `qiukingballball/worldjudge-ckpt-21500`, source URL, revision, license, SHA-256 of every shard |
| Runtime | Python version, OS, CUDA/driver, GPU model, PyTorch/Transformers versions |

The local directory layout expected by `run_all.sh` is:

```text
/path/to/Qwen3-VL-4B-Instruct/
/path/to/checkpoint/21500/
  config.json
  model.safetensors           # or model.safetensors.index.json + shards
```

The loader uses `local_files_only=True`; it will fail rather than silently fetch
an unrecorded model. Do not commit model weights, access tokens, or private
download URLs to Git.
