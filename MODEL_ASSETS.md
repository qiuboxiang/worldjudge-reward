# External model and checkpoint metadata

The inference repository intentionally excludes the Qwen3-VL-4B-Instruct base
model and the WorldJudge checkpoint. They are too large for a normal GitHub
source repository and may have separate licenses or access conditions.

For every citable release, record the following information in the release notes
or a DOI-minting archive (Zenodo/Code Ocean):

| Asset | Required metadata |
| --- | --- |
| Base model | exact model identifier, provider URL, revision/commit, license, SHA-256 of files |
| WorldJudge checkpoint | checkpoint name/step, source URL or accession, revision, license, SHA-256 of every shard |
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
