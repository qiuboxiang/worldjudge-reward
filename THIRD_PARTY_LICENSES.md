# Third-party license boundaries

The repository-level license follows the GE-Sim 2.0 repository: original code,
annotations, demo data, and rendered demo assets are under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

That project-level choice does **not** relicense material owned by other
projects. The following boundaries apply:

| Component | Source | License boundary |
| --- | --- | --- |
| Base model | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Apache-2.0 and upstream model-card terms; not relicensed here |
| WorldJudge checkpoint | [qiukingballball/worldjudge-ckpt-21500](https://huggingface.co/qiukingballball/worldjudge-ckpt-21500) | Use the license and terms stated by the checkpoint repository; not relicensed here |
| Python dependencies | `requirements.txt` | Each package retains its own license |
| GE-Sim 2.0-derived conventions | [AgibotTech/GE-Sim-V2](https://github.com/AgibotTech/GE-Sim-V2) | Follow the upstream repository's Apache-2.0/CC BY-NC-SA-4.0 split where applicable |

Before a public release, verify the checkpoint model card and the provenance and
redistribution rights of the bundled demo data. If a third-party term conflicts
with this repository's intended use, remove or replace that asset rather than
silently applying CC BY-NC-SA 4.0 to it.
