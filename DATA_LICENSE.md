# Data provenance and redistribution

The `demo_data/`, `demos/`, and `assets/demos/` directories contain the small
verification examples distributed with this release. If the corresponding
author has the right to redistribute them, they are intended to follow the
GE-Sim 2.0 repository's CC BY-NC-SA 4.0 terms; they are not covered by any
upstream model license.

Before publishing a public repository, the corresponding author must verify and
record all of the following for the three episodes:

1. the original dataset/project name and version;
2. the source URL or accession identifier;
3. permission to redistribute the decoded videos, annotations, and parquet files;
4. the applicable data license and attribution text; and
5. any restrictions on commercial use, derivative works, or redistribution.

Until those checks are complete, treat the bundled files as **verification-only
release assets** and do not represent them as being covered by CC BY-NC-SA 4.0.
If the source terms do not allow redistribution, remove the data and
video assets from the public repository and provide an accession/download script
instead. Update this file with the verified terms before tagging a release.

The code records SHA-256 digests for the expected decoded frames in
`tests/golden_manifest.json`; these digests are for integrity checking, not a
substitute for data provenance or permission.
