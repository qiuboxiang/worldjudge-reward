# NMI reproducibility checklist

This file is the release-side record for the code-management and reproducibility
items commonly requested by Nature Machine Intelligence (NMI). It complements,
but does not replace, the journal's submission forms or editorial review.

## Included in this repository

- [x] Public source code with a clear directory structure.
- [x] `README.md` with installation, model layout, run commands, and verification.
- [x] Pinned Python dependencies in `requirements.txt`.
- [x] Small test dataset and task annotations.
- [x] Dataset SHA-256 manifest and output parity checks.
- [x] Offline execution flags to prevent unrecorded model downloads.
- [x] OSI-approved software license and `CITATION.cff`.
- [x] Change history and a versioned release tag (`CHANGELOG.md`).
- [x] Continuous integration for source compilation and repository sanity checks.

## Must be completed before manuscript submission

- [ ] Verify the redistribution terms and provenance recorded in `DATA_LICENSE.md`.
- [ ] Record exact base-model/checkpoint identifiers and checksums using `MODEL_ASSETS.md`.
- [ ] Tag the exact manuscript release and archive it in Zenodo or Code Ocean for a DOI.
- [ ] Add the public repository URL and DOI to the manuscript's Code availability statement.
- [ ] Add the data accession/source and restrictions to the manuscript's Data availability statement.
- [ ] Make the exact review snapshot available to editors/referees if requested.

## Manuscript-ready statements (edit the bracketed fields)

**Code availability.** The inference code used in this study is available at
`https://github.com/qiuboxiang/worldjudge-reward` at release `[TAG]` and is
archived at `[DOI URL]`. The repository includes the source code, pinned
dependencies, verification data, and scripts needed to reproduce the released
inference outputs. The software is distributed under the MIT License. The base
model and WorldJudge checkpoint are separate release assets; their exact
identifiers, revisions, checksums, and access terms are listed at `[ARCHIVE URL]`.

**Data availability.** The three verification episodes are available in the
repository under the terms documented in `DATA_LICENSE.md` and are identified by
the SHA-256 manifest in `tests/golden_manifest.json`. The full training/evaluation
datasets are available from `[SOURCE/ACCESSION]` under `[TERMS]`; any restrictions
are described in the manuscript and repository metadata.

## Scope note

NMI policy requires transparent access to code, data, and materials needed to
evaluate the published claims. A GitHub repository is useful for collaboration,
but a DOI-minting archive and a pinned release are needed for a stable citation.
