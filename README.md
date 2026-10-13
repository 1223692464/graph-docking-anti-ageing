# Graph-Docking anti-ageing compound prioritisation

Computational reproducibility materials for *Development of a Graph-Docking Fusion model for anti-ageing compound prioritisation with phenotypic validation*.

This release contains the final QC-aware Docking-only, Graph-only and Graph-Docking Fusion implementations, modelling inputs, the fixed scaffold split, five-seed checkpoints, saved predictions, FDA-reference filtering evidence, and a 4,593-compound candidate-prediction database snapshot. The study website is https://ksl.xmnk.cn.

## Quick start

Use a local clone or download and extract GitHub's ZIP of this repository. Run commands from the repository root. Large CSV and PyTorch assets are losslessly gzip-compressed; the preparation command restores their original filenames and checks SHA-256 hashes.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install numpy pandas openpyxl
python scripts/prepare_data.py
python scripts/verify_release.py
python scripts/reproduce_selection.py
```

The quick checks do not require PyTorch, PyTorch Geometric or RDKit. They validate saved outputs, not a new training run. Expected results:

| Check | Expected value |
| --- | --- |
| Graph-input rows | 5,214 (P = 214; U = 5,000) |
| QC-matched Docking-only / Fusion rows | 5,212 (P = 212; U = 5,000) |
| Common test set | 78 (P = 54; U = 24) |
| Candidate-library rows | 4,593 |
| FDA AB reference records meeting coverage >= 0.75 | 20 of 21 |
| FDA_Q90, full precision | 24.62142791748047 |
| Candidates meeting logit_mean >= FDA_Q90 only | 941 |
| Candidates meeting all four final conditions | 7 |

See [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md) for the input-count discrepancy, split composition, exact feature dimensions and verification boundaries. The supplied data are not altered to force agreement with a manuscript count of 211 positives.

## Contents

| Directory / file | Contents |
| --- | --- |
| `scripts/` | Portable implementations and release-verification helpers |
| `data/processed/` | Structures, P/U labels, graph tensors, scaffold indices, docking scores and source metadata |
| `reports/FMH_screen_input_for_best_svm.csv.gz` | Actual 4,593-compound Fusion-screening input; filename is historical |
| `results/*multiseed_scaffold/` | Five seed checkpoints, train/validation/test predictions, histories, metrics and preprocessing statistics for each model |
| `results/three_model_test_diagnostics/` | Saved classification metrics using validation-F1 thresholds |
| `results/screening_qc_graph_docking_multiseed_v2/` | Full-library predictions, including seed-level logits |
| `results/final_candidate_selection_binding_le5/` | Complete predictions with binding-condition counts, 20-target energy matrix and column mapping |
| `results/fda_drug_library/` | FDA reference structures/metadata, inference input, predictions and threshold tables |
| `database/` | Excel candidate snapshot and the supplied named seven-candidate source table |
| `verification/` | Sample membership, recomputed metrics, selection evidence and release-check report |
| `SOURCE_MANIFEST.csv` / `.json` | Source-relative mapping and hashes of original and compressed assets |
| `RELEASE_CHECKSUMS.csv` | Checksums of files actually distributed on GitHub |

The full machine-readable candidate record is
`results/final_candidate_selection_binding_le5/candidate_predictions_with_binding_le5_metrics.csv.gz`.
The Excel snapshot contains 4,593 rows of structures, target scores, seed predictions and screening flags. It is not an export of all website metadata. Compound names are included only where present in the supplied seven-candidate table; absent names and herb-source annotations are not invented.

## Training and inference

Install the additional dependencies described in [ENVIRONMENT.md](ENVIRONMENT.md). Historical training-version pins were not recovered; `requirements.txt` is a dependency list, not a lockfile.

**Make a separate working copy of the prepared repository before retraining or running the original analysis scripts.** These scripts write to their configured `results/` directories and can overwrite existing outputs in that working copy. Preserve this release as a reference.

```bash
python scripts/train_docking_mlp_qc_all20_multiseed.py
python scripts/train_graph_only_multiseed_scaffold.py
python scripts/train_graph_docking_qc_all20_multiseed.py
python scripts/screen_fmh_candidates_qc_graph_docking_multiseed_v2.py
python scripts/screen_fda_ABC47_qc_graph_docking_v1.py
python scripts/calculate_fda_AB21_ABC47_thresholds_v1.py
python scripts/reproduce_selection.py
```

Use the provided graph tensors and scaffold-index file for direct reproduction. `build_graph_dataset.py` and `build_scaffold_split.py` are supplied for inspecting or rebuilding preprocessing; they are not needed for the quick start. The graph-builder also emits a legacy random split, which is not the split used by the reported final models.

The published classification metrics use a validation-selected F1 threshold, not the screening vote threshold. `verify_release.py` recomputes these model-level metrics and checks identifier-aligned membership.

## Provenance and reuse

The author-supplied positive collection originates from Geroprotectors.org; the U background was selected from ChemDiv. Candidate inputs are the author-supplied medicinal/edible library after structural QC. FDA metadata and supporting-reference fields are retained in the supplied reference CSVs.

Input/output assets and the retained core scripts are checksum-mapped. Portable scripts replace original absolute path constants with repository-relative resolution; they do not change model architectures, loss functions or sampling. New release helpers are separate from the historical implementations. Exploratory scripts and duplicate source copies are not distributed.

Third-party source terms continue to apply. No new blanket licence for third-party data, or unconfirmed historical software versions, is asserted by this release. Only load the included PyTorch/pickle files if you trust their provenance.
