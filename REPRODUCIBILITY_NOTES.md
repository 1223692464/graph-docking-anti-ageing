# Reproducibility notes

## Input counts and split

The supplied `graph_input.csv` contains 5,214 rows: P214 and U5000. The fixed scaffold split contains:

| Subset | Graph-only total | P | U | QC Docking-only / Fusion total | QC P | QC U |
| --- | --- | --- | --- | --- | --- | --- |
| Training | 5,019 | 105 | 4,914 | 5,019 | 105 | 4,914 |
| Validation | 117 | 55 | 62 | 115 | 53 | 62 |
| Test | 78 | 54 | 24 | 78 | 54 | 24 |

Docking matching excludes validation IDs P_000061 and P_000108 (see `verification/QC_excluded_validation_compounds.csv`). The 15 final test files share the same 78 IDs; training IDs also match. Five seeds vary training randomness on this single split.

The manuscript's previously stated 211-positive count is not reconciled by these supplied files. The repository reports actual source counts without removing rows retrospectively. Source selection, final deduplication and manuscript accounting should be reconciled by the authors before a corresponding data-availability claim is finalised.

The test positive fraction is 54/78; it differs substantially from the full modelling set. PR-AUC and P@K refer to this supplied test composition. Unlabelled rows are encoded as 0 for prediction/evaluation; that does not establish experimentally verified inactivity.

## Implemented representation

- PyG GINE graph branch: node features 48, edge features 13, graph embedding 128.
- QC docking input: 20 affinity-related values, 20 target-success flags, and 1 coverage value; total 41.
- Docking embedding: 64. Concatenated representation: 192.
- Fusion is concatenation followed by a classification head, not gated attention.
- QC scores average the available nonzero, nonmissing rank-score columns discovered by the supplied code. The code, not a legacy filename containing `top3mean`, defines the final feature reconstruction.
- Missing-value imputation and normalisation use training-subset statistics; inference reuses the saved seed-specific statistics. Docking-only and Fusion preprocessing details are not identical and are preserved rather than harmonised retrospectively.

Internal target order:

`1A9U, 2YXJ, 3O96, 4CFE, 4HHY, 4JPS, 4JSX, 4KIK, 4LXZ, 4OYS, 4WXX, 4XJT, 4ZZJ, 6MXE, 6OOK, 6SP1, 6VGL, 7PZC, 8DSE, 9KTK`.

The internal alias `6OOK` maps to PDB ID `6O0K` in display/output mapping. Changing checkpoint feature order or internal names would invalidate direct reproduction.

## Filtering and verification

FDA_Q90 uses the FDA AB21 reference, coverage >= 0.75, and pandas' linear quantile interpolation. The reference contains 21 records, of which 20 meet the specified rule. FDA ABC47 predictions provide the matching predictions but are not themselves the threshold-defining reference subset.

Screening votes use strict `logit > 0.0`; sample SD is computed with `ddof=1`. Binding filtering uses the 20 raw energy values, not imputed z-scores. Missing target energies cannot satisfy the 20-of-20 condition. The four-rule selection is a postprocessing operation and is not a probabilistic calibration claim.

`verify_release.py` checks saved-output arithmetic; it does not demonstrate independent regeneration by the neural-network code. Model-level classification metrics are independently recomputed from the provided validation/test predictions. Exploratory error-transition scripts and their outputs are not distributed.

## Publication scope

This release includes modelling/screening data and software. Experimental source data and a complete docking-generation workflow are outside its scope. Missing historical environment pins and unresolved sample accounting are stated explicitly. Inclusion of these files must not be described as complete reproduction of all manuscript results.
