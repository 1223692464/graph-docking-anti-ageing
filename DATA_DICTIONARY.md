# Data dictionary

## Modelling input

| Field / asset | Meaning |
| --- | --- |
| `pu_id` / prediction `uid` | Supplied stable compound identifier |
| `set` | P: curated positive; U: unlabelled background |
| `label` / `y_true` | Numeric modelling label, P = 1 and U = 0 |
| `smiles` / `smiles_canonical` | Supplied molecular structure strings |
| `graph_dataset.pt` | PyG graph objects, labels and compound identifiers |
| `split_idx_scaffold.pt` | Index lists into the provided graph/input order |
| `docking_input.xlsx` | Raw modelling docking scores and pose-count metadata |
| `*_norm_stats.csv` | Training-derived imputation and normalisation statistics |

## Candidate inputs and outputs

| Field family | Meaning / units |
| --- | --- |
| `screen_id`, `pu_id` | Compound ID used for candidate matching |
| `compound_name` | Supplied name where available; blank is missing metadata, not absence of activity |
| `smiles`, `canon_smiles` | Input and canonicalised structure strings |
| `{target}__dg_rank1` ... `rank5` | Provided ranked docking energies (kcal/mol) |
| `{target}__n_poses_found` | Recorded pose count |
| `energy_{target}` | QC reconstructed target energy (kcal/mol) |
| `aff_{target}` | Negative of energy before training normalisation |
| `succ_{target}` | Target-level success flag; 1 indicates available valid nonzero score(s) |
| `valid_pose_count_{target}` | Number of usable supplied rank-score columns, not total docking runs |
| `valid_target_count` | Number of targets with a usable score |
| `docking_coverage` | `valid_target_count / 20` |
| `logit_seed{seed}` | Pre-sigmoid output of the specified seed model |
| `score_seed{seed}` | Sigmoid of the corresponding logit |
| `pred_seed{seed}` | 1 if `logit_seed{seed} > 0.0`, otherwise 0 |
| `rank_seed{seed}` | Descending within-library logit rank for the specified seed |
| `logit_mean`, `logit_sd` | Mean and sample SD (`ddof=1`) of five logits |
| `score_mean`, `score_sd` | Mean and sample SD of five sigmoid outputs |
| `positive_vote` | Sum of five `pred_seed` indicators; range 0-5 |
| `rank_mean`, `rank_sd` | Mean and sample SD of five within-library ranks |
| `rank_final` | Recorded final-library ordering |
| `binding_target_count_le5` | Number of targets with supplied QC energy <= -5 kcal/mol |
| `binding_coverage_le5` | Binding-condition count / 20 |
| `passes_FDA_Q90` | Meets the full-precision FDA threshold only |
| `computational_high_confidence` | Meets all four final selection conditions |
| `qc_pass_cov12`, `qc_pass_cov15` | At least 12 or 15 usable targets (coverage >= 0.60 or 0.75) |

Additional physicochemical/fingerprint columns are retained from the supplied candidate table; they are not evidence of experimental activity. Source columns named `formula_calc` are retained unchanged and have not been independently repaired or validated. A zero, blank and missing value must not be interpreted interchangeably.

## FDA reference

`InChIKey14` is the connectivity-block matching key used by the supplied threshold script. It is not a full stereochemistry-sensitive InChIKey identity check. `drug_name`, reference-set membership, evidence/reference metadata, molecular structures and prediction fields are preserved in the FDA CSV files. Threshold matching lowercases/trims keys and keeps the highest-logit prediction when duplicate keys exist.

## Benchmark metrics

ROC-AUC is receiver operating characteristic area under the curve. Recorded `pr_auc` is scikit-learn **average precision**, not necessarily trapezoidal PR-curve area. P@K is the fraction of P labels in the top K saved scores. F1 is the harmonic mean of precision and recall. MCC is Matthews correlation coefficient. Balanced accuracy is the mean of sensitivity and specificity. Means/SDs over seeds use sample SD (`ddof=1`). Classification thresholds are selected by maximum validation F1, while screening votes use a fixed zero-logit threshold.
