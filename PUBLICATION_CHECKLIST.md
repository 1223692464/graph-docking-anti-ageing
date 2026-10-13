# Publication checklist

## Included and checked

- [x] Final three-model scripts and core preprocessing/inference scripts with portable paths.
- [x] Structures, P/U labels, provided graph tensors and fixed scaffold indices.
- [x] Modelling docking scores, QC tables and preprocessing statistics.
- [x] Five-seed checkpoints, histories and train/validation/test predictions.
- [x] 4,593-compound screening input and full prediction/filtering record.
- [x] FDA reference sets, input, predictions and threshold reconstruction.
- [x] Named seven-candidate table and 4,593-row Excel prediction snapshot.
- [x] Source manifest, hashes, saved-output checks and run instructions.

## Author follow-up before claiming full manuscript reproducibility

- [ ] Reconcile manuscript P211 with supplied P214 graph input / P212 QC input.
- [ ] Recover the exact training environment, or validate and archive a reproducible replacement environment.
- [ ] Execute and document a clean training/inference reproduction in that environment.
- [ ] Supply HUVEC and worm source data with group, replicate and analysis metadata.
- [ ] Supply docking-generation inputs/settings if making a from-scratch docking reproducibility claim.
- [ ] Confirm redistribution terms for author-supplied ChemDiv-derived and other third-party records.
- [ ] Ensure manuscript model architecture, threshold rules and dataset accounting match the supplied final code.
- [ ] Update the manuscript's exact `Data and Software Availability` section to the actual repository URL and release scope, without an unsupported all-data claim.

Experimental data were intentionally not included in this computational release. This checklist does not postpone or waive a journal's data-sharing requirements.
