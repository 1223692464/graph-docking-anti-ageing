# Environment and verification status

## Original training environment

The saved summaries report CUDA execution. Exact Python, CUDA, PyTorch, RDKit, PyTorch Geometric and other package versions were not recovered from the supplied files. Do not treat a current/latest version as the original version. `requirements.txt` lists imports required by the supplied scripts and intentionally does not fabricate version pins.

For training, install a mutually compatible PyTorch/PyTorch Geometric combination for the intended CPU or CUDA environment, RDKit, and the tabular dependencies. Graph layers are imported directly from `torch_geometric.nn.GINEConv`. The scripts attempt `torch.load(..., weights_only=False)` where available; only load trusted release artifacts.

```bash
python -m pip install -r requirements.txt
```

GPU-specific wheels or optional PyG extensions may require installation appropriate to the local PyTorch/CUDA build. This command is not a verified original-environment recreation. Deterministic numeric identity across hardware and versions is not guaranteed.

## Release checks performed

Saved-output verification was run on Windows with Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 and openpyxl 3.1.5. These are **verification-runtime versions**, not historical training versions.

- Source and stored-file hashes verified; compressed files restored byte-for-byte.
- Python source parsed without importing the training scripts.
- 45 prediction tables checked against input IDs and labels.
- Test ROC-AUC, average precision, P@20, P@50, validation-selected F1 thresholds, F1, MCC, balanced accuracy and confusion counts recomputed for 15 model/seed combinations.
- Candidate seed means, sample SDs, votes and 20-target binding counts checked.
- FDA AB21 threshold rejoined and recomputed; 941 threshold-passing candidates and 7 four-rule candidates verified.

Not executed: full retraining, checkpoint inference, RDKit graph rebuilding, or AutoDock-GPU docking. PyTorch and RDKit were not installed in the release-verification runtime.
