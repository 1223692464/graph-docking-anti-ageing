#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docking-only MLP with docking quality control (QC), multi-seed scaffold evaluation.

Purpose
-------
This script is an architecture-matched docking-only baseline for the Graph-Docking
fusion model, but uses QC-aware docking features:
  1) treat 0 docking values as failed/missing poses, not true binding energies;
  2) compute target-level success flags;
  3) impute missing energies using train-set target medians;
  4) standardize docking affinity features using train-set statistics;
  5) add compound-level docking coverage as an auxiliary QC feature;
  6) evaluate with the same scaffold split, seeds, and metrics as the fusion model.

Default output:
  /home/zyh/aging_proj/results/docking_mlp_qc_all20_multiseed_scaffold

Run:
  cd /home/zyh/aging_proj/scripts
  python3 train_docking_mlp_qc_all20_multiseed.py
"""
from repo_paths import project_path

import os
import json
import math
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader

# =========================
# 0. Paths and settings
# =========================
DATA_DIR = project_path('data/processed')
SCRIPTS_DIR = project_path('scripts')
RESULTS_DIR = project_path('results')

GRAPH_INPUT_PATH = os.path.join(DATA_DIR, "graph_input.csv")
RAW_DOCKING_XLSX = os.path.join(DATA_DIR, "docking_input.xlsx")
RAW_DOCKING_CSV_FALLBACK = os.path.join(DATA_DIR, "docking_features_20_top3mean.csv")

SPLIT_MODE = "scaffold"
if SPLIT_MODE == "random":
    SPLIT_CANDIDATES = [
        os.path.join(DATA_DIR, "split_idx.pt"),
        os.path.join(SCRIPTS_DIR, "split_idx.pt"),
    ]
    OUT_DIR = os.path.join(RESULTS_DIR, "docking_mlp_qc_all20_multiseed_random")
elif SPLIT_MODE == "scaffold":
    SPLIT_CANDIDATES = [
        os.path.join(DATA_DIR, "split_idx_scaffold.pt"),
        os.path.join(SCRIPTS_DIR, "split_idx_scaffold.pt"),
    ]
    OUT_DIR = os.path.join(RESULTS_DIR, "docking_mlp_qc_all20_multiseed_scaffold")
else:
    raise ValueError("SPLIT_MODE must be 'random' or 'scaffold'")

os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 412, 719, 666, 888]
MAX_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
HIDDEN_DIM = 64

# same 20-target order used in the previous fusion script
DEFAULT_TARGETS = [
    "1A9U", "2YXJ", "3O96", "4CFE", "4HHY",
    "4JPS", "4JSX", "4KIK", "4LXZ", "4OYS",
    "4WXX", "4XJT", "4ZZJ", "6MXE", "6OOK",
    "6SP1", "6VGL", "7PZC", "8DSE", "9KTK"
]

# QC policy
ZERO_AS_MISSING = True
IMPUTE_STRATEGY = "train_median"  # current option: train_median
USE_AFFINITY_DIRECTION = True      # convert energy to affinity by -energy, so larger = stronger
ADD_TARGET_SUCCESS_FLAGS = True
ADD_COVERAGE_FEATURE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Utilities
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if s == "":
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def precision_at_k(y_true, y_score, k):
    if len(y_true) == 0:
        return np.nan
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[order]))


def eval_metrics(y_true, y_score):
    metrics = {}
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    metrics["p_at_20"] = precision_at_k(y_true, y_score, 20)
    metrics["p_at_50"] = precision_at_k(y_true, y_score, 50)
    metrics["p_at_100"] = precision_at_k(y_true, y_score, 100)
    return metrics


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return "{:.4f}".format(float(x))


def format_metrics(m):
    return (
        "ROC-AUC={} | PR-AUC={} | P@20={} | P@50={} | P@100={}".format(
            fmt(m["roc_auc"]), fmt(m["pr_auc"]), fmt(m["p_at_20"]), fmt(m["p_at_50"]), fmt(m["p_at_100"])
        )
    )

# =========================
# 2. Build QC docking table
# =========================
def read_docking_source():
    """Prefer raw docking_input.xlsx. Fall back to 20-dimensional top3mean csv."""
    if os.path.exists(RAW_DOCKING_XLSX):
        print("[INFO] using raw docking xlsx: {}".format(RAW_DOCKING_XLSX))
        return pd.read_excel(RAW_DOCKING_XLSX), "raw_xlsx"
    if os.path.exists(RAW_DOCKING_CSV_FALLBACK):
        print("[WARN] raw docking_input.xlsx not found; using fallback csv: {}".format(RAW_DOCKING_CSV_FALLBACK))
        return pd.read_csv(RAW_DOCKING_CSV_FALLBACK), "fallback_csv"
    raise FileNotFoundError("No docking source found: {} or {}".format(RAW_DOCKING_XLSX, RAW_DOCKING_CSV_FALLBACK))


def infer_targets(df, source_type):
    if source_type == "raw_xlsx":
        raw_targets = sorted({c.split("__")[0] for c in df.columns if "__" in c})
        # keep the same target order as previous fusion if possible
        targets = [t for t in DEFAULT_TARGETS if t in raw_targets]
        extra = [t for t in raw_targets if t not in targets]
        targets.extend(extra)
    else:
        meta = {"ligand_id_pu", "pu_id", "set", "label", "SMILES", "smiles", "canon_smiles"}
        raw_targets = [c for c in df.columns if c not in meta]
        targets = [t for t in DEFAULT_TARGETS if t in raw_targets]
        extra = [t for t in raw_targets if t not in targets]
        targets.extend(extra)
    return targets


def build_qc_docking_features(df, source_type):
    if "SMILES" in df.columns:
        smiles_col = "SMILES"
    elif "smiles" in df.columns:
        smiles_col = "smiles"
    else:
        raise ValueError("Docking source has no SMILES/smiles column")

    targets = infer_targets(df, source_type)
    if len(targets) == 0:
        raise ValueError("No docking targets inferred from input columns")

    out = pd.DataFrame()
    for c in ["ligand_id_pu", "pu_id", "set", "label"]:
        if c in df.columns:
            out[c] = df[c]
    out["SMILES"] = df[smiles_col]

    qc_rows = []
    score_cols = []
    success_cols = []

    for t in targets:
        score_col = "{}__energy_qc".format(t)
        success_col = "{}__success".format(t)
        pose_count_col = "{}__valid_pose_count".format(t)

        if source_type == "raw_xlsx":
            rank_cols = ["{}__dg_rank{}".format(t, i) for i in range(1, 6) if "{}__dg_rank{}".format(t, i) in df.columns]
            top3_cols = ["{}__dg_rank{}".format(t, i) for i in range(1, 4) if "{}__dg_rank{}".format(t, i) in df.columns]

            if len(top3_cols) > 0:
                vals_top3 = df[top3_cols].apply(pd.to_numeric, errors="coerce")
                vals_top3 = vals_top3.mask(vals_top3 == 0) if ZERO_AS_MISSING else vals_top3
                energy = vals_top3.mean(axis=1, skipna=True)
            elif "{}__mean_top3".format(t) in df.columns:
                energy = pd.to_numeric(df["{}__mean_top3".format(t)], errors="coerce")
                energy = energy.mask(energy == 0) if ZERO_AS_MISSING else energy
            elif "{}__mean_top5".format(t) in df.columns:
                energy = pd.to_numeric(df["{}__mean_top5".format(t)], errors="coerce")
                energy = energy.mask(energy == 0) if ZERO_AS_MISSING else energy
            else:
                continue

            if len(rank_cols) > 0:
                vals_all = df[rank_cols].apply(pd.to_numeric, errors="coerce")
                vals_all = vals_all.mask(vals_all == 0) if ZERO_AS_MISSING else vals_all
                valid_pose_count = vals_all.notna().sum(axis=1).astype(float)
            else:
                valid_pose_count = energy.notna().astype(float)
        else:
            if t not in df.columns:
                continue
            energy = pd.to_numeric(df[t], errors="coerce")
            energy = energy.mask(energy == 0) if ZERO_AS_MISSING else energy
            valid_pose_count = energy.notna().astype(float)

        out[score_col] = energy
        out[success_col] = energy.notna().astype(float)
        out[pose_count_col] = valid_pose_count
        score_cols.append(score_col)
        success_cols.append(success_col)

        qc_rows.append({
            "target": t,
            "score_col": score_col,
            "success_col": success_col,
            "raw_valid_rate_all": float(out[success_col].mean()),
            "raw_missing_rate_all": float(1.0 - out[success_col].mean()),
        })

    out["valid_target_count"] = out[success_cols].sum(axis=1)
    out["docking_coverage"] = out["valid_target_count"] / max(len(success_cols), 1)
    out["canon_smiles"] = out["SMILES"].apply(canonicalize_smiles)

    qc_info = pd.DataFrame(qc_rows)
    return out, targets, score_cols, success_cols, qc_info

# =========================
# 3. Prepare matched graph/split/docking table
# =========================
if not os.path.exists(GRAPH_INPUT_PATH):
    raise FileNotFoundError("graph_input.csv not found: {}".format(GRAPH_INPUT_PATH))

split_path = find_first_existing(SPLIT_CANDIDATES)
if split_path is None:
    raise FileNotFoundError("split file not found in {}".format(SPLIT_CANDIDATES))

print("[INFO] device = {}".format(DEVICE))
print("[INFO] graph_input = {}".format(GRAPH_INPUT_PATH))
print("[INFO] split = {}".format(split_path))

graph_df = pd.read_csv(GRAPH_INPUT_PATH)
split_dict = safe_torch_load(split_path)

graph_df["split"] = None
for split_name in ["train", "val", "test"]:
    graph_df.loc[split_dict[split_name], "split"] = split_name

graph_df["canon_smiles"] = graph_df["smiles"].apply(canonicalize_smiles)
graph_df = graph_df.dropna(subset=["canon_smiles"]).copy()
graph_df = graph_df.drop_duplicates(subset=["canon_smiles"], keep="first").copy()

dock_source_df, source_type = read_docking_source()
qc_df, targets, score_cols, success_cols, qc_info = build_qc_docking_features(dock_source_df, source_type)
qc_df = qc_df.dropna(subset=["canon_smiles"]).copy()
qc_df = qc_df.drop_duplicates(subset=["canon_smiles"], keep="first").copy()

merge_cols = ["canon_smiles", "pu_id", "set", "label", "split", "smiles"]
graph_split_df = graph_df[merge_cols].copy()

matched_df = qc_df.merge(
    graph_split_df,
    on="canon_smiles",
    how="inner",
    suffixes=("_dock", "_graph")
)

# unify labels and IDs from graph_input
if "label_graph" in matched_df.columns:
    matched_df["label"] = matched_df["label_graph"].astype(int)
else:
    matched_df["label"] = matched_df["label"].astype(int)

if "set_graph" in matched_df.columns:
    matched_df["set"] = matched_df["set_graph"]
if "pu_id_graph" in matched_df.columns:
    matched_df["pu_id"] = matched_df["pu_id_graph"]

print("[INFO] source_type = {}".format(source_type))
print("[INFO] targets = {}".format(targets))
print("[INFO] n_targets = {}".format(len(score_cols)))
print("[INFO] matched samples = {}".format(len(matched_df)))
print("[INFO] train/val/test = {}/{}/{}".format(
    int((matched_df["split"] == "train").sum()),
    int((matched_df["split"] == "val").sum()),
    int((matched_df["split"] == "test").sum())
))

# Save raw QC summaries before imputation
qc_dir = os.path.join(OUT_DIR, "qc_tables")
os.makedirs(qc_dir, exist_ok=True)
qc_info.to_csv(os.path.join(qc_dir, "target_qc_valid_rate_all.csv"), index=False, encoding="utf-8-sig")

coverage_rows = []
for split_name in ["train", "val", "test", "all"]:
    sub = matched_df if split_name == "all" else matched_df[matched_df["split"] == split_name]
    row = {"split": split_name, "n": int(len(sub))}
    row["coverage_mean"] = float(sub["docking_coverage"].mean()) if len(sub) else np.nan
    row["coverage_median"] = float(sub["docking_coverage"].median()) if len(sub) else np.nan
    for thr in [0, 5, 8, 10, 12, 15, 18, 20]:
        row["n_valid_ge_{}".format(thr)] = int((sub["valid_target_count"] >= thr).sum())
    coverage_rows.append(row)
coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(os.path.join(qc_dir, "compound_coverage_summary.csv"), index=False, encoding="utf-8-sig")

# =========================
# 4. Missing-aware feature reconstruction
# =========================
train_mask = matched_df["split"] == "train"
train_df = matched_df[train_mask].copy()
val_df = matched_df[matched_df["split"] == "val"].copy()
test_df = matched_df[matched_df["split"] == "test"].copy()

if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    raise ValueError("Empty train/val/test after matching. Please check split and docking input.")

# impute missing energies per target using train median; then convert to affinity if enabled
feature_df = matched_df[["canon_smiles", "SMILES", "pu_id", "set", "label", "split", "valid_target_count", "docking_coverage"]].copy()

norm_rows = []
energy_feature_cols = []

for score_col in score_cols:
    target = score_col.replace("__energy_qc", "")
    raw_train = pd.to_numeric(matched_df.loc[train_mask, score_col], errors="coerce")
    if IMPUTE_STRATEGY == "train_median":
        impute_value = float(raw_train.median()) if raw_train.notna().any() else 0.0
    else:
        raise ValueError("Unsupported IMPUTE_STRATEGY: {}".format(IMPUTE_STRATEGY))

    x_energy = pd.to_numeric(matched_df[score_col], errors="coerce").fillna(impute_value).astype(float)
    x_value = -x_energy if USE_AFFINITY_DIRECTION else x_energy

    train_values = x_value.loc[train_mask].values.astype(float)
    mu = float(np.mean(train_values))
    sd = float(np.std(train_values))
    if sd == 0 or np.isnan(sd):
        sd = 1.0

    z_col = "{}__affinity_z".format(target) if USE_AFFINITY_DIRECTION else "{}__energy_z".format(target)
    feature_df[z_col] = (x_value - mu) / sd
    energy_feature_cols.append(z_col)

    norm_rows.append({
        "target": target,
        "raw_score_col": score_col,
        "impute_strategy": IMPUTE_STRATEGY,
        "impute_energy_value_train": impute_value,
        "feature_direction": "affinity=-energy" if USE_AFFINITY_DIRECTION else "energy",
        "train_feature_mean_before_z": mu,
        "train_feature_std_before_z": sd,
        "train_valid_rate": float(matched_df.loc[train_mask, target + "__success"].mean()) if target + "__success" in matched_df.columns else np.nan,
        "all_valid_rate": float(matched_df[target + "__success"].mean()) if target + "__success" in matched_df.columns else np.nan,
    })

flag_feature_cols = []
if ADD_TARGET_SUCCESS_FLAGS:
    for success_col in success_cols:
        target = success_col.replace("__success", "")
        flag_col = "{}__success_flag".format(target)
        feature_df[flag_col] = matched_df[success_col].astype(float)
        flag_feature_cols.append(flag_col)

extra_feature_cols = []
if ADD_COVERAGE_FEATURE:
    feature_df["docking_coverage_feature"] = matched_df["docking_coverage"].astype(float)
    extra_feature_cols.append("docking_coverage_feature")

feature_cols = energy_feature_cols + flag_feature_cols + extra_feature_cols

# ensure numeric
for c in feature_cols:
    feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce").fillna(0.0).astype(float)

feature_df.to_csv(os.path.join(qc_dir, "qc_matched_model_input.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(norm_rows).to_csv(os.path.join(qc_dir, "qc_impute_norm_stats.csv"), index=False, encoding="utf-8-sig")

print("[INFO] n_features = {}".format(len(feature_cols)))
print("[INFO] energy features = {}".format(len(energy_feature_cols)))
print("[INFO] success flag features = {}".format(len(flag_feature_cols)))
print("[INFO] extra features = {}".format(extra_feature_cols))
print("[OK] saved QC tables to: {}".format(qc_dir))

# split feature_df in original order
train_model_df = feature_df[feature_df["split"] == "train"].copy().reset_index(drop=True)
val_model_df = feature_df[feature_df["split"] == "val"].copy().reset_index(drop=True)
test_model_df = feature_df[feature_df["split"] == "test"].copy().reset_index(drop=True)

# =========================
# 5. Dataset and model
# =========================
class TabularDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.reset_index(drop=True).copy()
        self.x = self.df[feature_cols].values.astype(np.float32)
        self.y = self.df["label"].astype(int).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float),
            torch.tensor(self.y[idx], dtype=torch.float),
            int(idx)
        )


class DockingOnlyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb).view(-1)
        return logits, emb


def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(loader, model, ref_df, split_name="val", save_pred=False, seed_dir=None):
    model.eval()
    all_y = []
    all_score = []
    all_idx = []
    for x, y, idx in loader:
        x = x.to(DEVICE)
        logits, _ = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        all_y.append(y.numpy())
        all_score.append(prob)
        all_idx.extend(idx.numpy().tolist())

    y_true = np.concatenate(all_y).astype(int)
    y_score = np.concatenate(all_score)
    metrics = eval_metrics(y_true, y_score)

    pred_df = ref_df.iloc[all_idx][["pu_id", "set", "label", "SMILES", "canon_smiles", "valid_target_count", "docking_coverage"]].copy()
    pred_df["score"] = y_score
    pred_df = pred_df.sort_values("score", ascending=False).reset_index(drop=True)

    if save_pred and seed_dir is not None:
        pred_path = os.path.join(seed_dir, "{}_predictions.csv".format(split_name))
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return metrics, pred_df

# =========================
# 6. Multi-seed training
# =========================
summary_rows = []

for seed in SEEDS:
    print("\n" + "=" * 80)
    print("[RUN] SEED = {}".format(seed))
    set_seed(seed)

    seed_dir = os.path.join(OUT_DIR, "seed_{}".format(seed))
    os.makedirs(seed_dir, exist_ok=True)

    # deepcopy to avoid any accidental mutation
    train_dataset = TabularDataset(copy.deepcopy(train_model_df), feature_cols)
    val_dataset = TabularDataset(copy.deepcopy(val_model_df), feature_cols)
    test_dataset = TabularDataset(copy.deepcopy(test_model_df), feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = DockingOnlyMLP(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)

    y_train = train_model_df["label"].astype(int).values
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float, device=DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_pr = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion)
        train_metrics, _ = evaluate(train_loader, model, train_model_df, split_name="train", save_pred=False)
        val_metrics, _ = evaluate(val_loader, model, val_model_df, split_name="val", save_pred=False)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "train_p_at_20": train_metrics["p_at_20"],
            "train_p_at_50": train_metrics["p_at_50"],
            "train_p_at_100": train_metrics["p_at_100"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_p_at_20": val_metrics["p_at_20"],
            "val_p_at_50": val_metrics["p_at_50"],
            "val_p_at_100": val_metrics["p_at_100"],
        })

        print(
            "[Seed {}][Epoch {:03d}] loss={:.4f} | train PR-AUC={:.4f} | val PR-AUC={:.4f} | val P@50={:.4f}".format(
                seed, epoch, train_loss, train_metrics["pr_auc"], val_metrics["pr_auc"], val_metrics["p_at_50"]
            )
        )

        if val_metrics["pr_auc"] > best_val_pr:
            best_val_pr = val_metrics["pr_auc"]
            best_epoch = epoch
            bad_epochs = 0
            ckpt = {
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_pr_auc": best_val_pr,
                "feature_cols": feature_cols,
                "targets": targets,
                "qc_policy": {
                    "zero_as_missing": ZERO_AS_MISSING,
                    "impute_strategy": IMPUTE_STRATEGY,
                    "use_affinity_direction": USE_AFFINITY_DIRECTION,
                    "add_target_success_flags": ADD_TARGET_SUCCESS_FLAGS,
                    "add_coverage_feature": ADD_COVERAGE_FEATURE,
                }
            }
            torch.save(ckpt, os.path.join(seed_dir, "best_model.pt"))
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("[INFO] early stopping at epoch {}, best epoch = {}".format(epoch, best_epoch))
            break

    best_ckpt = safe_torch_load(os.path.join(seed_dir, "best_model.pt"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    train_metrics, train_pred = evaluate(train_loader, model, train_model_df, split_name="train", save_pred=True, seed_dir=seed_dir)
    val_metrics, val_pred = evaluate(val_loader, model, val_model_df, split_name="val", save_pred=True, seed_dir=seed_dir)
    test_metrics, test_pred = evaluate(test_loader, model, test_model_df, split_name="test", save_pred=True, seed_dir=seed_dir)

    print("[FINAL][SEED {}] TRAIN : {}".format(seed, format_metrics(train_metrics)))
    print("[FINAL][SEED {}] VAL   : {}".format(seed, format_metrics(val_metrics)))
    print("[FINAL][SEED {}] TEST  : {}".format(seed, format_metrics(test_metrics)))

    pd.DataFrame(history).to_csv(os.path.join(seed_dir, "training_history.csv"), index=False, encoding="utf-8-sig")

    for k in [20, 50, 100]:
        test_pred.head(k).to_csv(os.path.join(seed_dir, "test_top{}.csv".format(k)), index=False, encoding="utf-8-sig")

    metrics_dict = {
        "seed": seed,
        "split_mode": SPLIT_MODE,
        "source_type": source_type,
        "graph_input": GRAPH_INPUT_PATH,
        "raw_docking_xlsx": RAW_DOCKING_XLSX,
        "fallback_docking_csv": RAW_DOCKING_CSV_FALLBACK,
        "split_file": split_path,
        "matched_samples": int(len(feature_df)),
        "train_samples": int(len(train_model_df)),
        "val_samples": int(len(val_model_df)),
        "test_samples": int(len(test_model_df)),
        "train_positive_P": int(train_model_df["label"].sum()),
        "val_positive_P": int(val_model_df["label"].sum()),
        "test_positive_P": int(test_model_df["label"].sum()),
        "train_unlabeled_U": int((train_model_df["label"] == 0).sum()),
        "val_unlabeled_U": int((val_model_df["label"] == 0).sum()),
        "test_unlabeled_U": int((test_model_df["label"] == 0).sum()),
        "targets": targets,
        "n_targets": int(len(targets)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "pos_weight": float(pos_weight_value),
        "best_epoch": int(best_epoch),
        "best_val_pr_auc": float(best_val_pr),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "qc_policy": {
            "zero_as_missing": ZERO_AS_MISSING,
            "impute_strategy": IMPUTE_STRATEGY,
            "use_affinity_direction": USE_AFFINITY_DIRECTION,
            "add_target_success_flags": ADD_TARGET_SUCCESS_FLAGS,
            "add_coverage_feature": ADD_COVERAGE_FEATURE,
        }
    }
    with open(os.path.join(seed_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    summary_rows.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr,
        "train_roc_auc": train_metrics["roc_auc"],
        "train_pr_auc": train_metrics["pr_auc"],
        "train_p_at_20": train_metrics["p_at_20"],
        "train_p_at_50": train_metrics["p_at_50"],
        "train_p_at_100": train_metrics["p_at_100"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_p_at_20": val_metrics["p_at_20"],
        "val_p_at_50": val_metrics["p_at_50"],
        "val_p_at_100": val_metrics["p_at_100"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_p_at_20": test_metrics["p_at_20"],
        "test_p_at_50": test_metrics["p_at_50"],
        "test_p_at_100": test_metrics["p_at_100"],
    })

# =========================
# 7. Summary
# =========================
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "multiseed_summary.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

metric_cols = [
    "best_epoch", "best_val_pr_auc",
    "train_roc_auc", "train_pr_auc", "train_p_at_20", "train_p_at_50", "train_p_at_100",
    "val_roc_auc", "val_pr_auc", "val_p_at_20", "val_p_at_50", "val_p_at_100",
    "test_roc_auc", "test_pr_auc", "test_p_at_20", "test_p_at_50", "test_p_at_100",
]

stat_rows = []
for c in metric_cols:
    vals = summary_df[c].astype(float).values
    stat_rows.append({
        "metric": c,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    })
stat_df = pd.DataFrame(stat_rows)
stat_csv = os.path.join(OUT_DIR, "multiseed_mean_std.csv")
stat_df.to_csv(stat_csv, index=False, encoding="utf-8-sig")

summary_txt = os.path.join(OUT_DIR, "multiseed_summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Docking-only QC MLP multiseed summary\n")
    f.write("=" * 80 + "\n")
    f.write("SPLIT_MODE: {}\n".format(SPLIT_MODE))
    f.write("SOURCE_TYPE: {}\n".format(source_type))
    f.write("SEEDS: {}\n".format(SEEDS))
    f.write("ZERO_AS_MISSING: {}\n".format(ZERO_AS_MISSING))
    f.write("IMPUTE_STRATEGY: {}\n".format(IMPUTE_STRATEGY))
    f.write("USE_AFFINITY_DIRECTION: {}\n".format(USE_AFFINITY_DIRECTION))
    f.write("ADD_TARGET_SUCCESS_FLAGS: {}\n".format(ADD_TARGET_SUCCESS_FLAGS))
    f.write("ADD_COVERAGE_FEATURE: {}\n".format(ADD_COVERAGE_FEATURE))
    f.write("N_FEATURES: {}\n".format(len(feature_cols)))
    f.write("FEATURE_COLS: {}\n\n".format(feature_cols))
    f.write("Per-seed summary:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\nMean ± std:\n")
    for _, r in stat_df.iterrows():
        f.write("{}: {:.4f} ± {:.4f}\n".format(r["metric"], r["mean"], r["std"]))

print("\n" + "=" * 80)
print("[OK] saved multiseed summary: {}".format(summary_csv))
print("[OK] saved multiseed mean±std: {}".format(stat_csv))
print("[OK] saved multiseed txt: {}".format(summary_txt))
print("[OK] saved QC tables: {}".format(qc_dir))

for split_name in ["val", "test"]:
    print("\n[{} mean ± std]".format(split_name.upper()))
    sub = stat_df[stat_df["metric"].str.startswith(split_name + "_")].copy()
    for _, r in sub.iterrows():
        print("{}: {:.4f} ± {:.4f}".format(r["metric"], r["mean"], r["std"]))
