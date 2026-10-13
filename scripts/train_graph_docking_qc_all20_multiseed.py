# -*- coding: utf-8 -*-
"""
QC Graph-Docking Fusion, all 20 docking targets, multi-seed scaffold split.

Purpose
-------
This script is the QC version of train_graph_docking_all20_multiseed.py.
It keeps the same graph branch, same scaffold split, same seed list and same metrics,
but reconstructs docking features with failed-pose-aware QC before fusion.

Main design
-----------
Graph branch:
    GINEConv -> graph_emb (128)

QC docking branch:
    20 target affinity z-score features
    20 target pose-success flag features
    1 docking coverage feature
    -> MLP -> dock_emb (64)

Fusion:
    concat(graph_emb, dock_emb) -> prediction head -> logit

Outputs
-------
/home/zyh/aging_proj/results/graph_docking_qc_all20_multiseed_scaffold/
    seed_xxx/
        metrics.json
        training_history.csv
        train/val/test_predictions.csv
        test_top20.csv / test_top50.csv / test_top100.csv
        docking_qc_norm_stats.csv
    multiseed_summary.csv
    multiseed_mean_std.csv
    multiseed_summary.txt
    qc_tables/
        target_qc_valid_rate_all.csv
        compound_coverage_summary.csv
"""
from repo_paths import project_path

import os
import json
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, BatchNorm

# =========================
# 0. Paths and settings
# =========================
DATA_DIR = project_path('data/processed')
SCRIPTS_DIR = project_path('scripts')
RESULTS_DIR = project_path('results')

GRAPH_DATASET_PATH = os.path.join(DATA_DIR, "graph_dataset.pt")
GRAPH_INPUT_PATH = os.path.join(DATA_DIR, "graph_input.csv")

RAW_DOCKING_XLSX = os.path.join(DATA_DIR, "docking_input.xlsx")
RAW_DOCKING_CSV = os.path.join(DATA_DIR, "docking_input.csv")
FALLBACK_TOP3_CSV = os.path.join(DATA_DIR, "docking_features_20_top3mean.csv")

SPLIT_MODE = "scaffold"
if SPLIT_MODE == "random":
    SPLIT_CANDIDATES = [
        os.path.join(DATA_DIR, "split_idx.pt"),
        os.path.join(SCRIPTS_DIR, "split_idx.pt"),
    ]
    OUT_DIR = os.path.join(RESULTS_DIR, "graph_docking_qc_all20_multiseed_random")
elif SPLIT_MODE == "scaffold":
    SPLIT_CANDIDATES = [
        os.path.join(DATA_DIR, "split_idx_scaffold.pt"),
        os.path.join(SCRIPTS_DIR, "split_idx_scaffold.pt"),
    ]
    OUT_DIR = os.path.join(RESULTS_DIR, "graph_docking_qc_all20_multiseed_scaffold")
else:
    raise ValueError("SPLIT_MODE must be 'random' or 'scaffold'")

os.makedirs(OUT_DIR, exist_ok=True)
QC_DIR = os.path.join(OUT_DIR, "qc_tables")
os.makedirs(QC_DIR, exist_ok=True)

SEEDS = [42, 412, 719, 666, 888]

# Keep exactly the same 20-target order as the previous fusion script.
DOCK_FEATURES = [
    "1A9U", "2YXJ", "3O96", "4CFE", "4HHY",
    "4JPS", "4JSX", "4KIK", "4LXZ", "4OYS",
    "4WXX", "4XJT", "4ZZJ", "6MXE", "6OOK",
    "6SP1", "6VGL", "7PZC", "8DSE", "9KTK"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Utility functions
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

def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.4f}"

def format_metrics(m):
    return (
        f"ROC-AUC={fmt(m['roc_auc'])} | "
        f"PR-AUC={fmt(m['pr_auc'])} | "
        f"P@20={fmt(m['p_at_20'])} | "
        f"P@50={fmt(m['p_at_50'])} | "
        f"P@100={fmt(m['p_at_100'])}"
    )

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

def read_docking_source():
    """
    Prefer raw docking_input.xlsx so we can infer failed pose and valid pose count.
    Fallback to 20-dim top3mean CSV if raw file is unavailable.
    """
    if os.path.exists(RAW_DOCKING_XLSX):
        print(f"[INFO] use raw docking xlsx: {RAW_DOCKING_XLSX}")
        return pd.read_excel(RAW_DOCKING_XLSX), "raw_xlsx"
    if os.path.exists(RAW_DOCKING_CSV):
        print(f"[INFO] use raw docking csv: {RAW_DOCKING_CSV}")
        return pd.read_csv(RAW_DOCKING_CSV), "raw_csv"
    if os.path.exists(FALLBACK_TOP3_CSV):
        print(f"[WARN] raw docking input not found, fallback to top3mean csv: {FALLBACK_TOP3_CSV}")
        return pd.read_csv(FALLBACK_TOP3_CSV), "fallback_top3_csv"
    raise FileNotFoundError("No docking input found.")

def get_smiles_col(df):
    for c in ["SMILES", "smiles", "canonical_smiles", "canon_smiles"]:
        if c in df.columns:
            return c
    raise ValueError("Docking file has no SMILES/smiles column.")

def infer_rank_cols(df, target):
    """
    Try to find rank-level docking columns for a target.
    Compatible with common names like:
        4XJT__dg_rank1 ... 4XJT__dg_rank5
        4XJT__rank1 ... 4XJT__rank5
        4XJT__1 ... 4XJT__5
    """
    cols = []
    for i in range(1, 6):
        candidates = [
            f"{target}__dg_rank{i}",
            f"{target}__rank{i}",
            f"{target}__pose{i}",
            f"{target}__score_rank{i}",
            f"{target}__score{i}",
            f"{target}__{i}",
        ]
        hit = None
        for c in candidates:
            if c in df.columns:
                hit = c
                break
        if hit is not None:
            cols.append(hit)
    return cols

def build_qc_docking_features(dock_df, source_type):
    """
    Build failed-pose-aware docking features.

    For each target:
      - energy_qc: mean of valid nonzero rank scores if rank columns exist;
                   otherwise target__mean_top3 or target column with 0 treated as missing.
      - success_flag: 1 if at least one valid nonzero pose/score exists, else 0.
      - valid_pose_count: number of valid nonzero pose scores when available.

    Compound-level:
      - valid_target_count
      - docking_coverage = valid_target_count / 20

    Model features:
      - aff_<target> = -energy_qc, z-scored later using train statistics
      - succ_<target> = success flag
      - docking_coverage, z-scored later using train statistics
    """
    smiles_col = get_smiles_col(dock_df)
    out = dock_df[[smiles_col]].copy()
    out = out.rename(columns={smiles_col: "SMILES"})
    out["canon_smiles"] = out["SMILES"].apply(canonicalize_smiles)

    target_qc_rows = []

    for t in DOCK_FEATURES:
        rank_cols = infer_rank_cols(dock_df, t)

        if len(rank_cols) > 0:
            mat = dock_df[rank_cols].apply(pd.to_numeric, errors="coerce")
            valid = mat.notna() & (mat != 0)
            valid_pose_count = valid.sum(axis=1).astype(float)
            energy = mat.where(valid).mean(axis=1)
            success = (valid_pose_count > 0).astype(float)
        else:
            # Use top3mean if available; otherwise use direct target column.
            possible_cols = [
                f"{t}__mean_top3",
                f"{t}__top3mean",
                f"{t}__mean_top5",
                f"{t}__dg_rank1",
                t,
            ]
            col = None
            for c in possible_cols:
                if c in dock_df.columns:
                    col = c
                    break
            if col is None:
                raise ValueError(f"Cannot find docking columns for target {t}")

            s = pd.to_numeric(dock_df[col], errors="coerce")
            valid = s.notna() & (s != 0)
            energy = s.where(valid)
            success = valid.astype(float)
            valid_pose_count = success.copy()

        # energy is usually negative. affinity is positive-direction feature.
        out[f"energy_{t}"] = energy
        out[f"aff_{t}"] = -energy
        out[f"succ_{t}"] = success
        out[f"valid_pose_count_{t}"] = valid_pose_count

        target_qc_rows.append({
            "target": t,
            "rank_cols_found": ";".join(rank_cols) if len(rank_cols) > 0 else "NA",
            "valid_rate_all": float(success.mean()),
            "missing_or_failed_rate_all": float(1.0 - success.mean()),
            "median_valid_pose_count_all": float(pd.Series(valid_pose_count).median()),
            "source_type": source_type,
        })

    succ_cols = [f"succ_{t}" for t in DOCK_FEATURES]
    out["valid_target_count"] = out[succ_cols].sum(axis=1)
    out["docking_coverage"] = out["valid_target_count"] / float(len(DOCK_FEATURES))

    # Remove rows without canonical smiles and duplicates.
    out = out.dropna(subset=["canon_smiles"]).copy()
    out = out.drop_duplicates(subset=["canon_smiles"], keep="first").copy()

    target_qc_df = pd.DataFrame(target_qc_rows)
    target_qc_df.to_csv(os.path.join(QC_DIR, "target_qc_valid_rate_all.csv"), index=False, encoding="utf-8-sig")

    coverage_df = out[["SMILES", "canon_smiles", "valid_target_count", "docking_coverage"]].copy()
    coverage_df.to_csv(os.path.join(QC_DIR, "compound_coverage_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"[QC] saved target QC and compound coverage tables to: {QC_DIR}")
    print("[QC] target valid rate:")
    print(target_qc_df[["target", "valid_rate_all", "missing_or_failed_rate_all"]].to_string(index=False))
    print("[QC] compound docking coverage summary:")
    print(coverage_df["docking_coverage"].describe().to_string())

    return out

# =========================
# 2. Load graph, split and QC docking data
# =========================
split_path = find_first_existing(SPLIT_CANDIDATES)
if split_path is None:
    raise FileNotFoundError(f"Split file not found in: {SPLIT_CANDIDATES}")

graph_list = safe_torch_load(GRAPH_DATASET_PATH)
graph_df = pd.read_csv(GRAPH_INPUT_PATH)
split_dict = safe_torch_load(split_path)

if len(graph_list) != len(graph_df):
    raise ValueError("graph_dataset.pt and graph_input.csv have different row counts.")

dock_raw_df, docking_source_type = read_docking_source()
dock_qc_df = build_qc_docking_features(dock_raw_df, docking_source_type)

print(f"[INFO] split mode = {SPLIT_MODE}")
print(f"[INFO] device = {DEVICE}")
print(f"[INFO] graph_dataset = {GRAPH_DATASET_PATH}")
print(f"[INFO] graph_input = {GRAPH_INPUT_PATH}")
print(f"[INFO] split = {split_path}")
print(f"[INFO] docking source type = {docking_source_type}")

graph_df["split"] = None
for split_name in ["train", "val", "test"]:
    graph_df.loc[split_dict[split_name], "split"] = split_name

graph_df["canon_smiles"] = graph_df["smiles"].apply(canonicalize_smiles)

aff_cols = [f"aff_{t}" for t in DOCK_FEATURES]
succ_cols = [f"succ_{t}" for t in DOCK_FEATURES]
qc_feature_cols = aff_cols + succ_cols + ["docking_coverage"]

# map QC docking features by canonical smiles
dock_map = dock_qc_df.set_index("canon_smiles")[qc_feature_cols].to_dict(orient="index")

matched_data = []
missing_smiles = []

for i, g in enumerate(graph_list):
    row = graph_df.iloc[i]
    canon = row["canon_smiles"]
    if pd.isna(canon) or canon not in dock_map:
        missing_smiles.append(canon)
        continue

    vec = np.array([dock_map[canon][c] for c in qc_feature_cols], dtype=np.float32)
    # NaN in affinity is allowed for now; will be imputed from train median per seed.
    data = copy.deepcopy(g)
    data.raw_qc_dock_feat = torch.tensor(vec, dtype=torch.float).view(1, -1)
    data.split = row["split"]
    data.canon_smiles = canon
    data.uid = str(row["pu_id"])
    matched_data.append(data)

print(f"[INFO] matched fused samples = {len(matched_data)}")
print(f"[INFO] missing/unmatched samples = {len(missing_smiles)}")

with open(os.path.join(OUT_DIR, "missing_smiles.txt"), "w", encoding="utf-8") as f:
    for x in missing_smiles:
        f.write(f"{x}\n")

base_train_dataset = [d for d in matched_data if d.split == "train"]
base_val_dataset   = [d for d in matched_data if d.split == "val"]
base_test_dataset  = [d for d in matched_data if d.split == "test"]

print(f"[INFO] train/val/test = {len(base_train_dataset)}/{len(base_val_dataset)}/{len(base_test_dataset)}")
print(f"[INFO] train P = {sum(int(d.y.item()) for d in base_train_dataset)}, U = {sum(int(d.y.item()) == 0 for d in base_train_dataset)}")
print(f"[INFO] val   P = {sum(int(d.y.item()) for d in base_val_dataset)}, U = {sum(int(d.y.item()) == 0 for d in base_val_dataset)}")
print(f"[INFO] test  P = {sum(int(d.y.item()) for d in base_test_dataset)}, U = {sum(int(d.y.item()) == 0 for d in base_test_dataset)}")

node_dim = base_train_dataset[0].x.size(1)
edge_dim = base_train_dataset[0].edge_attr.size(1)
dock_dim = len(qc_feature_cols)

# =========================
# 3. Model
# =========================
def make_gine_block(hidden_dim, edge_dim):
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )
    conv = GINEConv(nn=mlp, edge_dim=edge_dim)
    bn = BatchNorm(hidden_dim)
    return conv, bn

class GraphDockingQCFusion(nn.Module):
    def __init__(self, node_dim, edge_dim, dock_dim,
                 graph_hidden=128, dock_hidden=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = nn.Linear(node_dim, graph_hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            conv, bn = make_gine_block(graph_hidden, edge_dim)
            self.convs.append(conv)
            self.bns.append(bn)

        self.dock_encoder = nn.Sequential(
            nn.Linear(dock_dim, dock_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dock_hidden, dock_hidden),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(graph_hidden + dock_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        # Graph branch
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        batch = data.batch

        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = global_add_pool(x, batch)

        # QC docking branch
        dock_feat = data.dock_feat.float()
        dock_emb = self.dock_encoder(dock_feat)

        fused = torch.cat([graph_emb, dock_emb], dim=1)
        logits = self.head(fused).view(-1)

        return logits, graph_emb, dock_emb, fused

# =========================
# 4. Seed dataset builder
# =========================
def build_seed_datasets():
    """
    Per seed, deepcopy base data and normalize QC docking features using train statistics only.

    Feature handling:
      - affinity features: NaN imputed by train median, then z-scored.
      - success flags: keep as 0/1.
      - docking_coverage: z-scored with train mean/std.
    """
    train_dataset = [copy.deepcopy(d) for d in base_train_dataset]
    val_dataset   = [copy.deepcopy(d) for d in base_val_dataset]
    test_dataset  = [copy.deepcopy(d) for d in base_test_dataset]

    train_raw = np.vstack([d.raw_qc_dock_feat.numpy().reshape(-1) for d in train_dataset])

    n_aff = len(aff_cols)
    n_succ = len(succ_cols)
    cov_idx = n_aff + n_succ

    # Train median for affinity imputation
    aff_train = train_raw[:, :n_aff]
    aff_median = np.nanmedian(aff_train, axis=0)
    # In case an entire target is NaN in training, fallback to 0.
    aff_median = np.where(np.isnan(aff_median), 0.0, aff_median)

    aff_train_imp = np.where(np.isnan(aff_train), aff_median.reshape(1, -1), aff_train)
    aff_mean = aff_train_imp.mean(axis=0)
    aff_std = aff_train_imp.std(axis=0)
    aff_std[aff_std == 0] = 1.0

    cov_train = train_raw[:, cov_idx]
    cov_mean = np.nanmean(cov_train)
    cov_std = np.nanstd(cov_train)
    if cov_std == 0 or np.isnan(cov_std):
        cov_std = 1.0
    if np.isnan(cov_mean):
        cov_mean = 0.0

    def transform_vec(vec):
        vec = vec.copy()

        # affinity impute + z-score
        aff = vec[:n_aff]
        aff = np.where(np.isnan(aff), aff_median, aff)
        aff = (aff - aff_mean) / aff_std

        # success flags: NaN -> 0, keep 0/1
        succ = vec[n_aff:n_aff+n_succ]
        succ = np.where(np.isnan(succ), 0.0, succ)

        # coverage z-score
        cov = vec[cov_idx]
        if np.isnan(cov):
            cov = 0.0
        cov = (cov - cov_mean) / cov_std

        return np.concatenate([aff, succ, np.array([cov], dtype=np.float32)]).astype(np.float32)

    for ds in [train_dataset, val_dataset, test_dataset]:
        for d in ds:
            x = d.raw_qc_dock_feat.numpy().reshape(-1)
            x = transform_vec(x)
            d.dock_feat = torch.tensor(x, dtype=torch.float).view(1, -1)

    norm_df = pd.DataFrame({
        "feature": aff_cols,
        "aff_train_median": aff_median,
        "aff_train_mean_after_imputation": aff_mean,
        "aff_train_std_after_imputation": aff_std,
    })
    norm_extra = pd.DataFrame({
        "feature": ["docking_coverage"],
        "aff_train_median": [np.nan],
        "aff_train_mean_after_imputation": [cov_mean],
        "aff_train_std_after_imputation": [cov_std],
    })
    norm_df = pd.concat([norm_df, norm_extra], axis=0, ignore_index=True)

    return train_dataset, val_dataset, test_dataset, norm_df

# =========================
# 5. Train and evaluate
# =========================
def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits, _, _, _ = model(batch)
        y = batch.y.view(-1).float()
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)

@torch.no_grad()
def evaluate(loader, model, split_name="val", save_pred=False, seed_dir=None):
    model.eval()

    all_y = []
    all_prob = []
    all_uid = []
    all_smiles = []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _, _, _ = model(batch)
        prob = torch.sigmoid(logits)

        y = batch.y.view(-1).detach().cpu().numpy()
        p = prob.detach().cpu().numpy()

        all_y.append(y)
        all_prob.append(p)

        uid = getattr(batch, "uid", None)
        smiles = getattr(batch, "canon_smiles", None)

        if isinstance(uid, (list, tuple)):
            all_uid.extend([str(x) for x in uid])
        else:
            all_uid.extend([None] * len(y))

        if isinstance(smiles, (list, tuple)):
            all_smiles.extend([str(x) for x in smiles])
        else:
            all_smiles.extend([None] * len(y))

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_prob)

    metrics = eval_metrics(y_true, y_score)

    pred_df = pd.DataFrame({
        "uid": all_uid,
        "canon_smiles": all_smiles,
        "y_true": y_true,
        "score": y_score
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if save_pred and seed_dir is not None:
        pred_path = os.path.join(seed_dir, f"{split_name}_predictions.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return metrics, pred_df

# =========================
# 6. Multi-seed main loop
# =========================
summary_rows = []

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"[RUN] SEED = {seed}")
    set_seed(seed)

    seed_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    train_dataset, val_dataset, test_dataset, norm_df = build_seed_datasets()
    norm_df.to_csv(os.path.join(seed_dir, "docking_qc_norm_stats.csv"), index=False, encoding="utf-8-sig")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = GraphDockingQCFusion(
        node_dim=node_dim,
        edge_dim=edge_dim,
        dock_dim=dock_dim,
        graph_hidden=128,
        dock_hidden=64,
        num_layers=3,
        dropout=0.3
    ).to(DEVICE)

    y_train = np.array([int(d.y.item()) for d in train_dataset])
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float, device=DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_pr = -1.0
    best_epoch = -1
    bad_epochs = 0
    patience = 20
    history = []

    for epoch in range(1, 101):
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion)

        train_metrics, _ = evaluate(train_loader, model, split_name="train", save_pred=False)
        val_metrics, _   = evaluate(val_loader, model, split_name="val", save_pred=False)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_p_at_20": val_metrics["p_at_20"],
            "val_p_at_50": val_metrics["p_at_50"],
            "val_p_at_100": val_metrics["p_at_100"],
        })

        print(
            f"[Seed {seed}][Epoch {epoch:03d}] "
            f"loss={train_loss:.4f} | "
            f"train PR-AUC={train_metrics['pr_auc']:.4f} | "
            f"val PR-AUC={val_metrics['pr_auc']:.4f} | "
            f"val P@50={val_metrics['p_at_50']:.4f}"
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
                "qc_feature_cols": qc_feature_cols,
                "aff_cols": aff_cols,
                "succ_cols": succ_cols,
            }
            torch.save(ckpt, os.path.join(seed_dir, "best_model.pt"))
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[INFO] early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    best_ckpt = safe_torch_load(os.path.join(seed_dir, "best_model.pt"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    train_metrics, train_pred = evaluate(train_loader, model, split_name="train", save_pred=True, seed_dir=seed_dir)
    val_metrics, val_pred     = evaluate(val_loader, model, split_name="val", save_pred=True, seed_dir=seed_dir)
    test_metrics, test_pred   = evaluate(test_loader, model, split_name="test", save_pred=True, seed_dir=seed_dir)

    print(f"[FINAL][SEED {seed}] TRAIN : {format_metrics(train_metrics)}")
    print(f"[FINAL][SEED {seed}] VAL   : {format_metrics(val_metrics)}")
    print(f"[FINAL][SEED {seed}] TEST  : {format_metrics(test_metrics)}")

    pd.DataFrame(history).to_csv(os.path.join(seed_dir, "training_history.csv"), index=False, encoding="utf-8-sig")

    metrics_dict = {
        "seed": seed,
        "split_mode": SPLIT_MODE,
        "graph_dataset": GRAPH_DATASET_PATH,
        "graph_input": GRAPH_INPUT_PATH,
        "docking_source_type": docking_source_type,
        "split_file": split_path,
        "matched_samples": len(matched_data),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_positive_P": int(sum(int(d.y.item()) for d in train_dataset)),
        "val_positive_P": int(sum(int(d.y.item()) for d in val_dataset)),
        "test_positive_P": int(sum(int(d.y.item()) for d in test_dataset)),
        "train_unlabeled_U": int(sum(int(d.y.item()) == 0 for d in train_dataset)),
        "val_unlabeled_U": int(sum(int(d.y.item()) == 0 for d in val_dataset)),
        "test_unlabeled_U": int(sum(int(d.y.item()) == 0 for d in test_dataset)),
        "qc_feature_cols": qc_feature_cols,
        "dock_dim": len(qc_feature_cols),
        "node_dim": int(node_dim),
        "edge_dim": int(edge_dim),
        "pos_weight": float(pos_weight_value),
        "best_epoch": int(best_epoch),
        "best_val_pr_auc": float(best_val_pr),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(seed_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    for k in [20, 50, 100]:
        test_pred.head(k).to_csv(os.path.join(seed_dir, f"test_top{k}.csv"), index=False, encoding="utf-8-sig")

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
    f.write(f"SPLIT_MODE: {SPLIT_MODE}\n")
    f.write(f"SEEDS: {SEEDS}\n")
    f.write(f"DOCKING_SOURCE_TYPE: {docking_source_type}\n")
    f.write(f"QC_FEATURE_DIM: {len(qc_feature_cols)}\n")
    f.write(f"QC_FEATURE_COLS: {qc_feature_cols}\n\n")

    f.write("Per-seed summary:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\nMean ± std:\n")
    for _, r in stat_df.iterrows():
        f.write(f"{r['metric']}: {r['mean']:.4f} ± {r['std']:.4f}\n")

print("\n" + "=" * 80)
print(f"[OK] saved multiseed summary: {summary_csv}")
print(f"[OK] saved multiseed mean±std: {stat_csv}")
print(f"[OK] saved multiseed txt: {summary_txt}")

for split_name in ["val", "test"]:
    print(f"\n[{split_name.upper()} mean ± std]")
    sub = stat_df[stat_df["metric"].str.startswith(split_name + "_")].copy()
    for _, r in sub.iterrows():
        print(f"{r['metric']}: {r['mean']:.4f} ± {r['std']:.4f}")
