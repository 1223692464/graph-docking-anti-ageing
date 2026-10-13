# -*- coding: utf-8 -*-
"""
Screen FMH candidates with QC Graph-Docking Fusion multi-seed models.

This version does NOT require a prebuilt candidate graph_dataset.pt.
It builds molecular graphs directly from SMILES in:
    /home/zyh/aging_proj/reports/FMH_screen_input_for_best_svm.csv

It uses trained QC Graph-Docking Fusion checkpoints from:
    /home/zyh/aging_proj/results/graph_docking_qc_all20_multiseed_scaffold

Outputs:
    /home/zyh/aging_proj/results/screening_qc_graph_docking_multiseed_v2
"""
from repo_paths import project_path

import os
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, BatchNorm

INPUT_FILE = project_path('reports/FMH_screen_input_for_best_svm.csv')
MODEL_ROOT = project_path('results/graph_docking_qc_all20_multiseed_scaffold')
OUT_DIR = project_path('results/screening_qc_graph_docking_multiseed_v2')
QC_DIR = os.path.join(OUT_DIR, "qc_tables")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

SEEDS = [42, 412, 719, 666, 888]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

DOCK_FEATURES = [
    "1A9U", "2YXJ", "3O96", "4CFE", "4HHY",
    "4JPS", "4JSX", "4KIK", "4LXZ", "4OYS",
    "4WXX", "4XJT", "4ZZJ", "6MXE", "6OOK",
    "6SP1", "6VGL", "7PZC", "8DSE", "9KTK"
]
TARGET_ALIASES = {"6OOK": ["6OOK", "6O0K"]}
POSITIVE_LOGIT_THRESHOLD = 0.0
COV12 = 12.0 / 20.0
COV15 = 15.0 / 20.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext}")


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def detect_smiles_col(df):
    for c in ["SMILES", "smiles", "smiles_canonical", "Canonical_SMILES", "canonical_smiles"]:
        if c in df.columns:
            return c
    raise ValueError("No SMILES column found. Need SMILES/smiles/smiles_canonical.")


def detect_id_col(df):
    for c in ["screen_id", "ligand_id_pu", "pu_id", "ligand_id", "compound_id", "molecule_id", "id", "ID"]:
        if c in df.columns:
            return c
    return None


def detect_title_col(df):
    for c in ["compound_title", "title", "name", "compound_name", "molecule_name", "Name"]:
        if c in df.columns:
            return c
    return None


def detect_herb_col(df):
    for c in ["herb_name", "herb", "source_herb", "herb_source", "food_name", "source_food", "Chinese_name"]:
        if c in df.columns:
            return c
    return None


def sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def target_display_name(t):
    return "6O0K" if t == "6OOK" else t


def aliases_for_target(t):
    out = [t]
    for a in TARGET_ALIASES.get(t, []):
        if a not in out:
            out.append(a)
    return out


def possible_rank_cols(alias, rank_i):
    return [
        f"{alias}__dg_rank{rank_i}", f"{alias}__rank{rank_i}",
        f"{alias}__pose{rank_i}", f"{alias}__score_rank{rank_i}",
        f"{alias}__score{rank_i}", f"{alias}__{rank_i}",
        f"{alias}_dg_rank{rank_i}", f"{alias}_rank{rank_i}",
        f"{alias}_pose{rank_i}", f"{alias}_score{rank_i}",
    ]


def infer_rank_cols(df, target):
    cols = []
    for i in range(1, 6):
        hit = None
        for alias in aliases_for_target(target):
            for c in possible_rank_cols(alias, i):
                if c in df.columns:
                    hit = c
                    break
            if hit is not None:
                break
        if hit is not None:
            cols.append(hit)
    return cols


def possible_single_cols(target):
    suffixes = [
        "__mean_top3", "__top3mean", "__top3mean_excl0", "__top3_mean_excl0",
        "__mean_top5", "__mean_all5_excl0", "__mean_all5_incl0", "__dg_rank1",
        "_mean_top3", "_top3mean", "_top3mean_excl0", "_top3_mean_excl0",
        "_mean_top5", "_mean_all5_excl0", "_mean_all5_incl0", "_dg_rank1", "",
    ]
    cols = []
    for alias in aliases_for_target(target):
        for suf in suffixes:
            x = f"{alias}{suf}"
            if x not in cols:
                cols.append(x)
    return cols


def one_hot_unknown(x, choices):
    if x not in choices:
        x = choices[-1]
    return [int(x == c) for c in choices]


ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53, 999]


def atom_features(atom):
    feats = []
    feats += one_hot_unknown(atom.GetAtomicNum(), ATOM_TYPES)
    feats += one_hot_unknown(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    feats += one_hot_unknown(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 99])
    feats += one_hot_unknown(int(atom.GetHybridization()), [0, 1, 2, 3, 4, 5, 6, 99])
    feats += [int(atom.GetIsAromatic())]
    feats += [int(atom.IsInRing())]
    feats += one_hot_unknown(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 99])
    feats += one_hot_unknown(int(atom.GetChiralTag()), [0, 1, 2, 3, 99])
    return feats


def bond_features(bond):
    bt = bond.GetBondType()
    feats = [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
    feats += one_hot_unknown(int(bond.GetStereo()), [0, 1, 2, 3, 4, 5, 99])
    return feats


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index_list, edge_attr_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [bf, bf]
    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 13), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    return x, edge_index, edge_attr


def build_qc_docking_features(df, smiles_col):
    out = df.copy()
    out["canon_smiles"] = out[smiles_col].apply(canonicalize_smiles)
    target_rows = []

    for t in DOCK_FEATURES:
        rank_cols = infer_rank_cols(out, t)
        if len(rank_cols) > 0:
            mat = out[rank_cols].apply(pd.to_numeric, errors="coerce")
            valid = mat.notna() & (mat != 0)
            valid_pose_count = valid.sum(axis=1).astype(float)
            energy = mat.where(valid).mean(axis=1)
            success = (valid_pose_count > 0).astype(float)
            source = "rank_cols"
            used = ";".join(rank_cols)
        else:
            col = None
            for c in possible_single_cols(t):
                if c in out.columns:
                    col = c
                    break
            if col is None:
                raise ValueError(f"Cannot find docking feature for target {t}. Checked aliases: {aliases_for_target(t)}")
            s = pd.to_numeric(out[col], errors="coerce")
            valid = s.notna() & (s != 0)
            energy = s.where(valid)
            success = valid.astype(float)
            valid_pose_count = success.copy()
            source = "single_summary_col"
            used = col

        out[f"energy_{t}"] = energy
        out[f"aff_{t}"] = -energy
        out[f"succ_{t}"] = success
        out[f"valid_pose_count_{t}"] = valid_pose_count
        target_rows.append({
            "target_model_name": t,
            "target_display": target_display_name(t),
            "source": source,
            "used_column_or_columns": used,
            "valid_rate": float(success.mean()),
            "failed_or_missing_rate": float(1.0 - success.mean()),
            "median_valid_pose_count": float(pd.Series(valid_pose_count).median()),
        })

    succ_cols = [f"succ_{t}" for t in DOCK_FEATURES]
    out["valid_target_count"] = out[succ_cols].sum(axis=1)
    out["docking_coverage"] = out["valid_target_count"] / float(len(DOCK_FEATURES))

    pd.DataFrame(target_rows).to_csv(os.path.join(QC_DIR, "candidate_target_qc_valid_rate.csv"), index=False, encoding="utf-8-sig")
    out[[smiles_col, "canon_smiles", "valid_target_count", "docking_coverage"]].to_csv(
        os.path.join(QC_DIR, "candidate_compound_coverage_summary.csv"), index=False, encoding="utf-8-sig"
    )
    print("[QC] compound docking coverage summary:")
    print(out["docking_coverage"].describe().to_string())
    return out


def make_gine_block(hidden_dim, edge_dim):
    mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
    conv = GINEConv(nn=mlp, edge_dim=edge_dim)
    bn = BatchNorm(hidden_dim)
    return conv, bn


class GraphDockingQCFusion(nn.Module):
    def __init__(self, node_dim, edge_dim, dock_dim, graph_hidden=128, dock_hidden=64, num_layers=3, dropout=0.3):
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
            nn.Linear(dock_dim, dock_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dock_hidden, dock_hidden), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(graph_hidden + dock_hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, data):
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
        dock_emb = self.dock_encoder(data.dock_feat.float())
        return self.head(torch.cat([graph_emb, dock_emb], dim=1)).view(-1)


def load_seed_norm(seed_dir, aff_cols):
    norm_path = os.path.join(seed_dir, "docking_qc_norm_stats.csv")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Cannot find QC norm stats: {norm_path}")
    norm_df = pd.read_csv(norm_path)
    norm_map = norm_df.set_index("feature").to_dict(orient="index")
    aff_median, aff_mean, aff_std = [], [], []
    for c in aff_cols:
        if c not in norm_map:
            raise ValueError(f"Missing feature {c} in {norm_path}")
        aff_median.append(float(norm_map[c]["aff_train_median"]))
        aff_mean.append(float(norm_map[c]["aff_train_mean_after_imputation"]))
        aff_std.append(float(norm_map[c]["aff_train_std_after_imputation"]))
    cov_row = norm_map.get("docking_coverage")
    if cov_row is None:
        raise ValueError(f"Missing docking_coverage in {norm_path}")
    cov_mean = float(cov_row["aff_train_mean_after_imputation"])
    cov_std = float(cov_row["aff_train_std_after_imputation"])
    if cov_std == 0 or math.isnan(cov_std):
        cov_std = 1.0
    aff_median = np.array(aff_median, dtype=np.float32)
    aff_mean = np.array(aff_mean, dtype=np.float32)
    aff_std = np.array(aff_std, dtype=np.float32)
    aff_std[aff_std == 0] = 1.0
    return aff_median, aff_mean, aff_std, cov_mean, cov_std


def transform_data_for_seed(base_list, seed_dir, aff_cols, succ_cols):
    aff_median, aff_mean, aff_std, cov_mean, cov_std = load_seed_norm(seed_dir, aff_cols)
    n_aff, n_succ = len(aff_cols), len(succ_cols)
    cov_idx = n_aff + n_succ
    seed_data = []
    for d in base_list:
        dd = copy.deepcopy(d)
        vec = dd.raw_qc_dock_feat.numpy().reshape(-1).astype(np.float32)
        aff = vec[:n_aff]
        aff = np.where(np.isnan(aff), aff_median, aff)
        aff = (aff - aff_mean) / aff_std
        succ = vec[n_aff:n_aff+n_succ]
        succ = np.where(np.isnan(succ), 0.0, succ)
        cov = vec[cov_idx]
        if np.isnan(cov):
            cov = 0.0
        cov = (cov - cov_mean) / cov_std
        feat = np.concatenate([aff, succ, np.array([cov], dtype=np.float32)]).astype(np.float32)
        dd.dock_feat = torch.tensor(feat, dtype=torch.float).view(1, -1)
        seed_data.append(dd)
    return seed_data


def score_one_seed(seed, base_list, node_dim, edge_dim, dock_dim, aff_cols, succ_cols):
    seed_dir = os.path.join(MODEL_ROOT, f"seed_{seed}")
    ckpt_path = os.path.join(seed_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find model checkpoint: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path)
    model = GraphDockingQCFusion(node_dim=node_dim, edge_dim=edge_dim, dock_dim=dock_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seed_data = transform_data_for_seed(base_list, seed_dir, aff_cols, succ_cols)
    loader = DataLoader(seed_data, batch_size=BATCH_SIZE, shuffle=False)
    logits_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits_all.extend(model(batch).detach().cpu().numpy().tolist())
    logits_all = np.array(logits_all, dtype=float)
    return logits_all, sigmoid_np(logits_all)


# =========================
# Main
# =========================
print(f"[INFO] INPUT_FILE = {INPUT_FILE}")
print(f"[INFO] MODEL_ROOT = {MODEL_ROOT}")
print(f"[INFO] DEVICE = {DEVICE}")

df0 = load_table(INPUT_FILE)
smiles_col = detect_smiles_col(df0)
id_col = detect_id_col(df0)
title_col = detect_title_col(df0)
herb_col = detect_herb_col(df0)

df = df0.copy()
if id_col is None:
    df["screen_id"] = [f"S_{i+1:06d}" for i in range(len(df))]
    id_col = "screen_id"
else:
    df["screen_id"] = df[id_col].astype(str)

df = build_qc_docking_features(df, smiles_col)
aff_cols = [f"aff_{t}" for t in DOCK_FEATURES]
succ_cols = [f"succ_{t}" for t in DOCK_FEATURES]
qc_feature_cols = aff_cols + succ_cols + ["docking_coverage"]

valid_rows, invalid_rows, base_data_list = [], [], []
for idx, row in df.iterrows():
    sid = str(row["screen_id"])
    canon = row["canon_smiles"]
    if pd.isna(canon):
        invalid_rows.append({"screen_id": sid, "row_index": idx, "reason": "invalid_smiles"})
        continue
    graph = smiles_to_graph(canon)
    if graph is None:
        invalid_rows.append({"screen_id": sid, "row_index": idx, "reason": "graph_build_failed"})
        continue
    qc_vec = row[qc_feature_cols].values.astype(np.float32)
    x, edge_index, edge_attr = graph
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([0.0], dtype=torch.float))
    data.uid = sid
    data.canon_smiles = canon
    data.raw_qc_dock_feat = torch.tensor(qc_vec, dtype=torch.float).view(1, -1)
    valid_rows.append(row)
    base_data_list.append(data)

valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
invalid_df = pd.DataFrame(invalid_rows)
invalid_df.to_csv(os.path.join(OUT_DIR, "invalid_rows.csv"), index=False, encoding="utf-8-sig")

print(f"[INFO] input shape = {df0.shape}")
print(f"[INFO] valid molecules = {len(valid_df)}")
print(f"[INFO] invalid molecules = {len(invalid_df)}")
print(f"[INFO] smiles_col = {smiles_col}")
print(f"[INFO] id_col = {id_col}")
print(f"[INFO] title_col = {title_col}")
print(f"[INFO] herb_col = {herb_col}")

if len(base_data_list) == 0:
    raise ValueError("No valid molecules for screening.")

node_dim = base_data_list[0].x.size(1)
edge_dim = base_data_list[0].edge_attr.size(1)
dock_dim = len(qc_feature_cols)

results_df = valid_df.copy()
for seed in SEEDS:
    print(f"[SCORING] seed {seed}")
    set_seed(seed)
    logits, probs = score_one_seed(seed, base_data_list, node_dim, edge_dim, dock_dim, aff_cols, succ_cols)
    results_df[f"logit_seed{seed}"] = logits
    results_df[f"score_seed{seed}"] = probs
    results_df[f"pred_seed{seed}"] = (logits > POSITIVE_LOGIT_THRESHOLD).astype(int)
    results_df[f"rank_seed{seed}"] = pd.Series(logits).rank(method="average", ascending=False).values
    print(f"[OK] seed {seed} scored")

logit_cols = [f"logit_seed{s}" for s in SEEDS]
score_cols = [f"score_seed{s}" for s in SEEDS]
pred_cols = [f"pred_seed{s}" for s in SEEDS]
rank_cols = [f"rank_seed{s}" for s in SEEDS]

results_df["logit_mean"] = results_df[logit_cols].mean(axis=1)
results_df["logit_sd"] = results_df[logit_cols].std(axis=1, ddof=1)
results_df["score_mean"] = results_df[score_cols].mean(axis=1)
results_df["score_sd"] = results_df[score_cols].std(axis=1, ddof=1)
results_df["positive_vote"] = results_df[pred_cols].sum(axis=1)
results_df["rank_mean"] = results_df[rank_cols].mean(axis=1)
results_df["rank_sd"] = results_df[rank_cols].std(axis=1, ddof=1)
results_df["qc_pass_cov12"] = results_df["docking_coverage"] >= COV12
results_df["qc_pass_cov15"] = results_df["docking_coverage"] >= COV15

results_df = results_df.sort_values(["logit_mean", "logit_sd", "rank_mean"], ascending=[False, True, True]).reset_index(drop=True)
results_df["rank_final"] = np.arange(1, len(results_df) + 1)

full_path = os.path.join(OUT_DIR, "screening_qc_graph_docking_predictions.csv")
results_df.to_csv(full_path, index=False, encoding="utf-8-sig")
for k in [10, 20, 50, 100, 200]:
    results_df.head(k).to_csv(os.path.join(OUT_DIR, f"screening_qc_graph_docking_top{k}.csv"), index=False, encoding="utf-8-sig")

short_cov12 = results_df[(results_df["positive_vote"] == len(SEEDS)) & (results_df["qc_pass_cov12"])].copy()
short_cov15 = results_df[(results_df["positive_vote"] == len(SEEDS)) & (results_df["qc_pass_cov15"])].copy()
short_cov12.to_csv(os.path.join(OUT_DIR, "screening_qc_graph_docking_shortlist_cov12.csv"), index=False, encoding="utf-8-sig")
short_cov15.to_csv(os.path.join(OUT_DIR, "screening_qc_graph_docking_shortlist_cov15.csv"), index=False, encoding="utf-8-sig")
short_cov12.head(50).to_csv(os.path.join(OUT_DIR, "screening_qc_graph_docking_shortlist_cov12_top50.csv"), index=False, encoding="utf-8-sig")
short_cov15.head(50).to_csv(os.path.join(OUT_DIR, "screening_qc_graph_docking_shortlist_cov15_top50.csv"), index=False, encoding="utf-8-sig")

heatmap_cols = ["rank_final", "screen_id", "canon_smiles", "logit_mean", "logit_sd", "positive_vote", "valid_target_count", "docking_coverage"]
if title_col is not None and title_col in results_df.columns:
    heatmap_cols.insert(2, title_col)
if herb_col is not None and herb_col in results_df.columns:
    heatmap_cols.insert(3, herb_col)
heatmap_cols += [f"energy_{t}" for t in DOCK_FEATURES if f"energy_{t}" in results_df.columns]
results_df[heatmap_cols].head(20).to_csv(os.path.join(OUT_DIR, "top20_docking_qc_heatmap_input.csv"), index=False, encoding="utf-8-sig")

herb_out = None
if herb_col is not None and herb_col in results_df.columns:
    top50_ids = set(results_df.head(50)["screen_id"].astype(str))
    top100_ids = set(results_df.head(100)["screen_id"].astype(str))
    herb_rows = []
    for herb, sub in results_df.groupby(herb_col):
        sub = sub.sort_values(["logit_mean", "logit_sd"], ascending=[False, True]).reset_index(drop=True)
        herb_rows.append({
            herb_col: herb,
            "n_molecules": len(sub),
            "herb_max_logit": sub["logit_mean"].iloc[0],
            "herb_top3_logit_mean": sub["logit_mean"].head(min(3, len(sub))).mean(),
            "herb_top5_logit_mean": sub["logit_mean"].head(min(5, len(sub))).mean(),
            "hit_count_top50": sum(sub["screen_id"].astype(str).isin(top50_ids)),
            "hit_count_top100": sum(sub["screen_id"].astype(str).isin(top100_ids)),
            "mean_docking_coverage_top5": sub["docking_coverage"].head(min(5, len(sub))).mean(),
        })
    herb_df = pd.DataFrame(herb_rows).sort_values(["herb_top3_logit_mean", "herb_max_logit", "hit_count_top50"], ascending=[False, False, False]).reset_index(drop=True)
    herb_out = os.path.join(OUT_DIR, "screening_qc_results_herb_level_simple_check.csv")
    herb_df.to_csv(herb_out, index=False, encoding="utf-8-sig")

counts = {
    "valid_molecules": int(len(results_df)),
    "invalid_molecules": int(len(invalid_df)),
    "positive_vote_5": int((results_df["positive_vote"] == len(SEEDS)).sum()),
    "positive_vote_ge4": int((results_df["positive_vote"] >= 4).sum()),
    "positive_vote_ge3": int((results_df["positive_vote"] >= 3).sum()),
    "cov_ge_12": int(results_df["qc_pass_cov12"].sum()),
    "cov_ge_15": int(results_df["qc_pass_cov15"].sum()),
    "vote5_cov_ge_12": int(len(short_cov12)),
    "vote5_cov_ge_15": int(len(short_cov15)),
}

summary_path = os.path.join(OUT_DIR, "screening_qc_summary.txt")
show_cols = ["rank_final", "screen_id", "positive_vote", "logit_mean", "logit_sd", "valid_target_count", "docking_coverage", "score_mean", "score_sd"]
if title_col is not None and title_col in results_df.columns:
    show_cols.insert(2, title_col)
if herb_col is not None and herb_col in results_df.columns:
    show_cols.insert(3, herb_col)
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("QC Graph-Docking candidate screening summary\n")
    f.write("=" * 80 + "\n")
    f.write(f"INPUT_FILE: {INPUT_FILE}\nMODEL_ROOT: {MODEL_ROOT}\nOUT_DIR: {OUT_DIR}\n")
    f.write(f"SEEDS: {SEEDS}\nDEVICE: {DEVICE}\n")
    f.write(f"SMILES_COL: {smiles_col}\nID_COL: {id_col}\nTITLE_COL: {title_col}\nHERB_COL: {herb_col}\n")
    f.write(f"NODE_DIM: {node_dim}\nEDGE_DIM: {edge_dim}\nDOCK_DIM: {dock_dim}\n")
    f.write("\n[COUNTS]\n")
    for k, v in counts.items():
        f.write(f"{k}: {v}\n")
    f.write("\n[Positive vote distribution]\n")
    f.write(results_df["positive_vote"].value_counts().sort_index().to_string())
    f.write("\n\n[Docking coverage summary]\n")
    f.write(results_df["docking_coverage"].describe().to_string())
    f.write("\n\n[Top 20]\n")
    f.write(results_df[show_cols].head(20).to_string(index=False))
    f.write("\n")
    if herb_out is not None:
        f.write(f"\nHERB_SIMPLE_CHECK: {herb_out}\n")

print("\n" + "=" * 80)
print(f"[OK] saved full predictions: {full_path}")
print(f"[OK] saved summary: {summary_path}")
print(f"[OK] output dir: {OUT_DIR}")
print("\n[COUNTS]")
for k, v in counts.items():
    print(f"{k}: {v}")
print("\n[TOP 20]")
print(results_df[show_cols].head(20).to_string(index=False))
