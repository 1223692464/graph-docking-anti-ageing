# -*- coding: utf-8 -*-
"""
Graph-only GINE multi-seed scaffold split.

Purpose
-------
This script is the multi-seed version of train_graph_scaffold.py.
It uses the same graph-only GINE architecture, the same scaffold split,
the same seed list as the Graph-Docking multi-seed scripts, and the same
evaluation metrics.

Outputs
-------
/home/zyh/aging_proj/results/graph_only_multiseed_scaffold/
    seed_xxx/
        metrics.json
        training_history.csv
        train_predictions.csv
        val_predictions.csv
        test_predictions.csv
        test_top20.csv / test_top50.csv / test_top100.csv
        best_model.pt
    multiseed_summary.csv
    multiseed_mean_std.csv
    multiseed_summary.txt
"""
from repo_paths import project_path

import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, BatchNorm

# =========================
# 0. Paths and settings
# =========================
DATA_DIR = project_path('data/processed')
SCRIPTS_DIR = project_path('scripts')
RESULTS_DIR = project_path('results')

GRAPH_PATH = os.path.join(DATA_DIR, "graph_dataset.pt")
SPLIT_PATH_CANDIDATES = [
    os.path.join(DATA_DIR, "split_idx_scaffold.pt"),
    os.path.join(SCRIPTS_DIR, "split_idx_scaffold.pt"),
]

OUT_DIR = os.path.join(RESULTS_DIR, "graph_only_multiseed_scaffold")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 412, 719, 666, 888]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters kept consistent with train_graph_scaffold.py
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL = 128


# =========================
# 1. Utility functions
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return f"{x:.4f}"


def format_metrics(m):
    return (
        f"ROC-AUC={fmt(m['roc_auc'])} | "
        f"PR-AUC={fmt(m['pr_auc'])} | "
        f"P@20={fmt(m['p_at_20'])} | "
        f"P@50={fmt(m['p_at_50'])} | "
        f"P@100={fmt(m['p_at_100'])}"
    )


# =========================
# 2. Load data
# =========================
split_path = find_first_existing(SPLIT_PATH_CANDIDATES)
if split_path is None:
    raise FileNotFoundError(f"Cannot find scaffold split file in: {SPLIT_PATH_CANDIDATES}")

if not os.path.exists(GRAPH_PATH):
    raise FileNotFoundError(f"Cannot find graph dataset: {GRAPH_PATH}")

graph_list = safe_torch_load(GRAPH_PATH)
split_dict = safe_torch_load(split_path)

base_train_dataset = [graph_list[i] for i in split_dict["train"]]
base_val_dataset   = [graph_list[i] for i in split_dict["val"]]
base_test_dataset  = [graph_list[i] for i in split_dict["test"]]

print(f"[INFO] device = {DEVICE}")
print(f"[INFO] graph path = {GRAPH_PATH}")
print(f"[INFO] scaffold split = {split_path}")
print(f"[INFO] train/val/test = {len(base_train_dataset)}/{len(base_val_dataset)}/{len(base_test_dataset)}")
print(f"[INFO] train P = {sum(int(d.y.item()) for d in base_train_dataset)}, U = {sum(int(d.y.item()) == 0 for d in base_train_dataset)}")
print(f"[INFO] val   P = {sum(int(d.y.item()) for d in base_val_dataset)}, U = {sum(int(d.y.item()) == 0 for d in base_val_dataset)}")
print(f"[INFO] test  P = {sum(int(d.y.item()) for d in base_test_dataset)}, U = {sum(int(d.y.item()) == 0 for d in base_test_dataset)}")


def get_dims(dataset):
    node_dim = dataset[0].x.size(1)
    edge_dim = None
    for d in dataset:
        if hasattr(d, "edge_attr") and d.edge_attr is not None and d.edge_attr.numel() > 0:
            edge_dim = d.edge_attr.size(1)
            break
    if edge_dim is None:
        raise ValueError("No valid edge_attr found.")
    return node_dim, edge_dim


node_dim, edge_dim = get_dims(base_train_dataset)
print(f"[INFO] node_dim = {node_dim}, edge_dim = {edge_dim}")


# =========================
# 3. Model definition
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


class GINEClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            conv, bn = make_gine_block(hidden_dim, edge_dim)
            self.convs.append(conv)
            self.bns.append(bn)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

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
        logits = self.head(graph_emb).view(-1)
        return logits, graph_emb


# =========================
# 4. Train/evaluate functions
# =========================
def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        logits, _ = model(batch)
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
    all_logits = []
    all_prob = []
    all_uid = []
    all_smiles = []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _ = model(batch)
        prob = torch.sigmoid(logits)

        y = batch.y.view(-1).detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()

        all_y.append(y)
        all_logits.append(logits_np)
        all_prob.append(prob_np)

        uid = getattr(batch, "uid", None)
        smiles = getattr(batch, "smiles", None)

        if isinstance(uid, (list, tuple)):
            all_uid.extend([str(x) for x in uid])
        else:
            all_uid.extend([None] * len(y))

        if isinstance(smiles, (list, tuple)):
            all_smiles.extend([str(x) for x in smiles])
        else:
            all_smiles.extend([None] * len(y))

    y_true = np.concatenate(all_y)
    logits = np.concatenate(all_logits)
    y_score = np.concatenate(all_prob)

    metrics = eval_metrics(y_true, y_score)

    pred_df = pd.DataFrame({
        "uid": all_uid,
        "smiles": all_smiles,
        "y_true": y_true,
        "logit": logits,
        "score": y_score
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if save_pred and seed_dir is not None:
        pred_path = os.path.join(seed_dir, f"{split_name}_predictions.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return metrics, pred_df


# =========================
# 5. Multi-seed loop
# =========================
summary_rows = []

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"[RUN] SEED = {seed}")
    set_seed(seed)

    seed_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # A generator makes DataLoader shuffling seed-controlled.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        base_train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        generator=generator
    )
    val_loader = DataLoader(
        base_val_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False
    )
    test_loader = DataLoader(
        base_test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False
    )

    model = GINEClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    y_train = np.array([int(d.y.item()) for d in base_train_dataset])
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

        train_metrics, _ = evaluate(train_loader, model, split_name="train", save_pred=False)
        val_metrics, _ = evaluate(val_loader, model, split_name="val", save_pred=False)

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
                "node_dim": int(node_dim),
                "edge_dim": int(edge_dim),
                "hidden_dim": int(HIDDEN_DIM),
                "num_layers": int(NUM_LAYERS),
                "dropout": float(DROPOUT),
            }
            torch.save(ckpt, os.path.join(seed_dir, "best_model.pt"))
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"[INFO] early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    best_ckpt = safe_torch_load(os.path.join(seed_dir, "best_model.pt"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    train_metrics, train_pred = evaluate(train_loader, model, split_name="train", save_pred=True, seed_dir=seed_dir)
    val_metrics, val_pred = evaluate(val_loader, model, split_name="val", save_pred=True, seed_dir=seed_dir)
    test_metrics, test_pred = evaluate(test_loader, model, split_name="test", save_pred=True, seed_dir=seed_dir)

    print(f"[FINAL][SEED {seed}] TRAIN : {format_metrics(train_metrics)}")
    print(f"[FINAL][SEED {seed}] VAL   : {format_metrics(val_metrics)}")
    print(f"[FINAL][SEED {seed}] TEST  : {format_metrics(test_metrics)}")

    pd.DataFrame(history).to_csv(os.path.join(seed_dir, "training_history.csv"), index=False, encoding="utf-8-sig")

    metrics_dict = {
        "seed": seed,
        "split_mode": "scaffold",
        "graph_dataset": GRAPH_PATH,
        "split_file": split_path,
        "train_samples": len(base_train_dataset),
        "val_samples": len(base_val_dataset),
        "test_samples": len(base_test_dataset),
        "train_positive_P": int(sum(int(d.y.item()) for d in base_train_dataset)),
        "val_positive_P": int(sum(int(d.y.item()) for d in base_val_dataset)),
        "test_positive_P": int(sum(int(d.y.item()) for d in base_test_dataset)),
        "train_unlabeled_U": int(sum(int(d.y.item()) == 0 for d in base_train_dataset)),
        "val_unlabeled_U": int(sum(int(d.y.item()) == 0 for d in base_val_dataset)),
        "test_unlabeled_U": int(sum(int(d.y.item()) == 0 for d in base_test_dataset)),
        "node_dim": int(node_dim),
        "edge_dim": int(edge_dim),
        "hidden_dim": int(HIDDEN_DIM),
        "num_layers": int(NUM_LAYERS),
        "dropout": float(DROPOUT),
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
# 6. Summary outputs
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
    f.write("Graph-only GINE multi-seed scaffold split\n")
    f.write("=" * 80 + "\n")
    f.write(f"GRAPH_PATH: {GRAPH_PATH}\n")
    f.write(f"SPLIT_PATH: {split_path}\n")
    f.write(f"SEEDS: {SEEDS}\n")
    f.write(f"DEVICE: {DEVICE}\n")
    f.write(f"node_dim: {node_dim}\n")
    f.write(f"edge_dim: {edge_dim}\n")
    f.write(f"hidden_dim: {HIDDEN_DIM}\n")
    f.write(f"num_layers: {NUM_LAYERS}\n")
    f.write(f"dropout: {DROPOUT}\n")
    f.write(f"lr: {LR}\n")
    f.write(f"weight_decay: {WEIGHT_DECAY}\n")
    f.write(f"max_epochs: {MAX_EPOCHS}\n")
    f.write(f"patience: {PATIENCE}\n\n")

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
