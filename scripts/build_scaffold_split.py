from repo_paths import project_path
# -*- coding: utf-8 -*-
import os
from collections import defaultdict

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR = project_path('data/processed')
CSV_PATH = os.path.join(DATA_DIR, "graph_input.csv")
OUT_PATH = os.path.join(DATA_DIR, "split_idx_scaffold.pt")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def make_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if scaf is None or scaf == "":
        return smiles
    return scaf

def count_pos(indices, labels):
    return sum(int(labels[i]) == 1 for i in indices)

df = pd.read_csv(CSV_PATH)
assert {"pu_id", "set", "label", "smiles"}.issubset(df.columns), "graph_input.csv 缺少必要列"

df["scaffold"] = df["smiles"].apply(make_scaffold)

scaffold_to_indices = defaultdict(list)
for i, scaf in enumerate(df["scaffold"].tolist()):
    scaffold_to_indices[scaf].append(i)

groups = []
labels = df["label"].tolist()
for scaf, idxs in scaffold_to_indices.items():
    n = len(idxs)
    p = sum(int(labels[i]) == 1 for i in idxs)
    groups.append({"scaffold": scaf, "indices": idxs, "n": n, "p": p})

# 先放“有正样本”的 scaffold，再按组大小降序
groups = sorted(groups, key=lambda g: (g["p"] > 0, g["p"], g["n"]), reverse=True)

N = len(df)
P = int(df["label"].sum())

targets = {
    "train": {"n": int(round(N * TRAIN_RATIO)), "p": int(round(P * TRAIN_RATIO))},
    "val":   {"n": int(round(N * VAL_RATIO)),   "p": int(round(P * VAL_RATIO))},
    "test":  {"n": N - int(round(N * TRAIN_RATIO)) - int(round(N * VAL_RATIO)),
              "p": P - int(round(P * TRAIN_RATIO)) - int(round(P * VAL_RATIO))}
}

splits = {"train": [], "val": [], "test": []}

def current_stats(name):
    idxs = splits[name]
    return len(idxs), count_pos(idxs, labels)

def score_after_put(name, g):
    cur_n, cur_p = current_stats(name)
    tgt_n = max(targets[name]["n"], 1)
    tgt_p = max(targets[name]["p"], 1)

    new_n = cur_n + g["n"]
    new_p = cur_p + g["p"]

    # 正样本偏差权重更大
    s = abs(new_n - tgt_n) / tgt_n + 2.5 * abs(new_p - tgt_p) / tgt_p
    # 轻度惩罚过量超标
    if new_n > tgt_n * 1.20:
        s += 0.5
    if new_p > tgt_p * 1.30:
        s += 0.8
    return s

for g in groups:
    best_name = min(["train", "val", "test"], key=lambda name: score_after_put(name, g))
    splits[best_name].extend(g["indices"])

# 如果某个 split 没有正样本，做一个简单修正
def has_pos(idxs):
    return any(int(labels[i]) == 1 for i in idxs)

for target_name in ["val", "test"]:
    if not has_pos(splits[target_name]):
        donor_candidates = []
        for donor_name in ["train", "val", "test"]:
            if donor_name == target_name:
                continue
            donor_scaffolds = defaultdict(list)
            for i in splits[donor_name]:
                donor_scaffolds[df.iloc[i]["scaffold"]].append(i)
            for scaf, idxs in donor_scaffolds.items():
                p = sum(int(labels[i]) == 1 for i in idxs)
                if p > 0:
                    donor_candidates.append((donor_name, scaf, idxs, p, len(idxs)))
        donor_candidates.sort(key=lambda x: (x[3], -x[4]), reverse=True)
        if donor_candidates:
            donor_name, scaf, idxs, p, n = donor_candidates[0]
            # 整个 scaffold 挪过去
            splits[donor_name] = [i for i in splits[donor_name] if df.iloc[i]["scaffold"] != scaf]
            splits[target_name].extend(idxs)

split_dict = {
    "train": sorted(splits["train"]),
    "val": sorted(splits["val"]),
    "test": sorted(splits["test"]),
}

torch.save(split_dict, OUT_PATH)

print(f"[OK] scaffold split saved: {OUT_PATH}")
for name in ["train", "val", "test"]:
    idxs = split_dict[name]
    n = len(idxs)
    p = count_pos(idxs, labels)
    scaffolds = len(set(df.iloc[idxs]["scaffold"].tolist()))
    print(f"{name}: n={n}, P={p}, scaffolds={scaffolds}")

print(f"total: n={len(df)}, P={int(df['label'].sum())}, scaffolds={df['scaffold'].nunique()}")
