from repo_paths import project_path
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# =========================
# 1. 路径
# =========================
input_xlsx = project_path('data/processed/graph_input.xlsx')
out_csv = project_path('data/processed/graph_input.csv')
out_pt = project_path('data/processed/graph_dataset.pt')

# =========================
# 2. 读取并整理方案A输入
# =========================
df = pd.read_excel(input_xlsx)

df = df[["pu_id", "set", "label", "smiles_canonical"]].copy()
df = df.rename(columns={"smiles_canonical": "smiles"})

# 去空、去重
df = df.dropna(subset=["smiles"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

# 保存最小输入表
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"[OK] saved csv: {out_csv}")
print(df["set"].value_counts())
print(df["label"].value_counts())

# =========================
# 3. 原子/键特征
# =========================
def one_hot(x, choices):
    return [int(x == c) for c in choices]

def one_hot_unknown(x, choices):
    if x not in choices:
        x = choices[-1]
    return [int(x == c) for c in choices]

# 常见元素，最后一个位置作为“其他”
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

# =========================
# 4. SMILES -> PyG graph
# =========================
def smiles_to_data(row):
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # 节点特征
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(bf)
        edge_attr_list.append(bf)

    if len(edge_index_list) == 0:
        # 单原子分子时防止报错
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 13), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([int(row["label"])], dtype=torch.float)
    )

    # 保留元信息
    data.uid = str(row["pu_id"])
    data.set_tag = str(row["set"])          # P / U
    data.smiles = smiles
    data.is_positive = 1 if row["set"] == "P" else 0

    return data

graph_list = []
bad_idx = []

for idx, row in df.iterrows():
    g = smiles_to_data(row)
    if g is None:
        bad_idx.append(idx)
    else:
        graph_list.append(g)

print(f"[OK] valid graphs: {len(graph_list)}")
print(f"[WARN] invalid smiles rows: {len(bad_idx)}")

torch.save(graph_list, out_pt)
print(f"[OK] saved pt: {out_pt}")

# =========================
# 5. 划分 train / val / test
# =========================
idx_all = list(range(len(df)))
y_all = df["label"].tolist()

train_idx, temp_idx = train_test_split(
    idx_all,
    test_size=0.30,
    random_state=42,
    stratify=y_all
)

temp_y = [y_all[i] for i in temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=temp_y
)

split_dict = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}

torch.save(split_dict, "split_idx.pt")

print("[OK] split saved: split_idx.pt")
print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

def count_pos(indices):
    return sum(df.iloc[i]["label"] == 1 for i in indices)

print(f"P in train: {count_pos(train_idx)}")
print(f"P in val:   {count_pos(val_idx)}")
print(f"P in test:  {count_pos(test_idx)}")