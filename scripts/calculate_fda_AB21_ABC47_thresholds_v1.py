# -*- coding: utf-8 -*-
"""
Calculate FDA thresholds for AB21 and ABC47 no-peptide sets from ABC47 prediction file.

特点
----
1. 输入只需要 ABC47 的预测文件，因为 AB21 是 ABC47 的子集；
2. 按 InChIKey14 / drug_name 自动合并 AB21 和 ABC47；
3. 同时计算：
   - all_qc_pass：所有阳性/准阳性药物，只要求 docking coverage 通过
   - vote5_qc_pass：positive_vote=5 且 docking coverage 通过
4. 同时给出 cov12 与 cov15 两套结果；
5. 推荐主阈值：cov15 + all_qc_pass 的 Q50；
   vote5 版本只作敏感性参考，不能作为主阈值唯一依据。

Run
---
python3 calculate_fda_AB21_ABC47_thresholds_v1.py

可指定预测文件：
python3 calculate_fda_AB21_ABC47_thresholds_v1.py \
  --pred /home/zyh/aging_proj/results/fda_drug_library/fda_model_predictions_nopeptide/set2_FDA_ABC47_qc_graph_docking/screening_qc_graph_docking_predictions.csv
"""
from repo_paths import project_path

import os
import argparse
import pandas as pd
import numpy as np

BASE = project_path('results/fda_drug_library')
DEFAULT_PRED = os.path.join(
    BASE,
    "fda_model_predictions_nopeptide",
    "set2_FDA_ABC47_qc_graph_docking",
    "screening_qc_graph_docking_predictions.csv"
)
SET1 = os.path.join(BASE, "fda_two_threshold_sets_nopeptide", "set1_FDA_AB21_nopeptide_unseen.csv")
SET2 = os.path.join(BASE, "fda_two_threshold_sets_nopeptide", "set2_FDA_ABC47_nopeptide_unseen.csv")
OUT_DIR = os.path.join(BASE, "fda_thresholds_nopeptide")
os.makedirs(OUT_DIR, exist_ok=True)


def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def find_key(a, b):
    for key in ["InChIKey14", "InChIKey_rdkit", "drug_name", "INN", "canonical_smiles_rdkit"]:
        if key in a.columns and key in b.columns:
            left = set(a[key].dropna().astype(str).str.lower())
            right = set(b[key].dropna().astype(str).str.lower())
            if len(left & right) > 0:
                return key, len(left & right)
    raise KeyError("No merge key found. Need InChIKey14 or drug_name in both set and prediction files.")


def attach_pred(set_df, pred_df, set_name):
    key, overlap = find_key(set_df, pred_df)
    s = set_df.copy()
    p = pred_df.copy()
    s["_merge_key"] = s[key].astype(str).str.lower().str.strip()
    p["_merge_key"] = p[key].astype(str).str.lower().str.strip()

    pred_cols = ["_merge_key"]
    for c in [
        "screen_id", "fda_screen_id_old", "compound_title",
        "rank_final", "logit_mean", "logit_sd", "score_mean", "score_sd",
        "positive_vote", "valid_target_count", "docking_coverage",
        "qc_pass_cov12", "qc_pass_cov15"
    ]:
        if c in p.columns and c not in pred_cols:
            pred_cols.append(c)

    # deduplicate predictions
    p["logit_mean"] = pd.to_numeric(p["logit_mean"], errors="coerce")
    p = p.sort_values("logit_mean", ascending=False).drop_duplicates("_merge_key", keep="first")

    m = s.merge(p[pred_cols], on="_merge_key", how="left", suffixes=("", "_pred"))
    m["threshold_set"] = set_name
    m["merge_key_used"] = key
    m["merge_key_overlap"] = overlap

    # Numeric
    for c in ["logit_mean", "logit_sd", "score_mean", "score_sd", "positive_vote", "valid_target_count", "docking_coverage", "rank_final"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


def threshold_stats(df, set_name, coverage_cutoff, subset_name, vote5_only=False):
    x = df.copy()
    x["has_prediction"] = x["logit_mean"].notna()
    x["coverage_pass"] = x["docking_coverage"] >= coverage_cutoff
    if vote5_only:
        x["subset_pass"] = x["has_prediction"] & x["coverage_pass"] & (x["positive_vote"] == 5)
    else:
        x["subset_pass"] = x["has_prediction"] & x["coverage_pass"]

    use = x[x["subset_pass"]].copy()
    if len(use) == 0:
        return dict(
            threshold_set=set_name, subset=subset_name, coverage_cutoff=coverage_cutoff,
            n_set=len(x), n_with_prediction=int(x["has_prediction"].sum()), n_used=0,
            Q20=np.nan, Q50=np.nan, Q75=np.nan, Q90=np.nan, mean=np.nan, min=np.nan, max=np.nan,
            positive_vote5_used=0
        )

    return dict(
        threshold_set=set_name,
        subset=subset_name,
        coverage_cutoff=coverage_cutoff,
        n_set=len(x),
        n_with_prediction=int(x["has_prediction"].sum()),
        n_used=len(use),
        Q20=float(use["logit_mean"].quantile(0.20)),
        Q50=float(use["logit_mean"].quantile(0.50)),
        Q75=float(use["logit_mean"].quantile(0.75)),
        Q90=float(use["logit_mean"].quantile(0.90)),
        mean=float(use["logit_mean"].mean()),
        min=float(use["logit_mean"].min()),
        max=float(use["logit_mean"].max()),
        positive_vote5_used=int((use["positive_vote"] == 5).sum()) if "positive_vote" in use.columns else np.nan
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--set1", default=SET1)
    ap.add_argument("--set2", default=SET2)
    args = ap.parse_args()

    pred = read_csv(args.pred)
    set1 = read_csv(args.set1)
    set2 = read_csv(args.set2)

    m1 = attach_pred(set1, pred, "FDA_AB21_nopeptide")
    m2 = attach_pred(set2, pred, "FDA_ABC47_nopeptide")

    rows = []
    for set_name, m in [("FDA_AB21_nopeptide", m1), ("FDA_ABC47_nopeptide", m2)]:
        for cov in [0.60, 0.75]:
            rows.append(threshold_stats(m, set_name, cov, f"all_qc_cov{int(cov*20)}", vote5_only=False))
            rows.append(threshold_stats(m, set_name, cov, f"vote5_qc_cov{int(cov*20)}", vote5_only=True))

    th = pd.DataFrame(rows)

    m1_path = os.path.join(OUT_DIR, "FDA_AB21_predictions_matched.csv")
    m2_path = os.path.join(OUT_DIR, "FDA_ABC47_predictions_matched.csv")
    th_path = os.path.join(OUT_DIR, "FDA_AB21_ABC47_thresholds_summary.csv")
    txt_path = os.path.join(OUT_DIR, "FDA_AB21_ABC47_thresholds_summary.txt")

    m1.to_csv(m1_path, index=False, encoding="utf-8-sig")
    m2.to_csv(m2_path, index=False, encoding="utf-8-sig")
    th.to_csv(th_path, index=False, encoding="utf-8-sig")

    # recommended rows
    rec = th[(th["subset"].isin(["all_qc_cov15", "all_qc_cov12"]))].copy()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("FDA AB21 vs ABC47 no-peptide threshold summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Prediction file: {args.pred}\n")
        f.write(f"Set1 file: {args.set1}\n")
        f.write(f"Set2 file: {args.set2}\n\n")
        f.write("[All thresholds]\n")
        f.write(th.to_string(index=False))
        f.write("\n\n[Recommended interpretation]\n")
        f.write("Primary: all_qc_cov15 Q50 as FDA_mid threshold.\n")
        f.write("Sensitivity: all_qc_cov12 Q50 and vote5_qc_cov15 Q50.\n")
        f.write("Do not use score_mean for thresholding because probability is saturated near 1.\n")

    print("\n[DONE]")
    print(f"[OK] AB21 matched predictions: {m1_path}")
    print(f"[OK] ABC47 matched predictions: {m2_path}")
    print(f"[OK] threshold CSV: {th_path}")
    print(f"[OK] threshold TXT: {txt_path}")

    print("\n[Threshold summary]")
    print(th.to_string(index=False))

    print("\n[Recommended rows: all_qc_cov15]")
    print(th[th["subset"] == "all_qc_cov15"].to_string(index=False))


if __name__ == "__main__":
    main()
