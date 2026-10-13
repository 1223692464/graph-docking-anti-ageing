"""Recompute FDA_Q90 and the four-rule candidate selection from saved predictions."""
import argparse
import json
import numpy as np
import pandas as pd
from repo_paths import ROOT

SEEDS = [42, 412, 719, 666, 888]
FDA_ROOT = ROOT / 'results/fda_drug_library'
CAND_ROOT = ROOT / 'results/final_candidate_selection_binding_le5'


def selection():
    reference = pd.read_csv(FDA_ROOT / 'fda_two_threshold_sets_nopeptide/set1_FDA_AB21_nopeptide_unseen.csv')
    prediction = pd.read_csv(FDA_ROOT / 'fda_model_predictions_nopeptide/set2_FDA_ABC47_qc_graph_docking/screening_qc_graph_docking_predictions.csv').copy()
    key = 'InChIKey14'
    for frame in (reference, prediction):
        if frame[key].isna().any():
            raise ValueError('Missing FDA matching key')
        frame['_merge_key'] = frame[key].astype(str).str.lower().str.strip()
    prediction = prediction.sort_values('logit_mean', ascending=False).drop_duplicates('_merge_key')
    joined = reference[['_merge_key', 'drug_name']].merge(
        prediction[['_merge_key', 'logit_mean', 'docking_coverage', 'positive_vote']],
        on='_merge_key', how='left', validate='many_to_one')
    used = joined[joined.logit_mean.notna() & (joined.docking_coverage >= 0.75)].copy()
    q90 = float(used.logit_mean.quantile(0.90, interpolation='linear'))

    candidates = pd.read_csv(ROOT / 'results/screening_qc_graph_docking_multiseed_v2/screening_qc_graph_docking_predictions.csv').copy()
    if candidates.screen_id.duplicated().any():
        raise ValueError('Duplicate candidate identifier')
    logits = candidates[[f'logit_seed{s}' for s in SEEDS]].to_numpy(dtype=float)
    # The supplied inference script uses a strict > 0 test, not >= 0.
    votes = (logits > 0.0).sum(axis=1)
    if not np.array_equal(votes, candidates.positive_vote.to_numpy()):
        raise ValueError('Vote mismatch')
    np.testing.assert_allclose(logits.mean(axis=1), candidates.logit_mean, rtol=0, atol=1e-10)
    np.testing.assert_allclose(logits.std(axis=1, ddof=1), candidates.logit_sd, rtol=0, atol=1e-10)

    energy_cols = [c for c in candidates if c.startswith('energy_')]
    energies = candidates[energy_cols]
    if len(energy_cols) != 20:
        raise ValueError('Expected 20 docking-energy columns')
    bind_count = (energies.notna() & (energies <= -5.0)).sum(axis=1).to_numpy()
    candidates['binding_target_count_le5'] = bind_count
    candidates['binding_coverage_le5'] = bind_count / 20.0
    candidates['passes_FDA_Q90'] = candidates.logit_mean >= q90
    candidates['computational_high_confidence'] = (
        (candidates.positive_vote == 5) & (candidates.logit_sd <= 10.0)
        & (bind_count == 20) & candidates.passes_FDA_Q90)
    names = pd.read_excel(ROOT / 'database/seven_candidates_source.xlsx')[['MOL ID', '化合物名称']]
    names = names.rename(columns={'MOL ID': 'screen_id', '化合物名称': 'compound_name'})
    candidates = candidates.merge(names, on='screen_id', how='left', validate='one_to_one')
    chosen = candidates[candidates.computational_high_confidence].sort_values('logit_mean', ascending=False)
    summary = dict(FDA_Q90=q90, reference_total=len(reference), reference_used=len(used),
                   candidate_count=len(candidates), passes_Q90_count=int(candidates.passes_FDA_Q90.sum()),
                   final_candidate_count=len(chosen), candidate_ids=chosen.screen_id.tolist(),
                   vote_rule='logit > 0.0', zero_logits_count=int((logits == 0).sum()), logit_sd_ddof=1)
    return candidates, chosen, used, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'verification/recomputed_selection'))
    parser.add_argument('--export-database', action='store_true')
    args = parser.parse_args()
    from pathlib import Path
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates, chosen, used, summary = selection()
    chosen.to_csv(out / 'seven_candidates.csv', index=False)
    used.to_csv(out / 'FDA_Q90_reference_records.csv', index=False)
    (out / 'selection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if args.export_database:
        core = ['screen_id', 'compound_name', 'smiles', 'canon_smiles', 'logit_mean', 'logit_sd',
                'score_mean', 'score_sd', 'positive_vote', 'valid_target_count', 'docking_coverage',
                'binding_target_count_le5', 'binding_coverage_le5', 'passes_FDA_Q90',
                'computational_high_confidence', 'rank_final']
        core += [c for c in candidates.columns if c.startswith(('energy_', 'succ_', 'logit_seed', 'score_seed', 'pred_seed'))]
        snapshot = candidates[[c for c in core if c in candidates]].copy()
        with pd.ExcelWriter(ROOT / 'database/candidate_prediction_database.xlsx', engine='openpyxl') as writer:
            snapshot.to_excel(writer, sheet_name='Candidates_4593', index=False)
            chosen[[c for c in core if c in chosen]].to_excel(writer, sheet_name='High_confidence_7', index=False)
            used.to_excel(writer, sheet_name='FDA_Q90_reference', index=False)
            for sheet in writer.book.worksheets:
                sheet.freeze_panes = 'A2'
                sheet.auto_filter.ref = sheet.dimensions
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
