"""Check saved-output arithmetic and identifiers without PyTorch or RDKit."""
import argparse
import ast
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from repo_paths import ROOT
from reproduce_selection import selection

MODELS = {'Docking-only': 'docking_mlp_qc_all20_multiseed_scaffold',
          'Graph-only': 'graph_only_multiseed_scaffold',
          'Graph-Docking Fusion': 'graph_docking_qc_all20_multiseed_scaffold'}
SEEDS = [42, 412, 719, 666, 888]


def classification(y, s, threshold):
    p = s >= threshold
    tp, tn = int(((y == 1) & p).sum()), int(((y == 0) & ~p).sum())
    fp, fn = int(((y == 0) & p).sum()), int(((y == 1) & ~p).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denom) if denom else 0.0
    return dict(f1=f1, mcc=mcc, balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                tp=tp, tn=tn, fp=fp, fn=fn)


def auc_ap(y, s):
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    ranks = pd.Series(s).rank(method='average').to_numpy()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-s, kind='stable')
    ys, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(s) - 1]
    tp = np.cumsum(ys)[ends]
    ap = np.sum(np.diff(np.r_[0, tp / n_pos]) * tp / (ends + 1))
    return float(auc), float(ap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'verification/release_checks.json'))
    args = parser.parse_args()
    parsed = []
    for folder in ['scripts']:
        for path in sorted((ROOT / folder).glob('*.py')):
            ast.parse(path.read_text(encoding='utf-8-sig'))
            parsed.append(path.relative_to(ROOT).as_posix())
    inputs = pd.read_csv(ROOT / 'data/processed/graph_input.csv').set_index('pu_id')
    expected = pd.read_csv(ROOT / 'results/three_model_test_diagnostics/three_model_test_metrics_by_seed.csv')
    records, sample_counts = [], []
    baseline_test = None
    for model, folder in MODELS.items():
        for seed in SEEDS:
            base = ROOT / 'results' / folder / f'seed_{seed}'
            metric = json.loads((base / 'metrics.json').read_text(encoding='utf-8'))
            frames = {}
            for split in ['train', 'val', 'test']:
                frame = pd.read_csv(base / (split + '_predictions.csv'))
                id_col = 'uid' if 'uid' in frame else 'pu_id'
                label_col = 'y_true' if 'y_true' in frame else 'label'
                ids = frame[id_col].astype(str)
                if ids.duplicated().any():
                    raise ValueError('Duplicate prediction identifier')
                np.testing.assert_array_equal(inputs.loc[ids, 'label'].to_numpy(), frame[label_col].to_numpy())
                sample_counts.append(dict(model=model, seed=seed, split=split, n=len(frame), P=int(frame[label_col].sum())))
                frames[split] = frame
                if split == 'test':
                    if baseline_test is None:
                        baseline_test = set(ids)
                    if baseline_test != set(ids):
                        raise ValueError('Test-set identifier mismatch')
            test, val = frames['test'], frames['val']
            label_col = 'y_true' if 'y_true' in test else 'label'
            y, s = test[label_col].to_numpy(dtype=int), test.score.to_numpy(dtype=float)
            vy, vs = val[label_col].to_numpy(dtype=int), val.score.to_numpy(dtype=float)
            vals = np.unique(vs)
            if len(vals) > 1000:
                vals = np.unique(np.quantile(vals, np.linspace(0, 1, 1001)))
            threshold = max(vals, key=lambda t: classification(vy, vs, t)['f1'])
            actual = classification(y, s, threshold)
            auc, ap = auc_ap(y, s)
            actual.update(roc_auc=auc, pr_auc=ap, threshold=float(threshold))
            ref = expected[(expected.model == model) & (expected.seed == seed)].iloc[0]
            for name, value in actual.items():
                if not np.isclose(value, ref[name], rtol=0, atol=1e-6):
                    raise ValueError(f'{model} seed {seed}: {name} differs from saved diagnostic')
            for name, value in [('roc_auc', auc), ('pr_auc', ap)]:
                if not np.isclose(value, metric['test_metrics'][name], rtol=0, atol=1e-6):
                    raise ValueError('Recorded test metric mismatch')
            order = np.argsort(-s)
            for k in [20, 50]:
                value = float(y[order[:k]].mean())
                if not np.isclose(value, ref[f'p_at_{k}'], rtol=0, atol=1e-6):
                    raise ValueError('Early-enrichment mismatch')
                actual[f'p_at_{k}'] = value
            records.append(dict(model=model, seed=seed, **actual))
    _, chosen, _, selected = selection()
    expected_ids = {'MOL011233', 'MOL009635', 'MOL011802', 'MOL000298', 'MOL003847', 'MOL011290', 'MOL011160'}
    if len(inputs) != 5214 or len(baseline_test) != 78:
        raise ValueError('Unexpected input/test size')
    if selected['candidate_count'] != 4593 or selected['reference_used'] != 20 or selected['passes_Q90_count'] != 941:
        raise ValueError('Unexpected screening/reference size')
    if set(chosen.screen_id) != expected_ids or not math.isclose(selected['FDA_Q90'], 24.62142791748047, abs_tol=1e-10):
        raise ValueError('Unexpected final selection')
    report = dict(status='PASS', verification_scope='Saved-output checks only; training and inference not executed',
                  python_scripts_parsed=parsed, input_rows=len(inputs), input_P=int(inputs.label.sum()),
                  input_U=int((inputs.label == 0).sum()), common_test_rows=len(baseline_test),
                  prediction_files_checked=len(sample_counts), model_metrics_checked=records,
                  sample_counts=sample_counts, selection=selected)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: {} scripts, {} prediction tables, 15 test evaluations, FDA_Q90 and 7 candidates'.format(len(parsed), len(sample_counts)))


if __name__ == '__main__':
    main()
