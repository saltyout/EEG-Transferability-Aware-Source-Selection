"""Public CPU reproduction from privately prepared frozen features."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

def fit_predict(x, y, train, test, source_weights=None, subjects=None):
    scaler = StandardScaler().fit(x[train])
    weights = np.ones(len(train))
    if source_weights is not None:
        weights = np.array([source_weights[s] for s in subjects[train]])
    model = LogisticRegression(C=0.01, max_iter=3000)
    model.fit(scaler.transform(x[train]), y[train], sample_weight=weights)
    return model.predict(scaler.transform(x[test]))

def smooth_weights(scores):
    values = np.tanh((scores - scores.mean()) / 0.10)
    return 1 + 0.45 * (values - values.mean())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = np.load(args.features, allow_pickle=False)
    x, y, subjects = data['X'], data['y'], data['subjects']
    bank = data['bank_indices']
    assert x.ndim == 2 and len(x) == len(y) == len(subjects)
    assert np.isfinite(x).all() and set(y) == {0, 1}
    assert bank.dtype.kind in 'iu' and len(np.unique(bank)) == len(bank)
    names = np.unique(subjects)
    indices = [bank[subjects[bank] == name] for name in names]
    assert len(names) >= 17 and all(len(ix) == 45 for ix in indices)
    args.output.mkdir(parents=True, exist_ok=False)
    matrix = np.full((len(names), len(names)), np.nan)
    for i, train in enumerate(indices):
        for j, test in enumerate(indices):
            if i != j:
                matrix[i, j] = balanced_accuracy_score(y[test], fit_predict(x, y, train, test))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(args.output/'transfer_matrix.csv')
    records = []
    for seed in (101, 202, 303):
        for hold, name in enumerate(names):
            pool = np.random.default_rng(seed + hold * 1000).choice(
                np.delete(np.arange(len(names)), hold), 16, replace=False)
            quality = np.nanmean(matrix[np.ix_(pool, pool)], axis=1)
            best = pool[np.argsort(-quality, kind='stable')[:4]]
            all_train = bank[np.isin(subjects[bank], names[pool])]
            test = np.flatnonzero(subjects == name)
            assert not np.intersect1d(all_train, test).size
            rng = np.random.default_rng(20261105 + seed + hold * 1000)
            uniform = []
            for k, source in enumerate(rng.permutation(pool)):
                count = 12 if k < 4 else 11
                extra = int(rng.integers(2))
                for label in (0, 1):
                    candidates = indices[source][y[indices[source]] == label]
                    uniform.extend(rng.permutation(candidates)[:count//2 + int(count%2 and label == extra)])
            arms = [('selected', bank[np.isin(subjects[bank], names[best])], None),
                    ('all180', np.array(uniform), None), ('all720', all_train, None)]
            for random_seed in (401, 502, 603):
                chosen = np.random.default_rng(random_seed + seed*10000 + hold*1000000).choice(pool, 4, replace=False)
                arms.append((f'random{random_seed}', bank[np.isin(subjects[bank], names[chosen])], None))
            source_weights = dict(zip(names[pool], smooth_weights(quality)))
            arms.append(('transfer_weighted720', all_train, source_weights))
            for arm, train, weights in arms:
                if weights is None:
                    # Equal subject contribution, including the 11/12-trial control.
                    counts = {s: np.sum(subjects[train] == s) for s in np.unique(subjects[train])}
                    weights = {s: len(train)/len(counts)/count for s, count in counts.items()}
                pred = fit_predict(x, y, train, test, weights, subjects)
                records.append(dict(target=name, pool_seed=seed, arm=arm,
                                    fit_labels=len(train), ranking_labels=720,
                                    BA=100*balanced_accuracy_score(y[test], pred)))
    pd.DataFrame(records).to_csv(args.output/'scores.csv', index=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(
        C=0.01, pool_seeds=[101,202,303], random_source_seeds=[401,502,603],
        target_used_for_fitting=False, scope='CPU source selection and full-source weighting'), indent=2))

if __name__ == '__main__':
    main()
