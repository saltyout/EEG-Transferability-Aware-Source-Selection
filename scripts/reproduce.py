"""Public CPU reproduction from privately prepared frozen features."""
import argparse
import json
import hashlib
import platform
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

def fit_predict(x, y, train, test, source_weights=None, subjects=None):
    train, test = np.asarray(train), np.asarray(test)
    for ix in (train, test):
        if ix.ndim != 1 or ix.dtype.kind not in "iu" or len(ix) == 0 or np.any(ix < 0) or np.any(ix >= len(x)) or len(np.unique(ix)) != len(ix):
            raise ValueError("Train/test indices must be nonempty, unique and in bounds")
    if set(y[train]) != {0, 1}:
        raise ValueError("Training needs both classes")
    if subjects is not None and np.intersect1d(subjects[train], subjects[test]).size:
        raise ValueError("Training and evaluation subjects overlap")
    if np.intersect1d(train, test).size:
        raise ValueError('Training and evaluation trials overlap')
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

def validate_features(x, y, subjects, bank):
    if x.dtype.kind not in "fiu" or x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("X must be a finite, nonempty numeric matrix")
    if y.ndim != 1 or subjects.ndim != 1 or len(x) != len(y) or len(x) != len(subjects):
        raise ValueError("X, y and subjects must have matching row counts")
    if y.dtype.kind not in "iu" or set(y) != {0, 1}:
        raise ValueError("y must contain integer labels 0 and 1")
    if subjects.dtype.kind not in "US" or np.any(subjects == ""):
        raise ValueError("subjects must contain nonempty strings")
    if bank.ndim != 1 or bank.dtype.kind not in "iu":
        raise ValueError("bank_indices must be a one-dimensional integer array")
    if np.any(bank < 0) or np.any(bank >= len(x)) or len(np.unique(bank)) != len(bank):
        raise ValueError("bank_indices must be unique and in bounds")
    if len(np.unique(subjects)) < 17:
        raise ValueError("At least 17 subjects are required")
    for subject in np.unique(subjects):
        labels = y[bank[subjects[bank] == subject]]
        if len(labels) != 45 or min(np.sum(labels == 0), np.sum(labels == 1)) < 6:
            raise ValueError("Each subject bank needs 45 trials and at least six per class")

def rank_sources(matrix, pool):
    matrix = np.asarray(matrix, dtype=float)
    pool = np.asarray(pool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or pool.ndim != 1 or pool.dtype.kind not in "iu" or len(pool) != 16 or len(np.unique(pool)) != 16 or np.any(pool < 0) or np.any(pool >= len(matrix)):
        raise ValueError("Expected a square matrix and 16 distinct valid source indices")
    block = matrix[np.ix_(pool, pool)].copy()
    np.fill_diagonal(block, np.nan)
    if not np.isfinite(block[~np.eye(16, dtype=bool)]).all():
        raise ValueError("Missing off-diagonal transfer scores")
    quality = np.nanmean(block, axis=1)
    if not np.isfinite(quality).all():
        raise ValueError("Source pool has invalid transfer scores")
    return quality, pool[np.argsort(-quality, kind="stable")[:4]]

def annotation_budget(arm, fit_labels):
    ranking = 720 if arm in ("selected", "transfer_weighted720") else 0
    return ranking, max(ranking, fit_labels)

def transfer_matrix(x, y, indices):
    """Each classifier/scaler fits one source; recipients are scored separately.

    The full matrix is a reusable cache. Outer folds consume only pool x pool.
    """
    matrix = np.full((len(indices), len(indices)), np.nan)
    for i, train in enumerate(indices):
        recipients = [j for j in range(len(indices)) if j != i]
        test = np.concatenate([indices[j] for j in recipients])
        predictions = fit_predict(x, y, train, test)
        offset = 0
        for j in recipients:
            count = len(indices[j])
            matrix[i, j] = balanced_accuracy_score(y[indices[j]], predictions[offset:offset+count])
            offset += count
    return matrix

def summarize_scores(records):
    frame = pd.DataFrame(records).copy()
    frame["arm"] = frame.arm.where(~frame.arm.str.startswith("random"), "random")
    table = frame.groupby(["target", "pool_seed", "arm"]).BA.mean().unstack()
    summary = {"means": table.groupby("target").mean().mean().to_dict(), "contrasts": {}}
    for arm in ("random", "all180", "all720"):
        delta = table["selected"] - table[arm]
        per_target = delta.groupby("target").mean()
        rng = np.random.default_rng(20261105)
        boot = rng.choice(per_target.to_numpy(), (20000, len(per_target)), replace=True).mean(1)
        summary["contrasts"]["selected-"+arm] = dict(mean_pp=float(per_target.mean()),
            bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist(),
            by_pool=delta.groupby("pool_seed").mean().to_dict(),
            by_target=per_target.to_dict())
    summary["interpretation"] = "Descriptive paired-target bootstrap; overlapping pools and fixed trial banks"
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = np.load(args.features, allow_pickle=False)
    x, y, subjects = data['X'], data['y'], data['subjects']
    bank = data['bank_indices']
    validate_features(x, y, subjects, bank)
    names = np.unique(subjects)
    indices = [bank[subjects[bank] == name] for name in names]
    args.output.mkdir(parents=True, exist_ok=False)
    matrix = transfer_matrix(x, y, indices)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(args.output/'transfer_matrix.csv')
    records = []
    manifests = []
    for seed in (101, 202, 303):
        for hold, name in enumerate(names):
            pool = np.random.default_rng(seed + hold * 1000).choice(
                np.delete(np.arange(len(names)), hold), 16, replace=False)
            quality, best = rank_sources(matrix, pool)
            all_train = bank[np.isin(subjects[bank], names[pool])]
            test = np.flatnonzero(subjects == name)
            if np.intersect1d(all_train, test).size:
                raise ValueError("Target is present in source pool")
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
                manifests.append(dict(target=name,pool_seed=seed,arm=arm,pool=names[pool].tolist(),source_quality=quality.tolist(),fit_indices=train.tolist(),fit_sources=np.unique(subjects[train]).tolist()))
                pred = fit_predict(x, y, train, test, weights, subjects)
                ranking, total = annotation_budget(arm, len(train))
                records.append(dict(target=name, pool_seed=seed, arm=arm,
                                    fit_labels=len(train), ranking_labels=ranking, total_unique_labels=total,
                                    BA=100*balanced_accuracy_score(y[test], pred)))
    pd.DataFrame(records).to_csv(args.output/'scores.csv', index=False)
    (args.output/'summary.json').write_text(json.dumps(summarize_scores(records), indent=2))
    (args.output/'fit_manifest.json').write_text(json.dumps(manifests, indent=2))
    (args.output/'protocol.json').write_text(json.dumps(dict(
        input_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
        python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ("numpy","scipy","pandas","scikit-learn")},
        C=0.01, pool_seeds=[101,202,303], random_source_seeds=[401,502,603],
        target_used_for_fitting=False, scope='CPU source selection and full-source weighting'), indent=2))

if __name__ == '__main__':
    main()
