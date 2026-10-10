"""wtune_oracle_train.py -- Oracle Stage 1 (truth) + Stage 2 (surrogate).

Stage 1: build a (features -> per-rank compute) dataset from an EXISTING measured WRF
run -- no re-simulation. Truth = per-rank compute (elapsed - mpi) from mpi_per_rank.csv;
features = per-rank column features from the matching wrfout snapshot.

Stage 2: train a cost surrogate to replace Stage 0's hand-set physics coefficients.
No LightGBM/sklearn on this host -> pure-numpy ridge + a small gradient-boosted regression
tree ensemble, both k-fold cross-validated (out-of-fold Spearman = honest accuracy).

HONEST LIMIT: one snapshot = 448 samples, per-rank TOTAL (not per-phase). Proves the
method + beats Stage 0's physics coefficients; generalisation needs multi-snapshot /
standalone-kernel truth (the rigorous Stage 1, needs infra+cluster).
"""
import sys, csv
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle as oc

PX, PY = 16, 28


def build_dataset(wrfout, mpi_csv, doy, utc0, run_hours, stride=4):
    feats = oc.extract_features(wrfout, stride=stride)
    fields = oc.phase_cost_fields(feats, doy, utc0, run_hours=run_hours)
    # per-rank feature columns (16x28 -> 448 rows, WRF order r = iy*px+ix)
    cols = {}
    for k in ('radiation', 'micro', 'surf'):
        cols[k] = oc.to_ranks(fields[k], PX, PY).ravel()
    for k in ('TCWV', 'LWP', 'RWP', 'IWP', 'SWP', 'GWP', 'nspecies', 'mixed_phase'):
        if k in feats and feats[k] is not None:
            cols[k] = oc.to_ranks(np.asarray(feats[k], float), PX, PY).ravel()
    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    # truth: per-rank compute = elapsed - mpi (rank 0 = serial-I/O outlier -> neutralise)
    rows = list(csv.DictReader(open(mpi_csv)))
    comp = np.array([float(r['elapsed_s']) - float(r['mpi_s']) for r in rows])
    comp[0] = np.median(comp)
    return X, comp, names


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _standardise(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    return (X - mu) / sd, mu, sd


def ridge_fit(X, y, lam=1.0):
    Xa = np.column_stack([np.ones(len(X)), X])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1]); A[0, 0] -= lam
    w = np.linalg.solve(A, Xa.T @ y)
    return w


def ridge_pred(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


# ---- minimal numpy gradient-boosted regression trees (depth-2 CART) ----
def _best_split(X, g, min_leaf):
    best = None
    for f in range(X.shape[1]):
        xs = X[:, f]
        for t in np.unique(np.quantile(xs, np.linspace(0.1, 0.9, 9))):
            L = xs <= t
            if L.sum() < min_leaf or (~L).sum() < min_leaf:
                continue
            gl, gr = g[L].mean(), g[~L].mean()
            sse = ((g[L] - gl) ** 2).sum() + ((g[~L] - gr) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, f, t, gl, gr)
    return best


def _fit_tree(X, g, depth, min_leaf):
    if depth == 0 or len(g) < 2 * min_leaf:
        return {'leaf': float(g.mean())}
    s = _best_split(X, g, min_leaf)
    if s is None:
        return {'leaf': float(g.mean())}
    _, f, t, _, _ = s
    L = X[:, f] <= t
    return {'f': f, 't': float(t),
            'L': _fit_tree(X[L], g[L], depth - 1, min_leaf),
            'R': _fit_tree(X[~L], g[~L], depth - 1, min_leaf)}


def _pred_tree(node, X):
    if 'leaf' in node:
        return np.full(len(X), node['leaf'])
    L = X[:, node['f']] <= node['t']
    out = np.empty(len(X))
    out[L] = _pred_tree(node['L'], X[L]) if L.any() else 0
    out[~L] = _pred_tree(node['R'], X[~L]) if (~L).any() else 0
    return out


def gbrt_fit(X, y, n_est=40, lr=0.15, depth=2, min_leaf=10):
    base = float(y.mean()); f = np.full(len(y), base); trees = []
    for _ in range(n_est):
        tree = _fit_tree(X, y - f, depth, min_leaf)
        f += lr * _pred_tree(tree, X); trees.append(tree)
    return {'base': base, 'lr': lr, 'trees': trees}


def gbrt_pred(m, X):
    f = np.full(len(X), m['base'])
    for tree in m['trees']:
        f += m['lr'] * _pred_tree(tree, X)
    return f


def kfold_spearman(X, y, fit, pred, k=5, seed_perm=None):
    n = len(y)
    idx = seed_perm if seed_perm is not None else np.arange(n)  # deterministic (no RNG in sandbox)
    oof = np.zeros(n)
    for kk in range(k):
        te = idx[kk::k]; tr = np.setdiff1d(idx, te)
        m = fit(X[tr], y[tr]); oof[te] = pred(m, X[te])
    return _spearman(oof, y), oof


if __name__ == '__main__':
    WRFOUT = '/home/u5727520/wrf-bench/v4.4_bench_conus2.5km/wrfout_d01_2019-11-27_00:00:00.ifort'
    MPI = '/home/u5727520/wrf-prof/mpi_per_rank.csv'
    X, y, names = build_dataset(WRFOUT, MPI, 331, 18, 6, stride=4)
    print('dataset: X', X.shape, 'features:', names)
    Xs, mu, sd = _standardise(X)
    # Stage 0 physics baseline (for reference)
    pr = oc.predict(WRFOUT, PX, PY, 331, 18, run_hours=6, stride=4)
    print('Stage 0 physics-coeff Spearman = %.3f' % _spearman(pr['total'].ravel(), y))
    r, _ = kfold_spearman(Xs, y, lambda a, b: ridge_fit(a, b, 3.0), ridge_pred)
    print('Stage 2 RIDGE (5-fold OOF) Spearman = %.3f' % r)
    g, _ = kfold_spearman(X, y, lambda a, b: gbrt_fit(a, b), gbrt_pred)
    print('Stage 2 GBRT  (5-fold OOF) Spearman = %.3f' % g)
    # feature importance from a full-data ridge (standardised coeffs)
    w = ridge_fit(Xs, y, 3.0)
    imp = sorted(zip(names, w[1:]), key=lambda t: -abs(t[1]))
    print('ridge feature weights (standardised):', [(n, round(c, 1)) for n, c in imp])
