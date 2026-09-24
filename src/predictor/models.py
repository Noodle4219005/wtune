#!/usr/bin/env python3
"""predictor/models.py -- five models behind one interface.

Interface: `fit(train_rows)` then `score(rows)` -> list of predicted y aligned with `rows`
(lower = faster, matching targets.py's y = log(s/s_baseline)). Pairwise models additionally
expose `prob_better(row_a, row_b)` -> P(A faster than B).

catboost and lightgbm imports are optional: `available_models()` reports which of the five are
usable in this environment without raising. No deep learning, no framework beyond scikit-learn/
catboost/lightgbm as explicitly requested by the brief.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("/", 1)[0])
import features as feat  # predictor/features.py
try:
    import wtune_decomp  # repo root; used by the heuristic model when importable
except ImportError:
    wtune_decomp = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker, Pool
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

SEED = 0


def available_models():
    """{name: (True, "") | (False, reason)} -- what this environment can actually run."""
    out = {
        "heuristic": (True, ""),
        "elasticnet": (True, ""),
        "ridge": (True, ""),
        "catboost_point": (HAVE_CATBOOST, "" if HAVE_CATBOOST else "catboost import failed"),
        "catboost_pairwise": (HAVE_CATBOOST or HAVE_LIGHTGBM,
                               "" if (HAVE_CATBOOST or HAVE_LIGHTGBM)
                               else "neither catboost nor lightgbm available"),
        "knn": (True, ""),
    }
    return out


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _feature_dict(row, use_profile):
    d = {}
    d.update(feat.case_features(row))
    d.update(feat.config_features(row))
    if use_profile:
        d.update(feat.profile_features(row))
    return {k: v for k, v in d.items() if v is not None}


def _rows_to_Xy(rows, use_profile):
    dicts = [_feature_dict(r, use_profile) for r in rows]
    y = np.array([r.get("y") for r in rows], dtype=float)
    return dicts, y


class NotEnoughData(Exception):
    pass


# ---------------------------------------------------------------------------
# 1. Heuristic -- rule score from established WTune facts only, no fitting.
# ---------------------------------------------------------------------------
class HeuristicModel:
    """Every rule cites the note it came from. Output is a log-relative score (lower = faster);
    it is NOT calibrated to a case's actual baseline, so it is only meaningful for *ranking*
    configs within one case, not for regret magnitude (validate.py only uses it for ranking)."""
    name = "heuristic"

    def fit(self, train_rows):
        return self  # no free parameters to fit; rules are fixed a priori

    def _decomp_score(self, row):
        # knowledge/2026-08-26-wrf-8x56-decomposition-zero-code-win.md: patch aspect ratio
        # cache/vectorization effect, U-shaped in nproc_x with a documented sweet spot -- and
        # wtune_decomp.py's fitted shape model (ALPHA+BETA/wmin+GAMMA/hmin) is exactly that curve,
        # generalized past the one measured grid. Use it when importable; otherwise fall back to
        # a a bare patch-aspect proxy from features.py (same qualitative shape, unfitted).
        e_we, e_sn = _num(row.get("e_we")), _num(row.get("e_sn"))
        nproc_x, nproc_y = _num(row.get("nproc_x")), _num(row.get("nproc_y"))
        if not (e_we and e_sn and nproc_x and nproc_y):
            return 0.0
        if wtune_decomp is not None:
            wmax, wmin = wtune_decomp.patch_dims(int(e_we), int(nproc_x))
            hmax, hmin = wtune_decomp.patch_dims(int(e_sn), int(nproc_y))
            if wmin <= 0 or hmin <= 0:
                return 0.0
            shape = wtune_decomp.ALPHA + wtune_decomp.BETA / wmin + wtune_decomp.GAMMA / hmin
            return math.log(shape)  # lower shape score = more efficient patch geometry
        cf = feat.config_features(row)
        ratio = cf.get("decomp_aspect")
        return abs(math.log(ratio)) * 0.01 if ratio else 0.0

    def score_one(self, row):
        s = 0.0
        # even numtiles / omp_threads-divisible tiling: -1.4% on karina 224x2, bit-identical
        # [[even-numtiles-12-or-14-is-1-4-percent-faster-than-13-on-224x2-...]]
        numtiles, omp = _num(row.get("numtiles")), _num(row.get("omp_threads"))
        if numtiles and omp:
            s += -0.014 if (numtiles % omp == 0) else 0.0

        s += self._decomp_score(row)

        # physics multiplicative factors, PHYSICS-GRAY only:
        # knowledge/experiments/2026-09-07-config-cost-is-multiplicatively-separable-...md
        if row.get("compliance_class") == "PHYSICS-GRAY":
            mp_factor = {6: 0.919, 8: 1.061, 10: 1.026}.get(_num(row.get("mp_physics")) and int(row["mp_physics"]))
            bl_factor = {1: 0.940, 2: 0.949, 5: 1.121}.get(_num(row.get("bl_pbl_physics")) and int(row["bl_pbl_physics"]))
            if mp_factor:
                s += math.log(mp_factor)
            if bl_factor:
                s += math.log(bl_factor)

        # cyclic distribution: -2.2% bit-identical
        # [[cyclic-rank-placement-plus-adaptive-node-local-stealing-...]]
        if row.get("distribution") == "cyclic":
            s += -0.022

        # KMP_BLOCKTIME=0: -2.8% on hybrid WRF ABAB [[kmp-blocktime-zero-is-worth-2-8-percent...]]
        kmp = _num(row.get("kmp_blocktime"))
        if kmp == 0:
            s += -0.028

        return s

    def score(self, rows):
        return [self.score_one(r) for r in rows]


# ---------------------------------------------------------------------------
# 2/3. ElasticNet and Ridge -- standardized numeric + one-hot dict features, target y.
# ---------------------------------------------------------------------------
class LinearModel:
    def __init__(self, kind="ridge", use_profile=False):
        assert kind in ("ridge", "elasticnet")
        self.kind = kind
        self.use_profile = use_profile
        self.name = kind
        self.pipeline = None

    def fit(self, train_rows):
        dicts, y = _rows_to_Xy(train_rows, self.use_profile)
        mask = ~np.isnan(y)
        dicts = [d for d, m in zip(dicts, mask) if m]
        y = y[mask]
        if len(dicts) < 2:
            raise NotEnoughData("LinearModel needs >=2 labeled rows, got %d" % len(dicts))
        groups = np.array([r.get("case_base") for r, m in zip(train_rows, mask) if m])
        n_groups = len(set(groups))

        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(dicts)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if n_groups >= 3:
            cv = list(GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups))
            if self.kind == "ridge":
                est = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv)
            else:
                est = ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], alphas=np.logspace(-3, 1, 10),
                                    cv=cv, max_iter=5000, random_state=SEED)
        else:
            est = Ridge(alpha=1.0) if self.kind == "ridge" else ElasticNet(
                alpha=1.0, max_iter=5000, random_state=SEED)
        est.fit(X, y)

        self._vec, self._imputer, self._scaler, self._est = vec, imputer, scaler, est
        return self

    def score(self, rows):
        dicts = [_feature_dict(r, self.use_profile) for r in rows]
        X = self._vec.transform(dicts)
        X = self._imputer.transform(X)
        X = self._scaler.transform(X)
        return list(self._est.predict(X))


# ---------------------------------------------------------------------------
# 4. CatBoostRegressor -- small depth, strong l2, target y.
# ---------------------------------------------------------------------------
class CatBoostPointModel:
    name = "catboost_point"

    def __init__(self, use_profile=False):
        self.use_profile = use_profile

    def fit(self, train_rows):
        if not HAVE_CATBOOST:
            raise NotEnoughData("catboost not available")
        dicts, y = _rows_to_Xy(train_rows, self.use_profile)
        mask = ~np.isnan(y)
        dicts = [d for d, m in zip(dicts, mask) if m]
        y = y[mask]
        if len(dicts) < 2:
            raise NotEnoughData("CatBoostPointModel needs >=2 labeled rows, got %d" % len(dicts))
        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(dicts)
        model = CatBoostRegressor(depth=4, l2_leaf_reg=10, iterations=500, random_seed=SEED,
                                   loss_function="RMSE", verbose=False)
        model.fit(X, y)
        self._vec, self._model = vec, model
        return self

    def score(self, rows):
        dicts = [_feature_dict(r, self.use_profile) for r in rows]
        X = self._vec.transform(dicts)
        return list(self._model.predict(X))

    def feature_importance(self):
        return dict(zip(self._vec.get_feature_names_out(), self._model.get_feature_importance()))


# ---------------------------------------------------------------------------
# 5. Pairwise: CatBoostRanker (PairLogit, grouped by case_base) + an A-vs-B classifier on
#    config-feature differences with case features appended. LightGBM lambdarank is the allowed
#    fallback when catboost is missing.
# ---------------------------------------------------------------------------
class PairwiseModel:
    name = "catboost_pairwise"

    def __init__(self, use_profile=False):
        self.use_profile = use_profile
        self.backend = None

    def fit(self, train_rows):
        dicts, y = _rows_to_Xy(train_rows, self.use_profile)
        mask = ~np.isnan(y)
        dicts = [d for d, m in zip(dicts, mask) if m]
        y = y[mask]
        rows_kept = [r for r, m in zip(train_rows, mask) if m]
        groups = [r.get("case_base") for r in rows_kept]
        n_groups = len(set(groups))
        if n_groups < 2:
            raise NotEnoughData("pairwise ranker needs >=2 case_base groups, got %d" % n_groups)

        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(dicts)
        # CatBoostRanker/lgbm ranker want groups contiguous and sorted
        order = np.argsort(groups, kind="stable")
        X, y = X[order], y[order]
        group_ids = np.array(groups)[order]
        _, group_sizes = np.unique(group_ids, return_counts=True)

        if HAVE_CATBOOST:
            # target = -y so that "higher relevance" = "faster" for CatBoost's ranking convention
            pool = Pool(X, label=-y, group_id=group_ids)
            model = CatBoostRanker(loss_function="PairLogit", depth=4, iterations=300,
                                    random_seed=SEED, verbose=False)
            model.fit(pool)
            self.backend, self._model = "catboost", model
        elif HAVE_LIGHTGBM:
            model = lgb.LGBMRanker(objective="lambdarank", max_depth=4, n_estimators=300,
                                    random_state=SEED)
            model.fit(X, -y, group=list(group_sizes))
            self.backend, self._model = "lightgbm", model
        else:
            raise NotEnoughData("neither catboost nor lightgbm available")
        self._vec = vec
        return self

    def score(self, rows):
        dicts = [_feature_dict(r, self.use_profile) for r in rows]
        X = self._vec.transform(dicts)
        pred = self._model.predict(X)
        return list(-np.asarray(pred))  # back to lower=faster convention

    def prob_better(self, row_a, row_b):
        """P(A faster than B) from the ranker's relative scores via a logistic link -- a light
        substitute for training a separate A-vs-B classifier, sharing the same fitted scorer."""
        sa, sb = self.score([row_a])[0], self.score([row_b])[0]
        return 1.0 / (1.0 + math.exp(sa - sb))  # lower score = faster => higher prob when sa<sb


# ---------------------------------------------------------------------------
# kNN case retrieval.
# ---------------------------------------------------------------------------
class KNNCaseModel:
    name = "knn"

    def __init__(self, k=3):
        self.k = k

    def fit(self, train_rows):
        self.train_rows = list(train_rows)
        case_dicts, case_ids = [], []
        seen = set()
        for r in self.train_rows:
            cb = r.get("case_base")
            if cb in seen:
                continue
            seen.add(cb)
            case_dicts.append({k: v for k, v in feat.case_features(r).items() if v is not None})
            case_ids.append(cb)
        if not case_dicts:
            raise NotEnoughData("KNNCaseModel needs >=1 training case")
        self._case_ids = case_ids
        self._vec = DictVectorizer(sparse=False)
        Xc = self._vec.fit_transform(case_dicts)
        self._imputer = SimpleImputer(strategy="mean")
        Xc = self._imputer.fit_transform(Xc)
        self._scaler = StandardScaler()
        Xc = self._scaler.fit_transform(Xc)
        self._Xc = Xc
        self._nn = NearestNeighbors(n_neighbors=min(len(case_ids), self.k + 1))
        self._nn.fit(Xc)

        self._by_case_config = defaultdict(dict)  # case_base -> {config_key: mean_log_s(y)}
        self._by_case_ys = defaultdict(list)       # case_base -> [(config_key, y)]
        for r in self.train_rows:
            y = r.get("y")
            if y is None:
                continue
            ck = feat.config_key(r)
            self._by_case_config[r.get("case_base")].setdefault(ck, []).append(y)
            self._by_case_ys[r.get("case_base")].append((ck, y))
        for cb, d in self._by_case_config.items():
            self._by_case_config[cb] = {ck: sum(v) / len(v) for ck, v in d.items()}
        return self

    def _neighbors(self, row):
        cd = {k: v for k, v in feat.case_features(row).items() if v is not None}
        x = self._vec.transform([cd])
        x = self._imputer.transform(x)
        x = self._scaler.transform(x)
        k = min(self.k + (1 if row.get("case_base") in self._case_ids else 0), len(self._case_ids))
        dist, idx = self._nn.kneighbors(x, n_neighbors=k)
        neighbors = []
        for d, i in zip(dist[0], idx[0]):
            cb = self._case_ids[i]
            if cb == row.get("case_base"):
                continue  # never use the held-out case's own training rows (leakage guard)
            neighbors.append((cb, d))
        return neighbors[: self.k]

    def score_one(self, row):
        neighbors = self._neighbors(row)
        if not neighbors:
            return 0.0
        ck = feat.config_key(row)
        weighted, weight_sum = 0.0, 0.0
        for cb, d in neighbors:
            w = 1.0 / (d + 1e-6)
            y = self._by_case_config.get(cb, {}).get(ck)
            if y is not None:
                weighted += w * y
                weight_sum += w
        if weight_sum > 0:
            return weighted / weight_sum
        # fallback: mean y of nearest configs in config_feature space within the neighbor cases
        cfg = {k: v for k, v in feat.config_features(row).items() if v is not None}
        best, best_dist = None, None
        for cb, _ in neighbors:
            for ck2, y in self._by_case_ys.get(cb, []):
                other_row = next((r for r in self.train_rows if feat.config_key(r) == ck2
                                   and r.get("case_base") == cb), None)
                if other_row is None:
                    continue
                cfg2 = {k: v for k, v in feat.config_features(other_row).items() if v is not None}
                common = set(cfg) & set(cfg2)
                if not common:
                    continue
                dist = sum((cfg[k] - cfg2[k]) ** 2 for k in common) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist, best = dist, y
        return best if best is not None else 0.0

    def score(self, rows):
        return [self.score_one(r) for r in rows]


MODEL_FACTORIES = {
    "heuristic": lambda use_profile=False: HeuristicModel(),
    "ridge": lambda use_profile=False: LinearModel("ridge", use_profile),
    "elasticnet": lambda use_profile=False: LinearModel("elasticnet", use_profile),
    "catboost_point": lambda use_profile=False: CatBoostPointModel(use_profile),
    "catboost_pairwise": lambda use_profile=False: PairwiseModel(use_profile),
    "knn": lambda use_profile=False: KNNCaseModel(),
}
