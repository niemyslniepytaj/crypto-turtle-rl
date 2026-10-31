"""
UL reżimów rynkowych (k-means + silhouette) — M3 / R3 / praca 3.4, 5.4.

Fit wyłącznie na przeszłości względem okna eval; na eval tylko predict.
Jedna globalna etykieta (cechy uśrednione BTC+ETH), one-hot K.
K z regime_lock.json (wybór raz na T1–T6); standaryzacja cech; kanoniczna
permutacja po vol_ratio/vol_short; lookback bez zerowania warm-upu w oknie eval.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REGIME_K_MIN = 2
REGIME_K_MAX = 5
REGIME_K_DEFAULT = 3  # tylko testy / fallback gdy brak locka w unit testach
REGIME_SUBSAMPLE = 10_000
REGIME_VOL_SHORT = 60
REGIME_VOL_LONG = 1440
REGIME_RET_HORIZON = 60
# Stały seed wyboru K (nie seed linii eksperymentu)
REGIME_LOCK_SEED = 20241023

REGIME_LOCK_REL = os.path.join("shared", "files", "regime_lock.json")
REGIME_MODEL_TRAIN_REL = os.path.join("shared", "files", "regime_model_train.joblib")


@dataclass
class RegimeConfig:
    """Parametry dopasowania modelu reżimów rynkowych (k-means / silhouette)."""
    method: str = "kmeans"
    k_min: int = REGIME_K_MIN
    k_max: int = REGIME_K_MAX
    k_locked: Optional[int] = None
    seed: int = REGIME_LOCK_SEED
    subsample: int = REGIME_SUBSAMPLE
    vol_short: int = REGIME_VOL_SHORT
    vol_long: int = REGIME_VOL_LONG
    ret_horizon: int = REGIME_RET_HORIZON
    encoding: str = "onehot"


@dataclass
class RegimeFitResult:
    """Wynik fit reżimów: etykiety, K, ścieżki artefaktów."""
    k: int
    model_path: str
    profile_path: str
    fit_time_s: float
    silhouette: float
    n_samples: int
    feature_names: List[str] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)


def regime_lock_path(base_dir: str) -> str:
    return os.path.join(base_dir, REGIME_LOCK_REL)


def regime_model_train_path(base_dir: str) -> str:
    return os.path.join(base_dir, REGIME_MODEL_TRAIN_REL)


def load_regime_lock(base_dir: str) -> Dict[str, Any]:
    path = regime_lock_path(base_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Brak {path} — uruchom z_maina_funkcje/plgrid_regime_lock.py "
            f"(K i model UL na T1–T6)."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "k" not in data:
        raise ValueError(f"regime_lock.json bez pola 'k': {path}")
    return data


def require_regime_k(base_dir: str) -> int:
    return int(load_regime_lock(base_dir)["k"])


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if win <= 1 or len(x) < win:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    n = win
    s = c[n:] - c[:-n]
    s2 = c2[n:] - c2[:-n]
    var = (s2 - (s * s) / n) / max(n - 1, 1)
    out[win - 1 :] = np.sqrt(np.maximum(var, 0.0))
    return out


def features_from_closes(
    closes_by_symbol: Dict[str, np.ndarray],
    *,
    vol_short: int = REGIME_VOL_SHORT,
    vol_long: int = REGIME_VOL_LONG,
    ret_horizon: int = REGIME_RET_HORIZON,
    out_start: int = 0,
    out_end: Optional[int] = None,
    drop_warmup_for_fit: bool = False,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Cechy: ret_h, vol_ratio, vol_short (średnia po symbolach).

    Liczone na całej serii (lookback); zwracany wycinek [out_start:out_end].
    drop_warmup_for_fit: przy fit na samym początku historii odrzuć wiersze
    bez pełnego okna (bez sztucznego zerowania).
    Zwraca (X, names, warm) gdzie warm = długość warm-upu cech.
    """
    syms = sorted(closes_by_symbol.keys())
    if not syms:
        raise ValueError("brak symboli do cech reżimu")
    n = min(len(closes_by_symbol[s]) for s in syms)
    feats_per = []
    for s in syms:
        c = np.asarray(closes_by_symbol[s][:n], dtype=np.float64)
        logc = np.log(np.clip(c, 1e-12, None))
        ret = np.full(n, np.nan, dtype=np.float64)
        if ret_horizon < n:
            ret[ret_horizon:] = logc[ret_horizon:] - logc[:-ret_horizon]
        r1 = np.diff(logc, prepend=logc[0])
        vs = _rolling_std(r1, vol_short)
        vl = _rolling_std(r1, vol_long)
        ratio = np.divide(vs, np.maximum(vl, 1e-12))
        feats_per.append(np.column_stack([ret, ratio, vs]))
    stacked = np.nanmean(np.stack(feats_per, axis=0), axis=0)
    warm = max(vol_long, ret_horizon, vol_short)
    names = ["ret_h", "vol_ratio", "vol_short"]

    end = n if out_end is None else min(int(out_end), n)
    start = max(0, int(out_start))
    if start >= end:
        raise ValueError(f"pusty wycinek cech [{start}:{end}]")
    X = stacked[start:end].copy()
    if drop_warmup_for_fit and start == 0:
        # fit: odrzuć warm-up (NaN), nie zero
        valid = np.isfinite(X).all(axis=1)
        # po warm indeksy względem pełnej serii ≥ warm
        valid[: max(0, warm - start)] = False
        X = X[valid]
    else:
        # eval: lookback zapewnia skończone wartości; ewentualne NaN → nan_to_num
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float64), names, warm


def _canonical_permute(centroids: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Sortuj klastry po poziomie zmienności (vol_ratio, potem vol_short)."""
    names = list(feature_names)
    for key in ("vol_ratio", "vol_short"):
        if key in names:
            idx = names.index(key)
            scores = centroids[:, idx]
            return np.argsort(scores)
    scores = centroids[:, -1] if centroids.shape[1] else np.zeros(centroids.shape[0])
    return np.argsort(scores)


class RegimeService:
    """Fit i inferencja etykiet reżimu (UL) wyłącznie na przeszłości względem okna eval."""
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model = None
        self.k: Optional[int] = self.config.k_locked
        self.feature_names: List[str] = []
        self.permutation_: Optional[np.ndarray] = None
        self.profile_: Dict[str, Any] = {}
        self.last_fit_time_s: float = 0.0
        self.last_silhouette: float = float("nan")
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_scale_: Optional[np.ndarray] = None

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        self.scaler_mean_ = mean
        self.scaler_scale_ = scale
        return (X - mean) / scale

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean_ is None or self.scaler_scale_ is None:
            return X
        return (X - self.scaler_mean_) / self.scaler_scale_

    def select_k(
        self,
        X: np.ndarray,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[int, float]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        cfg = self.config
        if cfg.k_locked is not None:
            return int(cfg.k_locked), float("nan")
        rng = np.random.RandomState(int(seed if seed is not None else cfg.seed))
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > cfg.subsample:
            idx = rng.choice(X.shape[0], size=cfg.subsample, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        best_k, best_s = int(cfg.k_min), -1.0
        for k in range(int(cfg.k_min), int(cfg.k_max) + 1):
            if Xs.shape[0] <= k:
                continue
            km = KMeans(
                n_clusters=k,
                random_state=int(seed if seed is not None else cfg.seed),
                n_init=10,
            )
            labels = km.fit_predict(Xs)
            if len(np.unique(labels)) < 2:
                continue
            s = float(silhouette_score(Xs, labels))
            if s > best_s:
                best_s, best_k = s, k
        return int(best_k), float(best_s)

    def fit(
        self,
        X: np.ndarray,
        *,
        seed: Optional[int] = None,
        k: Optional[int] = None,
    ) -> int:
        """Dopasowuje scaler i k-means; zwraca wybrane K (silhouette lub zadanego)."""
        from sklearn.cluster import KMeans

        t0 = time.time()
        X = np.asarray(X, dtype=np.float64)
        X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] < 10:
            raise ValueError(f"za mało próbek do fit UL: {X.shape[0]}")
        Xs = self._fit_scaler(X)
        if k is None:
            k, sil = self.select_k(Xs, seed=seed)
            self.last_silhouette = sil
        else:
            k = int(k)
            self.last_silhouette = float("nan")
        km = KMeans(
            n_clusters=k,
            random_state=int(seed if seed is not None else self.config.seed),
            n_init=10,
        )
        labels = km.fit_predict(Xs)
        perm = _canonical_permute(km.cluster_centers_, self.feature_names or [])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        km.cluster_centers_ = km.cluster_centers_[perm]
        labels = inv[labels]
        self.model = km
        self.k = k
        self.permutation_ = perm
        self.profile_ = self._build_profile(Xs, labels, k)
        self.last_fit_time_s = time.time() - t0
        return k

    def _build_profile(self, X: np.ndarray, labels: np.ndarray, k: int) -> Dict[str, Any]:
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        groups = {}
        for i in range(k):
            mask = labels == i
            if not np.any(mask):
                groups[str(i)] = {"n": 0, "mean": {}}
                continue
            mean = X[mask].mean(axis=0)
            groups[str(i)] = {
                "n": int(mask.sum()),
                "mean": {names[j]: float(mean[j]) for j in range(len(names))},
            }
        return {
            "k": k,
            "groups": groups,
            "feature_names": names,
            "scaler_mean": None if self.scaler_mean_ is None else self.scaler_mean_.tolist(),
            "scaler_scale": None if self.scaler_scale_ is None else self.scaler_scale_.tolist(),
        }

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Predykcja indeksów klastrów reżimu (wymaga wcześniejszego fit)."""
        if self.model is None:
            raise RuntimeError("RegimeService: najpierw fit()")
        X = np.asarray(X, dtype=np.float64)
        Xs = self._transform(X)
        return self.model.predict(Xs).astype(np.int32)

    def predict_onehot(self, X: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """One-hot etykiet reżimu o szerokości K."""
        k = int(k or self.k or self.config.k_locked or REGIME_K_DEFAULT)
        labels = self.predict_labels(X)
        out = np.zeros((len(labels), k), dtype=np.float32)
        for i, lab in enumerate(labels):
            if 0 <= int(lab) < k:
                out[i, int(lab)] = 1.0
        return out

    def neutral_onehot(self, n: int, k: Optional[int] = None) -> np.ndarray:
        k = int(k or self.k or self.config.k_locked or REGIME_K_DEFAULT)
        return np.zeros((n, k), dtype=np.float32)

    def save(self, path: str, profile_path: Optional[str] = None) -> Tuple[str, str]:
        import joblib

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self.model,
            "k": self.k,
            "feature_names": self.feature_names,
            "permutation": None if self.permutation_ is None else self.permutation_.tolist(),
            "config": asdict(self.config),
            "silhouette": self.last_silhouette,
            "scaler_mean": None if self.scaler_mean_ is None else self.scaler_mean_.tolist(),
            "scaler_scale": None if self.scaler_scale_ is None else self.scaler_scale_.tolist(),
            "profile": self.profile_,
        }
        joblib.dump(payload, path)
        profile_path = profile_path or (os.path.splitext(path)[0] + "_profile.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(self.profile_, f, indent=2)
        return path, profile_path

    def load(self, path: str) -> None:
        import joblib

        payload = joblib.load(path)
        self.model = payload["model"]
        self.k = int(payload["k"])
        self.feature_names = list(payload.get("feature_names") or [])
        perm = payload.get("permutation")
        self.permutation_ = np.asarray(perm) if perm is not None else None
        if payload.get("config"):
            self.config = RegimeConfig(**{
                kk: vv for kk, vv in payload["config"].items()
                if kk in RegimeConfig.__dataclass_fields__
            })
        self.last_silhouette = float(payload.get("silhouette") or float("nan"))
        self.config.k_locked = self.k
        sm = payload.get("scaler_mean")
        ss = payload.get("scaler_scale")
        self.scaler_mean_ = np.asarray(sm, dtype=np.float64) if sm is not None else None
        self.scaler_scale_ = np.asarray(ss, dtype=np.float64) if ss is not None else None
        self.profile_ = dict(payload.get("profile") or {})

    def fit_from_closes(
        self,
        closes_by_symbol: Dict[str, np.ndarray],
        *,
        seed: Optional[int] = None,
        k: Optional[int] = None,
        out_start: int = 0,
        out_end: Optional[int] = None,
    ) -> int:
        X, names, _warm = features_from_closes(
            closes_by_symbol,
            vol_short=self.config.vol_short,
            vol_long=self.config.vol_long,
            ret_horizon=self.config.ret_horizon,
            out_start=out_start,
            out_end=out_end,
            drop_warmup_for_fit=True,
        )
        self.feature_names = names
        return self.fit(X, seed=seed, k=k)

    def predict_onehot_from_closes(
        self,
        closes_by_symbol: Dict[str, np.ndarray],
        *,
        out_start: int = 0,
        out_end: Optional[int] = None,
    ) -> np.ndarray:
        X, _names, _warm = features_from_closes(
            closes_by_symbol,
            vol_short=self.config.vol_short,
            vol_long=self.config.vol_long,
            ret_horizon=self.config.ret_horizon,
            out_start=out_start,
            out_end=out_end,
            drop_warmup_for_fit=False,
        )
        return self.predict_onehot(X)

    def label_stability(
        self,
        closes_a: Dict[str, np.ndarray],
        closes_b: Dict[str, np.ndarray],
        *,
        seed: int = REGIME_LOCK_SEED,
    ) -> Dict[str, Any]:
        """
        Stabilność znaczenia etykiet między dwoma refitami (ten sam K):
        korelacja profili centroidów (vol_ratio) po dopasowaniu permutacji.
        """
        k = int(self.k or self.config.k_locked or 0)
        if k <= 0:
            raise ValueError("label_stability wymaga znanego K")
        s1 = RegimeService(RegimeConfig(**{**asdict(self.config), "k_locked": k}))
        s2 = RegimeService(RegimeConfig(**{**asdict(self.config), "k_locked": k}))
        s1.fit_from_closes(closes_a, seed=seed, k=k)
        s2.fit_from_closes(closes_b, seed=seed, k=k)
        c1 = s1.model.cluster_centers_
        c2 = s2.model.cluster_centers_
        # po kanonicznej permutacji ID powinny być porównywalne
        vol_idx = (s1.feature_names or []).index("vol_ratio") if "vol_ratio" in (s1.feature_names or []) else 0
        corr = float(np.corrcoef(c1[:, vol_idx], c2[:, vol_idx])[0, 1])
        return {
            "k": k,
            "vol_ratio_centroid_corr": corr,
            "stable": bool(np.isfinite(corr) and corr > 0.5),
        }
