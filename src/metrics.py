"""Metric helpers for plain and CIS-weighted benchmark evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def weighted_brier(y_true, y_prob, w=None):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if w is None:
        return float(np.mean((y_prob - y_true) ** 2))
    w = np.asarray(w).astype(float)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w * (y_prob - y_true) ** 2))


def compute_metrics_binary(y_true, y_prob, sample_weight=None):
    """Return AUROC, AUPRC, and Brier score for binary classification."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    auroc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
    auprc = average_precision_score(y_true, y_prob, sample_weight=sample_weight)
    brier = weighted_brier(y_true, y_prob, w=sample_weight)
    return {"auroc": float(auroc), "auprc": float(auprc), "brier": float(brier)}


def cis_weighted_aggregate_by_tier(tier_metrics, tier_weights):
    """Weighted aggregate across tiers using tier_weights (e.g., sum CIS in tier)."""
    metrics = set()
    for v in tier_metrics.values():
        metrics |= set(v.keys())
    total_w = sum(float(tier_weights.get(t, 0.0)) for t in tier_metrics.keys()) + 1e-12
    out = {}
    for m in metrics:
        num = 0.0
        for t in tier_metrics.keys():
            w = float(tier_weights.get(t, 0.0))
            if m in tier_metrics[t]:
                num += w * float(tier_metrics[t][m])
        out[m] = num / total_w
    return out


def _safe_auroc(y, p, w=None):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    if w is not None:
        w = np.asarray(w).astype(float).ravel()
    # roc_auc_score fails if only one class present
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def _safe_auprc(y, p, w=None):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    if w is not None:
        w = np.asarray(w).astype(float).ravel()
    return average_precision_score(y, p, sample_weight=w)


def _weighted_brier(y, p, w=None):
    y = np.asarray(y).astype(float).ravel()
    p = np.asarray(p).astype(float).ravel()
    if w is None:
        return float(np.mean((p - y) ** 2))
    w = np.asarray(w).astype(float).ravel()
    return float(np.average((p - y) ** 2, weights=w))


def _compute_metrics(y, p, w=None):
    """
    Uses your global compute_basic_metrics if available; otherwise uses safe local metrics.
    Always returns dict with keys: auroc, auprc, brier.
    """
    if "compute_basic_metrics" in globals():
        try:
            m = compute_basic_metrics(y, p, sample_weight=w)
            # normalize keys / fill missing
            out = {
                "auroc": m.get("auroc", _safe_auroc(y, p, w)),
                "auprc": m.get("auprc", _safe_auprc(y, p, w)),
                "brier": m.get("brier", _weighted_brier(y, p, w)),
            }
            return out
        except TypeError:
            # compute_basic_metrics exists but doesn't accept sample_weight
            pass
    return {
        "auroc": _safe_auroc(y, p, w),
        "auprc": _safe_auprc(y, p, w),
        "brier": _weighted_brier(y, p, w),
    }


def weighted_brier_local(y_true, y_prob, w=None):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if w is None:
        return float(np.mean((y_prob - y_true) ** 2))
    w = np.asarray(w).astype(float)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w * (y_prob - y_true) ** 2))


def compute_metrics_local(y_true, y_prob, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "n": int(len(y_true))}

    return {
        "auroc": float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight)),
        "auprc": float(average_precision_score(y_true, y_prob, sample_weight=sample_weight)),
        "brier": float(weighted_brier_local(y_true, y_prob, w=sample_weight)),
        "n": int(len(y_true)),
    }


def compute_metrics_auroc_only(y_true, y_prob, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "n": int(len(y_true))}
    return {"auroc": float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight)),
            "n": int(len(y_true))}


def fit_eval_tabular(name, clf, X_tr, y_tr, X_te, y_te, pid_te, tier_test, cis_patient=None):
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]

    # unweighted + weighted
    m_unw = _compute_metrics(y_te, prob, w=None)
    m_w = None
    if cis_patient is not None:
        w = cis_patient[pid_te.astype(int)]
        m_w = _compute_metrics(y_te, prob, w=w)

    # tier AUROC (unweighted; keep fast)
    by_tier = None
    if tier_test is not None:
        rows = []
        tiers_sorted = sorted(np.unique(tier_test))
        for t in tiers_sorted:
            mask = (tier_test[pid_te.astype(int)] == t)
            if mask.sum() < 100:
                continue
            rows.append({
                "tier": int(t),
                "auroc": _safe_auroc(y_te[mask], prob[mask], w=None),
                "n_weeks": int(mask.sum())
            })
        by_tier = pd.DataFrame(rows)

    return m_unw, m_w, by_tier
