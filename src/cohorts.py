"""Cohort-materialization and matched-resampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .complexity import compute_patient_complexity_from_weekly, compute_cis_weights
from .metrics import compute_metrics_local, compute_metrics_auroc_only

def attach_pid_row_map(entry):
    pid = np.asarray(entry["pid"], dtype=int)
    pid_to_rows = {}
    for p in np.unique(pid):
        pid_to_rows[int(p)] = np.where(pid == p)[0]
    entry["_pid_to_rows"] = pid_to_rows
    return entry


def attach_row_maps(entry):
    pid = np.asarray(entry["pid"], dtype=int)
    week = np.asarray(entry["week"], dtype=int)
    row_keys = [(int(p), int(w)) for p, w in zip(pid, week)]
    rowkey_to_row = {k: i for i, k in enumerate(row_keys)}
    entry["_row_keys"] = row_keys
    entry["_rowkey_to_row"] = rowkey_to_row
    return entry


def build_sampled_patient_df(df_pool, sampled_ids):
    sampled_df = df_pool.iloc[sampled_ids].copy().reset_index(drop=True)
    if "who" in sampled_df.columns:
        sampled_df["_orig_who"] = sampled_df["who"].values
        sampled_df["who"] = np.arange(len(sampled_df), dtype=int)
    return sampled_df


def materialize_sample_from_cache(cache_entry, sampled_patient_ids):
    y_parts, p_parts, pid_new_parts = [], [], []

    for new_pid, orig_pid in enumerate(sampled_patient_ids):
        rows = cache_entry["_pid_to_rows"][int(orig_pid)]
        y_parts.append(cache_entry["y"][rows])
        p_parts.append(cache_entry["p"][rows])
        pid_new_parts.append(np.full(len(rows), new_pid, dtype=int))

    y_sub = np.concatenate(y_parts)
    p_sub = np.concatenate(p_parts)
    pid_new = np.concatenate(pid_new_parts)
    return y_sub, p_sub, pid_new


def evaluate_sampled_cohort(cache_entry, sampled_patient_ids, sampled_df, df_train_ref, pe_train_ref, gamma=1.5):
    y_sub, p_sub, pid_new = materialize_sample_from_cache(cache_entry, sampled_patient_ids)

    pe_sample = compute_patient_complexity_from_weekly(sampled_df)
    cis_sample = compute_cis_weights(sampled_df, gamma=gamma)
    w_sub = cis_sample[pid_new]

    m_plain = compute_metrics_local(y_sub, p_sub, sample_weight=None)
    m_w = compute_metrics_local(y_sub, p_sub, sample_weight=w_sub)

    return {
        "plain_auroc": m_plain["auroc"],
        "plain_auprc": m_plain["auprc"],
        "plain_brier": m_plain["brier"],
        "cis_auroc": m_w["auroc"],
        "cis_auprc": m_w["auprc"],
        "cis_brier": m_w["brier"],
        "n_weeks": m_plain["n"],
    }


def sample_patients_by_bin(df_pool, target_probs, n_patients, rng):
    bins = np.array([1, 2, 3, 4, 5], dtype=int)
    target_probs = np.asarray(target_probs, dtype=float)
    target_probs = target_probs / target_probs.sum()

    counts = rng.multinomial(n_patients, target_probs)

    sampled_ids = []
    for b, cnt in zip(bins, counts):
        ids_b = np.where(df_pool["_complexity_bin"].to_numpy() == b)[0]
        replace = cnt > len(ids_b)
        if cnt > 0:
            take = rng.choice(ids_b, size=cnt, replace=replace)
            sampled_ids.extend(take.tolist())

    sampled_ids = np.asarray(sampled_ids, dtype=int)
    rng.shuffle(sampled_ids)
    return sampled_ids


def get_observed_bin_fraction(df_sample):
    frac = (
        df_sample["_complexity_bin"]
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1,2,3,4,5], fill_value=0.0)
        .to_numpy(dtype=float)
    )
    return frac


def search_matched_cohort(
    df_pool,
    cache_entry_anchor,
    df_train_ref,
    pe_train_ref,
    target_probs,
    n_patients,
    target_plain_auroc,
    n_trials=800,
    seed=123,
    auroc_tol=0.008,
    comp_weight=0.25,
):
    rng = np.random.default_rng(seed)

    best = None
    best_obj = np.inf

    for t in range(n_trials):
        sampled_ids = sample_patients_by_bin(df_pool, target_probs, n_patients=n_patients, rng=rng)
        sampled_df = build_sampled_patient_df(df_pool, sampled_ids)

        observed_frac = get_observed_bin_fraction(sampled_df)

        m_anchor = evaluate_sampled_cohort(
            cache_entry_anchor, sampled_ids, sampled_df, df_train_ref, pe_train_ref
        )

        if np.isnan(m_anchor["plain_auroc"]):
            continue

        plain_err = abs(m_anchor["plain_auroc"] - target_plain_auroc)
        comp_err = float(np.abs(observed_frac - np.asarray(target_probs)).sum())

        obj = plain_err + comp_weight * comp_err

        if obj < best_obj:
            best_obj = obj
            best = {
                "sampled_ids": sampled_ids,
                "sampled_df": sampled_df,
                "anchor_metrics": m_anchor,
                "observed_bin_frac": observed_frac,
                "objective": obj,
                "plain_err": plain_err,
                "comp_err": comp_err,
            }

            if plain_err <= auroc_tol and comp_err <= 0.08:
                break

    return best


def materialize_prefix_sample_from_cache(cache_entry, sampled_row_keys):
    row_idx = [cache_entry["_rowkey_to_row"][rk] for rk in sampled_row_keys]
    row_idx = np.asarray(row_idx, dtype=int)
    y_sub = cache_entry["y"][row_idx]
    p_sub = cache_entry["p"][row_idx]
    w_sub = cache_entry["prefix_cis"][row_idx]
    return y_sub, p_sub, w_sub


def evaluate_prefix_sampled_cohort(cache_entry, sampled_row_keys):
    y_sub, p_sub, w_sub = materialize_prefix_sample_from_cache(cache_entry, sampled_row_keys)
    m_plain = compute_metrics_auroc_only(y_sub, p_sub, sample_weight=None)
    m_w = compute_metrics_auroc_only(y_sub, p_sub, sample_weight=w_sub)
    return {
        "plain_auroc": m_plain["auroc"],
        "cis_auroc": m_w["auroc"],
        "n_weeks": m_plain["n"],
    }


def sample_rows_by_bin(cache_entry_anchor, target_probs, n_rows, rng):
    bins = np.array([1, 2, 3, 4, 5], dtype=int)
    target_probs = np.asarray(target_probs, dtype=float)
    target_probs = target_probs / target_probs.sum()
    counts = rng.multinomial(n_rows, target_probs)
    sampled_rows = []
    prefix_bin = np.asarray(cache_entry_anchor["prefix_bin"], dtype=int)
    for b, cnt in zip(bins, counts):
        ids_b = np.where(prefix_bin == b)[0]
        replace = cnt > len(ids_b)
        if cnt > 0:
            take = rng.choice(ids_b, size=cnt, replace=replace)
            sampled_rows.extend(take.tolist())
    sampled_rows = np.asarray(sampled_rows, dtype=int)
    rng.shuffle(sampled_rows)
    return sampled_rows


def get_observed_bin_fraction_from_rows(cache_entry_anchor, sampled_rows):
    bins = np.asarray(cache_entry_anchor["prefix_bin"], dtype=int)[sampled_rows]
    frac = (
        pd.Series(bins)
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1, 2, 3, 4, 5], fill_value=0.0)
        .to_numpy(dtype=float)
    )
    return frac


def search_matched_prefix_cohort(cache_entry_anchor, target_probs, n_rows, target_plain_auroc,
                          n_trials=800, seed=123, auroc_tol=0.008, comp_weight=0.25):
    rng = np.random.default_rng(seed)
    best, best_obj = None, np.inf
    for _ in range(n_trials):
        sampled_rows = sample_rows_by_bin(cache_entry_anchor, target_probs, n_rows=n_rows, rng=rng)
        sampled_row_keys = [cache_entry_anchor["_row_keys"][i] for i in sampled_rows]
        observed_frac = get_observed_bin_fraction_from_rows(cache_entry_anchor, sampled_rows)
        m_anchor = evaluate_prefix_sampled_cohort(cache_entry_anchor, sampled_row_keys)
        if np.isnan(m_anchor["plain_auroc"]):
            continue
        plain_err = abs(m_anchor["plain_auroc"] - target_plain_auroc)
        comp_err = float(np.abs(observed_frac - np.asarray(target_probs)).sum())
        obj = plain_err + comp_weight * comp_err
        if obj < best_obj:
            best_obj = obj
            best = {
                "sampled_row_keys": sampled_row_keys,
                "observed_bin_frac": observed_frac,
                "objective": obj,
            }
            if plain_err <= auroc_tol and comp_err <= 0.08:
                break
    return best
