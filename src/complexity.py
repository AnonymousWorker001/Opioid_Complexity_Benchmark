"""Behavioral-complexity and CIS weighting utilities."""

from __future__ import annotations

import math
from collections import Counter
import numpy as np
import pandas as pd

def calculate_permutation_entropy(time_series, D=3):
    """
    Calculate normalized permutation entropy for a categorical time series.
    """
    n = len(time_series)
    if n < D:
        raise ValueError("Time-series length must be greater than embedding dimension.")

    ordinal_patterns = []
    for i in range(n - D + 1):
        segment = time_series[i:i + D]
        ordinal_pattern = tuple(np.argsort(segment))
        ordinal_patterns.append(ordinal_pattern)

    pattern_counts = Counter(ordinal_patterns)
    total_patterns = len(ordinal_patterns)
    probabilities = np.array(list(pattern_counts.values())) / total_patterns

    permutation_entropy = -np.sum(probabilities * np.log(probabilities))
    max_entropy = np.log(math.factorial(D))
    return permutation_entropy / max_entropy


def hamming_distance(ts1, ts2):
    assert len(ts1) == len(ts2), "Time-series must have the same length"
    return np.sum(ts1 != ts2) / len(ts1)


def compute_patient_complexity_from_weekly(df, opioid_prefix="Opioid_week", n_timesteps=24, miss_value=2, D=3):
    """Compute per-patient permutation entropy (PE) from the opioid weekly trajectory.
    Uses calculate_permutation_entropy(...) defined above.
    """
    cols = [f"{opioid_prefix}{i}" for i in range(n_timesteps)]
    if cols[0] not in df.columns:
        cols = [c for c in df.columns if c.startswith(opioid_prefix)][:n_timesteps]
    x = df[cols].values
    pe = np.zeros(df.shape[0], dtype=float)
    for i in range(df.shape[0]):
        pe[i] = calculate_permutation_entropy(x[i, :], D=D)
    return pe


def compute_cis_weights(df_any, gamma=1.5, D=3):
    """Compute CIS weights using cohort heterogeneity from training reference set.
    CIS_j = PE_j ** gamma.
    """
    pe_any = compute_patient_complexity_from_weekly(df_any, D=D)
    pe_any = np.clip(pe_any,0.0, 1.0)
    cis_any = pe_any ** gamma
    return cis_any


def get_prediction_weeks(n_timesteps=24, future_window=1, start_week=3):
    return np.arange(start_week, n_timesteps - future_window, dtype=int)


def compute_prefix_complexity_matrix_from_weekly(df, opioid_prefix="Opioid_week", n_timesteps=24, D=3):
    """
    Compute normalized permutation entropy for every prefix x_{0:t} of the weekly opioid sequence.
    Returns an array of shape (N, n_timesteps), with NaN for prefixes shorter than D.
    """
    cols = [f"{opioid_prefix}{i}" for i in range(n_timesteps)]
    if cols[0] not in df.columns:
        cols = [c for c in df.columns if c.startswith(opioid_prefix)][:n_timesteps]
    x = df[cols].to_numpy(dtype=float)
    pe = np.full((df.shape[0], n_timesteps), np.nan, dtype=float)
    min_t = D - 1
    for i in range(df.shape[0]):
        for t in range(min_t, n_timesteps):
            pe[i, t] = calculate_permutation_entropy(x[i, :t+1], D=D)
    return pe


def compute_prefix_patient_complexity_from_weekly(df, opioid_prefix="Opioid_week", n_timesteps=24,
                                           start_week=3, future_window=1, summary="mean"):
    """
    Patient-level prefix-complexity summary used for benchmark split construction and patient-level cohort bins.
    summary='mean' averages prefix PE over all valid prediction weeks t.
    """
    pe_mat = compute_prefix_complexity_matrix_from_weekly(
        df, opioid_prefix=opioid_prefix, n_timesteps=n_timesteps, D=3
    )
    valid_weeks = get_prediction_weeks(n_timesteps=n_timesteps, future_window=future_window, start_week=start_week)
    vals = pe_mat[:, valid_weeks]
    if summary == "mean":
        out = np.nanmean(vals, axis=1)
    elif summary == "median":
        out = np.nanmedian(vals, axis=1)
    elif summary == "last":
        out = pe_mat[:, valid_weeks[-1]]
    else:
        raise ValueError("summary must be one of {'mean','median','last'}")
    return np.asarray(out, dtype=float)


def flatten_prefix_complexities_for_reference(df, opioid_prefix="Opioid_week", n_timesteps=24,
                                              start_week=3, future_window=1):
    """
    Flatten all valid prediction-time prefix PE values from a dataframe into one 1D reference vector.
    """
    pe_mat = compute_prefix_complexity_matrix_from_weekly(
        df, opioid_prefix=opioid_prefix, n_timesteps=n_timesteps, D=3
    )
    valid_weeks = get_prediction_weeks(n_timesteps=n_timesteps, future_window=future_window, start_week=start_week)
    vals = pe_mat[:, valid_weeks].reshape(-1)
    vals = vals[np.isfinite(vals)]
    return np.asarray(vals, dtype=float)


def compute_sample_prefix_pe(df_eval, sample_idx, week_idx, opioid_prefix="Opioid_week", n_timesteps=24,
                             prefix_matrix_eval=None):
    if prefix_matrix_eval is None:
        prefix_matrix_eval = compute_prefix_complexity_matrix_from_weekly(
            df_eval, opioid_prefix=opioid_prefix, n_timesteps=n_timesteps, D=3
        )
    sample_idx = np.asarray(sample_idx, dtype=int)
    week_idx = np.asarray(week_idx, dtype=int)
    return prefix_matrix_eval[sample_idx, week_idx]


def compute_prefix_patient_cis_weights(df_train_ref, df_any, pe_train=None, pe_any=None):
    """
    Backward-compatible patient-level CIS weights using patient-level prefix-complexity summaries.
    """
    if pe_train is None:
        pe_train = compute_prefix_patient_complexity_from_weekly(df_train_ref)
    if pe_any is None:
        pe_any = compute_prefix_patient_complexity_from_weekly(df_any)
    U = 2.0 * float(np.std(pe_train))
    cis_any = pe_any * U
    return cis_any, U


def compute_sample_prefix_cis_weights(df_train_ref, df_eval, sample_idx, week_idx,
                                       train_prefix_values=None, prefix_matrix_eval=None,
                                       opioid_prefix="Opioid_week", n_timesteps=24,
                                       start_week=3, future_window=1, gamma=1.5):
    """
    Prefix-based complexity weights at the patient-week sample level.
    The weight for sample i is defined as CIS_i = (PE_i)^gamma.
    """
    if prefix_matrix_eval is None:
        prefix_matrix_eval = compute_prefix_complexity_matrix_from_weekly(
            df_eval, opioid_prefix=opioid_prefix, n_timesteps=n_timesteps, D=3
        )

    prefix_pe = compute_sample_prefix_pe(
        df_eval, sample_idx, week_idx,
        opioid_prefix=opioid_prefix, n_timesteps=n_timesteps,
        prefix_matrix_eval=prefix_matrix_eval
    )

    prefix_pe = np.clip(np.asarray(prefix_pe, dtype=float), 0.0, 1.0)
    cis = prefix_pe ** gamma

    return np.asarray(cis, dtype=float), np.asarray(prefix_pe, dtype=float)


def assign_bins_from_cutpoints(values, cutpoints):
    bins = [-np.inf, cutpoints[0], cutpoints[1], cutpoints[2], cutpoints[3], np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(pd.Series(values), bins=bins, labels=labels, include_lowest=True).astype(int).to_numpy()


def recode_opioid_states(seq):
    arr = pd.to_numeric(pd.Series(seq), errors="coerce").to_numpy(dtype=float)

    # initialize everything as missing
    rec = np.full(arr.shape, 2, dtype=int)

    # current coding in your data:
    #   -1 = negative
    #    1 = positive
    # everything else (including NaN) stays as missing = 2
    rec[arr == -1] = 0
    rec[arr == 1] = 1

    return rec
