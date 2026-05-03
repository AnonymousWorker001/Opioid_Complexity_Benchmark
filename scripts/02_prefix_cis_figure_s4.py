#!/usr/bin/env python3
"""
Run the prefix-based CIS AUROC comparison for supplementary Figure S4.

This script follows the workflow in `notebooks/02_prefix_cis_figure_s4.ipynb`
and is formatted as a plain Python command-line script. The notebook is kept as
the executed reference record with printed outputs; this script is provided for
users who prefer a script-based workflow.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import os
import random
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


_REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_HINT))


from src.cohorts import (
    attach_row_maps,
    evaluate_prefix_sampled_cohort as evaluate_sampled_cohort,
    get_observed_bin_fraction_from_rows,
    materialize_prefix_sample_from_cache as materialize_sample_from_cache,
    sample_rows_by_bin,
    search_matched_prefix_cohort as search_matched_cohort,
)
from src.complexity import (
    assign_bins_from_cutpoints,
    calculate_permutation_entropy,
    compute_prefix_complexity_matrix_from_weekly,
    compute_prefix_patient_cis_weights as compute_cis_weights,
    compute_prefix_patient_complexity_from_weekly as compute_patient_complexity_from_weekly,
    compute_sample_prefix_cis_weights,
    compute_sample_prefix_pe,
    flatten_prefix_complexities_for_reference,
    get_prediction_weeks,
    hamming_distance,
)
from src.dataset import (
    enumerate_prediction_samples,
    generate_dataset_from_dataframe,
    split_train_test_stratify_prefix_entropy as split_train_test_stratify_permutation_entropy,
)
from src.figures import plot_lstm_grouped_auroc, plot_pe_tier_metrics
from src.metrics import (
    cis_weighted_aggregate_by_tier,
    compute_metrics_auroc_only as compute_metrics_local,
    compute_metrics_binary,
    weighted_brier,
)
from src.models import (
    TimeDependentLSTM,
    collect_week_level_predictions,
    collect_week_level_predictions_with_week,
    evaluate_by_pe_tier,
    evaluate_week_level,
    fit_model,
    load_LSTM_model_pars,
    prefix_cis_weighted_metrics_for_dataset as cis_weighted_metrics_for_dataset,
)
from src.utils import _move_to_repo_root, display, seed_everything



REPO_ROOT = _move_to_repo_root()

warnings.filterwarnings("ignore")


os.makedirs("Figs", exist_ok=True)

SEED = 42

seed_everything(SEED)
g = torch.Generator()
g.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Benchmark split
df_train, df_val, df_test, df_train_real, _ = split_train_test_stratify_permutation_entropy(
    csv_file="data/processed/static_timeSeries_new.csv", bins=5, no_filter=False
)

# 2) Train the treatment-aware LSTM
treat_ds_train = generate_dataset_from_dataframe(
    df_train, miss_value=2, future_window=1, prev_week_mode=2,
    include_treat_stats=True, include_tlstm_treat=True, daily_treatment=True
)
treat_ds_train_real = generate_dataset_from_dataframe(
    df_train_real, miss_value=2, future_window=1, prev_week_mode=2,
    include_treat_stats=True, include_tlstm_treat=True, daily_treatment=True
)
treat_ds_val = generate_dataset_from_dataframe(
    df_val, miss_value=2, future_window=1, prev_week_mode=2,
    include_treat_stats=True, include_tlstm_treat=True, daily_treatment=True
)

treat_train_loader = DataLoader(treat_ds_train, batch_size=128, shuffle=True, generator=g)
treat_train_real_loader = DataLoader(treat_ds_train_real, batch_size=128, shuffle=True, generator=g)
treat_val_loader = DataLoader(treat_ds_val, batch_size=256, shuffle=False)

model_treat, crit_t, opt_t = load_LSTM_model_pars(treat_ds_train, hidden_size=64, lr=0.0005, device=device)
model_treat_real, crit_t_real, opt_t_real = load_LSTM_model_pars(treat_ds_train_real, hidden_size=64, lr=0.0005, device=device)
_, hist_treat, best_epoch = fit_model(
    model_treat_real, treat_train_real_loader, treat_val_loader, crit_t_real, opt_t_real,
    no_static=False, include_tlstm_treat=True, n_epochs=50, device=device,
    return_history=True
)
model_treat, _, _ = fit_model(
    model_treat, treat_train_loader, treat_val_loader, crit_t, opt_t,
    no_static=False, include_tlstm_treat=True, n_epochs=best_epoch, device=device,
    return_history=True
)

# 3) Build fixed LSTM prediction cache on the original test set


df_train_ref = df_train.copy().reset_index(drop=True)
train_prefix_values_ref = flatten_prefix_complexities_for_reference(df_train_ref)
prefix_cutpoints_sample_ref = np.quantile(train_prefix_values_ref, [0.2, 0.4, 0.6, 0.8])

df_test_pool = df_test.copy().reset_index(drop=True)
treat_ds_full = generate_dataset_from_dataframe(
    df_test_pool,
    miss_value=2, future_window=1, prev_week_mode=2,
    include_treat_stats=True, include_tlstm_treat=True, daily_treatment=True
)

y_full, p_full, sidx_full, _ = collect_week_level_predictions_with_week(
    model_treat, treat_ds_full, include_tlstm_treat=True, device=device
)

expected_pid, expected_actual_week = enumerate_prediction_samples(
    df_test_pool, n_timesteps=24, future_window=1, start_week=3
)
if len(sidx_full) != len(expected_pid):
    raise RuntimeError("Unexpected mismatch between LSTM prediction rows and expected sample enumeration.")
if not np.array_equal(np.asarray(sidx_full, dtype=int), expected_pid):
    raise RuntimeError("LSTM prediction row order does not match expected patient/sample order.")

prefix_w, prefix_pe = compute_sample_prefix_cis_weights(
    df_train_ref, df_test_pool, expected_pid, expected_actual_week,
    train_prefix_values=train_prefix_values_ref
)
prefix_bin = assign_bins_from_cutpoints(prefix_pe, prefix_cutpoints_sample_ref)

prediction_cache = {}
prediction_cache["LSTM"] = attach_row_maps({
    "model": "LSTM",
    "y": np.asarray(y_full, dtype=int),
    "p": np.asarray(p_full, dtype=float),
    "pid": np.asarray(expected_pid, dtype=int),
    "week": np.asarray(expected_actual_week, dtype=int),
    "prefix_pe": np.asarray(prefix_pe, dtype=float),
    "prefix_cis": np.asarray(prefix_w, dtype=float),
    "prefix_bin": np.asarray(prefix_bin, dtype=int),
})

# 4) Matched cohort construction


cohort_target_probs = {
    "C1": [0.00, 0.10, 0.80, 0.10, 0.00],
    "C2": [0.05, 0.15, 0.60, 0.15, 0.05],
    "C3": [0.20, 0.10, 0.40, 0.10, 0.20],
    "C4": [0.30, 0.10, 0.20, 0.10, 0.30],
    "C5": [0.40, 0.10, 0.00, 0.10, 0.40],
}

anchor_entry = prediction_cache["LSTM"]
anchor_full_plain = compute_metrics_local(anchor_entry["y"], anchor_entry["p"], None)["auroc"]
n_rows_sim = len(anchor_entry["y"])

sampled_cohort_store = {}
for i, (cohort_name, target_probs) in enumerate(cohort_target_probs.items()):
    sampled_cohort_store[cohort_name] = search_matched_cohort(
        cache_entry_anchor=anchor_entry,
        target_probs=target_probs,
        n_rows=n_rows_sim,
        target_plain_auroc=anchor_full_plain,
        n_trials=800,
        seed=100 + i,
        auroc_tol=0.008,
        comp_weight=0.25,
    )

# 5) Evaluate LSTM across cohorts
rows = []
for cohort_name, info in sampled_cohort_store.items():
    m = evaluate_sampled_cohort(prediction_cache["LSTM"], info["sampled_row_keys"])
    rows.append({
        "cohort": cohort_name,
        "model": "LSTM",
        **m
    })
eval_df = pd.DataFrame(rows)

# 6) Plot Figure
cohort_order = list(cohort_target_probs.keys())
purple_base = "#7B1FA2"
cohort_facecolors = [
    to_rgba(purple_base, 0.20),
    to_rgba(purple_base, 0.40),
    to_rgba(purple_base, 0.60),
    to_rgba(purple_base, 0.80),
    to_rgba(purple_base, 1.0),
]
cohort_edgecolor = "#4A148C"


plot_lstm_grouped_auroc(eval_df, fname="FigureS4_AUROC.pdf", zoom_pad=0.015)

