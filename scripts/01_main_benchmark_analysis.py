#!/usr/bin/env python3
"""
Run the main OUD complexity benchmark analysis.

This script follows the analysis workflow in `notebooks/01_main_benchmark_analysis.ipynb`
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
from matplotlib.gridspec import GridSpec
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
    attach_pid_row_map,
    build_sampled_patient_df,
    evaluate_sampled_cohort,
    get_observed_bin_fraction,
    materialize_sample_from_cache,
    sample_patients_by_bin,
    search_matched_cohort,
)
from src.complexity import (
    calculate_permutation_entropy,
    compute_cis_weights,
    compute_patient_complexity_from_weekly,
    hamming_distance,
    recode_opioid_states,
)
from src.dataset import (
    build_week_level_tabular,
    generate_dataset_from_dataframe,
    split_train_test_stratify_permutation_entropy,
)
from src.figures import plot_loss_curves, plot_one_model_grouped, plot_pe_tier_metrics
from src.metrics import (
    _compute_metrics,
    _safe_auprc,
    _safe_auroc,
    _weighted_brier,
    cis_weighted_aggregate_by_tier,
    compute_metrics_binary,
    compute_metrics_local,
    fit_eval_tabular,
    weighted_brier,
    weighted_brier_local,
)
from src.models import (
    TimeDependentLSTM,
    cis_weighted_metrics_for_dataset,
    collect_week_level_predictions,
    collect_week_level_probs_with_pid,
    evaluate_by_pe_tier,
    evaluate_week_level,
    fit_model,
    load_LSTM_model_pars,
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
print("device:", device)

# Stratified development / test split by permutation-entropy tier
df_train, df_val, df_test, df_train_real, bins_res = split_train_test_stratify_permutation_entropy(
    csv_file="data/processed/static_timeSeries_new.csv", bins=5, no_filter=False
)

# Figure S2: tier composition in development and test splits
bins = 5
freqs = []
freqs_test = []
for i in range(1, bins + 1):
    freqs.append(np.sum(df_train["Target_Permutation_Entropy"] == i))
    freqs_test.append(np.sum(df_test["Target_Permutation_Entropy"] == i))

x = np.arange(1, bins + 1)
plt.figure(figsize=(6, 5))
plt.barh(x, freqs, label="Train", color="lightblue")
plt.barh(x, freqs_test, left=freqs, label="Test", color="pink")
plt.yticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=20)
plt.xlabel("Frequency", fontsize=20)
plt.legend(fontsize=16, loc="upper left")
plt.tight_layout()
plt.savefig("Figs/FigS2.pdf", bbox_inches="tight")

# Treatment-aware LSTM
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

print(f"Setup complete: model_treat is ready, best_epoch is {best_epoch+1}.")


df_train, df_val, df_test, df_test_random, bins_res = split_train_test_stratify_permutation_entropy(
    csv_file="data/processed/static_timeSeries_new.csv", bins=5, no_filter=False
)
n_timesteps = int(globals().get("n_timesteps", 24))
df_tr_all = pd.concat([df_train, df_val], ignore_index=True)

X_tr_nt, y_tr_nt, pid_tr_nt, _ = build_week_level_tabular(df_tr_all, n_timesteps=n_timesteps, include_treat=False, include_dose=False)
X_te_nt, y_te_nt, pid_te_nt, _ = build_week_level_tabular(df_test,  n_timesteps=n_timesteps, include_treat=False, include_dose=False)

X_tr_tx, y_tr_tx, pid_tr_tx, _ = build_week_level_tabular(df_tr_all, n_timesteps=n_timesteps, include_treat=True, include_dose=True)
X_te_tx, y_te_tx, pid_te_tx, _ = build_week_level_tabular(df_test,  n_timesteps=n_timesteps, include_treat=True, include_dose=True)

tier_test = df_test["Target_Permutation_Entropy"].values if "Target_Permutation_Entropy" in df_test.columns else None

# CIS patient weights (CIS weights)
if ("compute_patient_complexity_from_weekly" in globals()) and ("compute_cis_weights" in globals()):
    pe_train = compute_patient_complexity_from_weekly(df_tr_all)
    pe_test  = compute_patient_complexity_from_weekly(df_test)
    cis_test = compute_cis_weights(df_test)
else:
    # fallback: equal weights
    cis_test = np.ones(df_test.shape[0], dtype=float)


# Define tabular baseline suite
models = [
    ("LR (no-tx)", Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
     X_tr_nt, y_tr_nt, X_te_nt, y_te_nt, pid_te_nt),

    ("RF (no-tx)", RandomForestClassifier(n_estimators=300, random_state=42,
                                          class_weight="balanced_subsample", n_jobs=-1),
     X_tr_nt, y_tr_nt, X_te_nt, y_te_nt, pid_te_nt),

    ("GBDT (no-tx)", GradientBoostingClassifier(random_state=42),
     X_tr_nt, y_tr_nt, X_te_nt, y_te_nt, pid_te_nt),

    ("LR (+tx)", Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
     X_tr_tx, y_tr_tx, X_te_tx, y_te_tx, pid_te_tx),

    ("RF (+tx)", RandomForestClassifier(n_estimators=300, random_state=42,
                                        class_weight="balanced_subsample", n_jobs=-1),
     X_tr_tx, y_tr_tx, X_te_tx, y_te_tx, pid_te_tx),
]

rows = []
tier_maps = {}

for name, clf, Xtr, ytr, Xte, yte, pidte in models:
    m_unw, m_w, by_tier = fit_eval_tabular(name, clf, Xtr, ytr, Xte, yte, pidte, tier_test, cis_patient=cis_test)
    rows.append({
        "model": name,
        "auroc": m_unw["auroc"], "auprc": m_unw["auprc"], "brier": m_unw["brier"],
        "cis_auroc": (m_w["auroc"] if m_w is not None else np.nan),
        "cis_auprc": (m_w["auprc"] if m_w is not None else np.nan),
        "cis_brier": (m_w["brier"] if m_w is not None else np.nan),
    })
    tier_maps[name] = by_tier

# Add LSTM baselines (overall metrics + CIS/CIS-weighted)

device = globals().get("device", "cpu")
miss_value = float(globals().get("miss_value", 2))
prev_week_mode = int(globals().get("prev_week_mode", 2))


# Prefer existing loaders if already created earlier; else build here
base_test_loader = globals().get("base_test_loader", None)
treat_test_loader = globals().get("treat_test_loader", None)

if base_test_loader is None or treat_test_loader is None:
    base_ds_test = generate_dataset_from_dataframe(
        df_test, miss_value=miss_value, future_window=1,
        daily_treatment=False, prev_week_mode=prev_week_mode,
        include_treat_stats=False, include_tlstm_treat=False
    )
    treat_ds_test = generate_dataset_from_dataframe(
        df_test, miss_value=miss_value, future_window=1,
        daily_treatment=True, prev_week_mode=prev_week_mode,
        include_treat_stats=True, include_tlstm_treat=True
    )
    base_test_loader = DataLoader(base_ds_test, batch_size=256, shuffle=False)
    treat_test_loader = DataLoader(treat_ds_test, batch_size=256, shuffle=False)

# Model names might vary; try common fallbacks
model_treat = globals().get("model_treat", None)

y_t2, p_t2, pid_t2 = collect_week_level_probs_with_pid(model_treat, treat_test_loader, include_tlstm_treat=True, device=device)

treat_unw = _compute_metrics(y_t2, p_t2, w=None)

w_treat = cis_test[pid_t2.astype(int)]
treat_w = _compute_metrics(y_t2, p_t2, w=w_treat)

rows.append({
    "model": "LSTM +Treatment",
    "auroc": treat_unw["auroc"], "auprc": treat_unw["auprc"], "brier": treat_unw["brier"],
    "cis_auroc": treat_w["auroc"], "cis_auprc": treat_w["auprc"], "cis_brier": treat_w["brier"],
})

df_models = pd.DataFrame(rows)

# Sort by CIS-weighted AUROC if available, else fallback to AUROC
sort_col = "cis_auroc" if df_models["cis_auroc"].notna().any() else "auroc"
df_models = df_models.sort_values(sort_col, ascending=False).reset_index(drop=True)

# Training / validation loss curves


plot_loss_curves(hist_treat, outpath="Figs/FigS1.pdf", best_epoch=best_epoch+1)

# Build the fixed prediction cache on the original test set
#   - Keep only: LSTM, LR, RF, GBDT
#   - LR / RF use treatment-aware features
#   - GBDT kept as fallback without treatment features


required_names = [
    "df_train", "df_val", "df_test",
    "model_treat", "device",
    "generate_dataset_from_dataframe", "collect_week_level_predictions",
    "compute_patient_complexity_from_weekly", "compute_cis_weights",
    "build_week_level_tabular"
]
missing = [x for x in required_names if x not in globals()]
if len(missing) > 0:
    raise RuntimeError(f"Missing required notebook objects/functions: {missing}")


df_train_ref = df_train.copy().reset_index(drop=True)
pe_train_ref = compute_patient_complexity_from_weekly(df_train_ref)

df_test_pool = df_test.copy().reset_index(drop=True)
pe_test_pool = compute_patient_complexity_from_weekly(df_test_pool)
cis_test_pool = compute_cis_weights(df_test_pool)
df_test_pool["_cis_score"] = cis_test_pool

rank_score = pd.Series(df_test_pool["_cis_score"]).rank(method="first")
df_test_pool["_complexity_bin"] = (
    pd.qcut(rank_score, q=5, labels=[1, 2, 3, 4, 5]).astype(int)
)


prediction_cache = {}

# LSTM (treatment-aware version only)
treat_ds_full = generate_dataset_from_dataframe(
    df_test_pool,
    miss_value=2, future_window=1, prev_week_mode=2,
    include_treat_stats=True, include_tlstm_treat=True, daily_treatment=True
)

y_lstm_full, p_lstm_full, sidx_lstm_full = collect_week_level_predictions(
    model_treat, treat_ds_full, include_tlstm_treat=True, device=device
)

prediction_cache["LSTM"] = attach_pid_row_map({
    "model": "LSTM",
    "kind": "lstm",
    "y": np.asarray(y_lstm_full, dtype=int),
    "p": np.asarray(p_lstm_full, dtype=float),
    "pid": np.asarray(sidx_lstm_full, dtype=int),
})

# Tabular models
#   LR / RF with treatment-aware features
#   GBDT without treatment-aware features
X_tr_tx, y_tr_tx, pid_tr_tx, _ = build_week_level_tabular(
    df_train_ref, include_treat=True, include_dose=True
)
X_te_tx, y_te_tx, pid_te_tx, _ = build_week_level_tabular(
    df_test_pool, include_treat=True, include_dose=True
)

X_tr_nt, y_tr_nt, pid_tr_nt, _ = build_week_level_tabular(
    df_train_ref, include_treat=False, include_dose=False
)
X_te_nt, y_te_nt, pid_te_nt, _ = build_week_level_tabular(
    df_test_pool, include_treat=False, include_dose=False
)

tabular_models = [
    (
        "LR",
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        X_tr_tx, y_tr_tx, X_te_tx, y_te_tx, pid_te_tx
    ),
    (
        "RF",
        RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1
        ),
        X_tr_tx, y_tr_tx, X_te_tx, y_te_tx, pid_te_tx
    ),
    (
        "GBDT",
        GradientBoostingClassifier(random_state=42),
        X_tr_tx, y_tr_tx, X_te_tx, y_te_tx, pid_te_tx
    ),
]

for name, clf, Xtr, ytr, Xte, yte, pidte in tabular_models:
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    prediction_cache[name] = attach_pid_row_map({
        "model": name,
        "kind": "tabular",
        "y": np.asarray(yte, dtype=int),
        "p": np.asarray(prob, dtype=float),
        "pid": np.asarray(pidte, dtype=int),
    })

print("Built prediction cache for:")
print(list(prediction_cache.keys()))

baseline_rows = []
for model_name, entry in prediction_cache.items():
    m_plain = compute_metrics_local(entry["y"], entry["p"], sample_weight=None)
    w_full = df_test_pool["_cis_score"].to_numpy()[entry["pid"]]
    m_w = compute_metrics_local(entry["y"], entry["p"], sample_weight=w_full)

    baseline_rows.append({
        "model": model_name,
        "plain_auroc_full": m_plain["auroc"],
        "cis_auroc_full": m_w["auroc"],
        "plain_auprc_full": m_plain["auprc"],
        "cis_auprc_full": m_w["auprc"],
        "plain_brier_full": m_plain["brier"],
        "cis_brier_full": m_w["brier"],
        "AUROC gap": m_plain["auroc"] - m_w["auroc"],
        "AUPRC gap": m_plain["auprc"] - m_w["auprc"],
        "Brier gap": m_w["brier"] - m_plain["brier"],
    })

baseline_df = (
    pd.DataFrame(baseline_rows)
    .sort_values("plain_auroc_full", ascending=False)
    .reset_index(drop=True)
)
print("Results for Table 1")
display(baseline_df)

# exact class counts on the original held-out test set
entry = prediction_cache["LSTM"]   # or any model; y is the same label vector
y = np.asarray(entry["y"], dtype=int)
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
print(n_pos, n_neg, n_pos / len(y))

n_pos = int((y_tr_tx == 1).sum())
n_neg = int((y_tr_tx == 0).sum())

print(f"Positive samples: {n_pos}")
print(f"Negative samples: {n_neg}")
print(f"Event prevalence: {n_pos / len(y_tr_tx):.4f}")


gamma_list = [0.5, 1, 1.5, 2, 5]
model_order = ["LSTM", "GBDT", "LR", "RF"]

all_rows = []

for gamma in gamma_list:

    # compute CIS / CIS weights on the original test pool
    cis_test_pool = compute_cis_weights(df_test_pool, gamma=gamma)
    df_test_pool["_cis_score"] = cis_test_pool

    baseline_rows = []
    for model_name, entry in prediction_cache.items():
        if model_name not in model_order:
            continue

        m_plain = compute_metrics_local(entry["y"], entry["p"], sample_weight=None)
        w_full = df_test_pool["_cis_score"].to_numpy()[entry["pid"]]
        m_w = compute_metrics_local(entry["y"], entry["p"], sample_weight=w_full)

        baseline_rows.append({
            "gamma": gamma,
            "model": model_name,
            "AUROC gap": m_plain["auroc"] - m_w["auroc"],
            "AUPRC gap": m_plain["auprc"] - m_w["auprc"],
            "Brier gap": m_w["brier"] - m_plain["brier"],
        })

    baseline_df = pd.DataFrame(baseline_rows)

    # enforce model order
    baseline_df["model"] = pd.Categorical(
        baseline_df["model"], categories=model_order, ordered=True
    )
    baseline_df = baseline_df.sort_values("model").reset_index(drop=True)

    print(f"Results for gamma = {gamma}")
    display(baseline_df)

    all_rows.append(baseline_df)

# Combine all results
gap_df = pd.concat(all_rows, ignore_index=True)

# Plot settings
color_map = {
    "LSTM": "#1f77b4",
    "GBDT": "#ff7f0e",
    "LR": "#2ca02c",
    "RF": "#d62728",
}
marker_map = {
    "LSTM": "o",
    "GBDT": "s",
    "LR": "^",
    "RF": "D",
}

# Combined 3-panel figure: AUROC / AUPRC / Brier gaps vs gamma
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharex=True)

metric_specs = [
    ("AUROC gap", "AUROC gap", "(a)"),
    ("AUPRC gap", "AUPRC gap", "(b)"),
    ("Brier gap", "Brier gap", "(c)")
]

for ax, (metric_col, ylab, panel_tag) in zip(axes, metric_specs):
    for model in model_order:
        sub = gap_df[gap_df["model"] == model].sort_values("gamma")
        if sub.empty:
            continue

        x_vals = sub["gamma"].to_numpy(dtype=float)
        y_vals = sub[metric_col].to_numpy(dtype=float)

        ax.plot(
            x_vals, y_vals,
            label=model,
            color=color_map[model],
            marker=marker_map[model],
            linewidth=2.2,
            markersize=6
        )

    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.set_title(ylab, fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)

    ax.text(-0.18, 1.05, panel_tag, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    # title="Model",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    frameon=False,
    fontsize=11,
    title_fontsize=11
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("Figs/FigSX_gamma_gap_3panel.pdf", dpi=300, bbox_inches="tight")


D_list = [3, 4, 5, 6, 7, 8, 9, 10]
model_order = ["LSTM", "GBDT", "LR", "RF"]

all_rows = []

for D in D_list:

    # compute CIS / CIS weights on the original test pool
    cis_test_pool = compute_cis_weights(df_test_pool, D=D)
    df_test_pool["_cis_score"] = cis_test_pool

    baseline_rows = []
    for model_name, entry in prediction_cache.items():
        if model_name not in model_order:
            continue

        m_plain = compute_metrics_local(entry["y"], entry["p"], sample_weight=None)
        w_full = df_test_pool["_cis_score"].to_numpy()[entry["pid"]]
        m_w = compute_metrics_local(entry["y"], entry["p"], sample_weight=w_full)

        baseline_rows.append({
            "D": D,
            "model": model_name,
            "AUROC gap": m_plain["auroc"] - m_w["auroc"],
            "AUPRC gap": m_plain["auprc"] - m_w["auprc"],
            "Brier gap": m_w["brier"] - m_plain["brier"],
        })

    baseline_df = pd.DataFrame(baseline_rows)

    # enforce model order
    baseline_df["model"] = pd.Categorical(
        baseline_df["model"], categories=model_order, ordered=True
    )
    baseline_df = baseline_df.sort_values("model").reset_index(drop=True)

    print(f"Results for D = {D}")
    display(baseline_df)

    all_rows.append(baseline_df)

# Combine all results
gap_df = pd.concat(all_rows, ignore_index=True)

# Plot settings
color_map = {
    "LSTM": "#1f77b4",
    "GBDT": "#ff7f0e",
    "LR": "#2ca02c",
    "RF": "#d62728",
}
marker_map = {
    "LSTM": "o",
    "GBDT": "s",
    "LR": "^",
    "RF": "D",
}

# Combined 3-panel figure: AUROC / AUPRC / Brier gaps vs D
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharex=True)

metric_specs = [
    ("AUROC gap", "AUROC gap", "(a)"),
    ("AUPRC gap", "AUPRC gap", "(b)"),
    ("Brier gap", "Brier gap", "(c)")
]

for ax, (metric_col, ylab, panel_tag) in zip(axes, metric_specs):
    for model in model_order:
        sub = gap_df[gap_df["model"] == model].sort_values("D")
        if sub.empty:
            continue

        x_vals = sub["D"].to_numpy(dtype=float)
        y_vals = sub[metric_col].to_numpy(dtype=float)

        ax.plot(
            x_vals, y_vals,
            label=model,
            color=color_map[model],
            marker=marker_map[model],
            linewidth=2.2,
            markersize=6
        )

    ax.set_xlabel("D", fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.set_title(ylab, fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)

    ax.text(-0.18, 1.05, panel_tag, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    # title="Model",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    frameon=False,
    fontsize=11,
    title_fontsize=11
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("Figs/FigSY_D_gap_3panel.pdf", dpi=300, bbox_inches="tight")

# Apply tier cutpoint correction:
# define NON-equal-frequency tiers using train-reference cutpoints,
# then apply them to the original test set


# 1) continuous patient-level complexity score on train-ref and test
train_complexity_score = pd.Series(
    compute_patient_complexity_from_weekly(df_train_ref),
    index=df_train_ref.index,
    dtype=float
)

test_complexity_score = pd.Series(
    compute_patient_complexity_from_weekly(df_test_pool),
    index=df_test_pool.index,
    dtype=float
)

# 2) define fixed cutpoints from the TRAIN reference distribution
#    These are thresholds, not equal-frequency bins on the test set.
cutpoints = np.quantile(train_complexity_score.dropna(), [0.2, 0.4, 0.6, 0.8])

# 3) apply those fixed cutpoints to the ORIGINAL test set
#    This will usually produce an uneven distribution.
bins = [-np.inf, cutpoints[0], cutpoints[1], cutpoints[2], cutpoints[3], np.inf]
labels = [1, 2, 3, 4, 5]

df_test_pool["_complexity_score_raw"] = test_complexity_score.values

# Figure 2: complexity-tier distribution + example trajectories
# Left: horizontal tier bars
# Right: example 24-week opioid trajectories shown as subbands within each tier

# 1) Select dataframe
if "df_test_pool" in globals():
    plot_df = df_test_pool.copy()
elif "df_test" in globals():
    plot_df = df_test.copy()
else:
    raise RuntimeError("Need df_test_pool or df_test in memory before plotting Figure 2.")

if "Target_Permutation_Entropy" in plot_df.columns:
    tier_col = "Target_Permutation_Entropy"
elif "_complexity_bin_real" in plot_df.columns:
    tier_col = "_complexity_bin_real"
else:
    raise RuntimeError("Could not find a tier column.")

# 2) Detect opioid weekly columns
opioid_cols = [c for c in plot_df.columns if re.match(r"Opioid_week\d+$", c)]
if len(opioid_cols) == 0:
    raise RuntimeError("No columns like 'Opioid_week0', 'Opioid_week1', ... were found.")

opioid_cols = sorted(opioid_cols, key=lambda x: int(re.findall(r"(\d+)$", x)[0]))
n_weeks = len(opioid_cols)

plot_df = plot_df.loc[plot_df[tier_col].notna()].copy()
plot_df[tier_col] = plot_df[tier_col].astype(int)

# 3) Tier summary
tier_order = [1, 2, 3, 4, 5]
tier_labels = [f"Q{i}" for i in tier_order]
tier_colors = ['#50C878', '#98FF98', '#00FFFF', '#113388', '#512888']

tier_counts = np.array([(plot_df[tier_col] == q).sum() for q in tier_order], dtype=int)
tier_perc = 100 * tier_counts / tier_counts.sum()

# 4) Choose example trajectories per tier
rng = np.random.default_rng(42)
n_examples_per_tier = 4

example_rows = {}
for q in tier_order:
    idx = plot_df.index[plot_df[tier_col] == q].to_numpy()
    if len(idx) == 0:
        example_rows[q] = np.array([], dtype=int)
    elif len(idx) <= n_examples_per_tier:
        example_rows[q] = idx
    else:
        example_rows[q] = rng.choice(idx, size=n_examples_per_tier, replace=False)

# 5) Helper: force trajectory states into 0/1/2
#    0 = negative, 1 = positive, 2 = missing

# 6) Plot
fig = plt.figure(figsize=(15.5, 9.0))
gs = GridSpec(
    nrows=5, ncols=2,
    width_ratios=[2.05, 3.20],
    height_ratios=[1, 1, 1, 1, 1],
    wspace=0.15, hspace=0.25
)

# Left panel: tier distribution
ax_bar = fig.add_subplot(gs[:, 0])

ypos = np.arange(len(tier_order))
bars = ax_bar.barh(
    ypos, tier_counts,
    color=tier_colors,
    edgecolor="black",
    linewidth=0.8,
    height=0.78,
    alpha=0.68
)

ax_bar.set_yticks(ypos)
ax_bar.set_yticklabels(tier_labels, fontsize=13)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Patients", fontsize=13)
ax_bar.set_title("Complexity-tier distribution", fontsize=14)
ax_bar.tick_params(axis="x", labelsize=11)
ax_bar.grid(axis="x", alpha=0.25)

# rotate percentage labels to save space
x_pad = max(tier_counts) * 0.015
for y, c, p in zip(ypos, tier_counts, tier_perc):
    ax_bar.text(
        c + x_pad,
        y,
        f"{c} ({p:.1f}%)",
        rotation=-90,
        va="center",
        ha="left",
        fontsize=14
    )

ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# Right panel: example trajectories by tier
# Each tier gets one axis, and within it each example gets its own subband
# traj_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
traj_markers = ["o", "s", "^", "D", "P", "X"]

right_axes = []

for i, q in enumerate(tier_order):
    sharex_ax = right_axes[0] if len(right_axes) > 0 else None
    ax = fig.add_subplot(gs[i, 1], sharex=sharex_ax)
    right_axes.append(ax)

    selected = example_rows[q]
    n_sel = len(selected)

    # layout for subbands
    row_height = 3.0     # for the 3 state levels 0/1/2
    row_gap = 1.0
    block_height = row_height + row_gap

    # total y span
    y_max = max(1, n_examples_per_tier) * block_height - row_gap

    # light separators / subband backgrounds
    for j in range(n_examples_per_tier):
        base = (n_examples_per_tier - 1 - j) * block_height
        ax.axhspan(base - 0.15, base + 2.15, color="0.97", zorder=0)
        ax.hlines(base - 0.15, 1, n_weeks, color="0.90", linewidth=0.8, zorder=0)

    # plot selected trajectories
    for j, row_id in enumerate(selected):
        seq = recode_opioid_states(plot_df.loc[row_id, opioid_cols])
        x = np.arange(1, n_weeks + 1)

        base = (n_examples_per_tier - 1 - j) * block_height
        y = seq + base

        ax.step(
            x, y,
            where="post",
            color = tier_colors[i],#traj_colors[j % len(traj_colors)],
            linewidth=1.8,
            alpha=0.95
        )
        ax.plot(
            x, y,
            linestyle="None",
            marker=traj_markers[j % len(traj_markers)],
            markersize=3.2,
            color = tier_colors[i],#traj_colors[j % len(traj_colors)],
            alpha=0.95
        )

        # label each example row
        ax.text(
            0.35, base + 1.0,
            f"Ex {j+1}",
            fontsize=9,
            ha="right",
            va="center"
            # color = tier_colors[i]#traj_colors[j % len(traj_colors)]
        )

    ax.set_ylim(-0.4, y_max + 0.4)
    ax.set_xlim(1, n_weeks)

    # remove crowded y-axis
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)

    ax.grid(axis="x", alpha=0.18)

    if i < len(tier_order) - 1:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Week", fontsize=13)
        ax.tick_params(axis="x", labelsize=11)

    if i == 0:
        ax.set_title(
            "Representative 24-week opioid trajectories",
            # "(state levels within each row: 0 = negative, 1 = positive, 2 = missing)",
            fontsize=14
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("Figs/Figure2_revised_distribution_plus_examples.pdf", dpi=300, bbox_inches="tight")

#   1) LSTM performance by pure complexity tier (Q1..Q5), independent of cohorts
#   2) LSTM plain vs CIS-weighted performance on the full original test set


axis_label_fs = 14
tick_fs = 13
title_fs = 15

# Safety checks
required_names = ["prediction_cache", "df_test_pool", "compute_metrics_local"]
missing = [x for x in required_names if x not in globals()]
if len(missing) > 0:
    raise RuntimeError(f"Missing required objects from previous analysis sections: {missing}")

if "LSTM" not in prediction_cache:
    raise RuntimeError("prediction_cache['LSTM'] not found. Please run the previous analysis sections first.")

lstm_entry = prediction_cache["LSTM"]
y_full = np.asarray(lstm_entry["y"], dtype=int)
p_full = np.asarray(lstm_entry["p"], dtype=float)
pid_full = np.asarray(lstm_entry["pid"], dtype=int)

# patient-level tier / weight mapped to week-level rows
tier_full = df_test_pool["Target_Permutation_Entropy"].to_numpy().astype(int)[pid_full]
cis_w_full = df_test_pool["_cis_score"].to_numpy()[pid_full]

# 1) Pure tier-specific performance on the ORIGINAL dataset
tier_rows = []
for q in [1, 2, 3, 4, 5]:
    mask = (tier_full == q)
    y_q = y_full[mask]
    p_q = p_full[mask]

    m_q = compute_metrics_local(y_q, p_q, sample_weight=None)

    tier_rows.append({
        "tier": f"Q{q}",
        "n_weeks": int(mask.sum()),
        "positive_rate": float(np.mean(y_q)) if len(y_q) > 0 else np.nan,
        "auroc": m_q["auroc"],
        "auprc": m_q["auprc"],
        "brier": m_q["brier"],
    })

tier_perf_df = pd.DataFrame(tier_rows)
print("LSTM tier-specific performance on the full original test set:")
display(tier_perf_df)

# AUROC by tier
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = np.arange(tier_perf_df.shape[0])
vals = tier_perf_df["auroc"].to_numpy(dtype=float)

ax.bar(x, vals)
ax.set_xticks(x)
ax.set_xticklabels(tier_perf_df["tier"].tolist(), fontsize=tick_fs)
ax.tick_params(axis="y", labelsize=tick_fs)
ax.set_ylabel("AUROC", fontsize=axis_label_fs)
ax.set_xlabel("Complexity tier on original test set", fontsize=title_fs)
finite_vals = vals[np.isfinite(vals)]
if len(finite_vals) > 0:
    ax.set_ylim(max(0.0, float(finite_vals.min()) - 0.03), min(1.0, float(finite_vals.max()) + 0.03))
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig('Figs/Figure3a.pdf')
print(vals)

# AUPRC by tier
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = np.arange(tier_perf_df.shape[0])
vals = tier_perf_df["auprc"].to_numpy(dtype=float)

ax.bar(x, vals)
ax.set_xticks(x)
ax.set_xticklabels(tier_perf_df["tier"].tolist(), fontsize=tick_fs)
ax.tick_params(axis="y", labelsize=tick_fs)
ax.set_ylabel("AUPRC", fontsize=axis_label_fs)
ax.set_xlabel("Complexity tier on original test set", fontsize=title_fs)
finite_vals = vals[np.isfinite(vals)]
if len(finite_vals) > 0:
    ax.set_ylim(max(0.0, float(finite_vals.min()) - 0.03), min(1.0, float(finite_vals.max()) + 0.03))
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig('Figs/Figure3b.pdf')
print(vals)
# 2) Full original dataset: plain vs CIS-weighted
m_plain_full = compute_metrics_local(y_full, p_full, sample_weight=None)
m_cis_full = compute_metrics_local(y_full, p_full, sample_weight=cis_w_full)

orig_compare_df = pd.DataFrame([
    {
        "score_type": "Plain",
        "auroc": m_plain_full["auroc"],
        "auprc": m_plain_full["auprc"],
        "brier": m_plain_full["brier"],
    },
    {
        "score_type": "CIS-weighted",
        "auroc": m_cis_full["auroc"],
        "auprc": m_cis_full["auprc"],
        "brier": m_cis_full["brier"],
    }
])

# Complexity-based resampling design
#   5 cohorts with gradually increased bimodal low+high composition
#   Target compositions:
#     C1 = [0,10,80,10,0]
#     C2 = [5,15,60,15,5]
#     C3 = [20,10,40,10,20]
#     C4 = [30,10,20,10,30]
#     C5 = [40,10,0,10,40]
#   Search objective:
#     - keep anchor-model plain AUROC similar
#     - match target composition


cohort_target_probs = {
    "C1": [0.00, 0.10, 0.80, 0.10, 0.00],
    "C2": [0.05, 0.15, 0.60, 0.15, 0.05],
    "C3": [0.20, 0.10, 0.40, 0.10, 0.20],
    "C4": [0.30, 0.10, 0.20, 0.10, 0.30],
    "C5": [0.40, 0.10, 0.00, 0.10, 0.40],
}

n_patients_sim = df_test_pool.shape[0]

anchor_model = "LSTM"
anchor_entry = prediction_cache[anchor_model]
anchor_full_plain = compute_metrics_local(anchor_entry["y"], anchor_entry["p"], None)["auroc"]

print(f"Anchor model: {anchor_model}")
print(f"Target plain AUROC to match: {anchor_full_plain:.4f}")

sampled_cohort_store = {}
cohort_search_rows = []

for i, (cohort_name, target_probs) in enumerate(cohort_target_probs.items()):
    best = search_matched_cohort(
        df_pool=df_test_pool,
        cache_entry_anchor=anchor_entry,
        df_train_ref=df_train_ref,
        pe_train_ref=pe_train_ref,
        target_probs=target_probs,
        n_patients=n_patients_sim,
        target_plain_auroc=anchor_full_plain,
        n_trials=800,
        seed=100 + i,
        auroc_tol=0.008,
        comp_weight=0.25,
    )

    sampled_cohort_store[cohort_name] = best

    row = {
        "cohort": cohort_name,
        "target_plain_auroc_anchor": anchor_full_plain,
        "achieved_plain_auroc_anchor": best["anchor_metrics"]["plain_auroc"],
        "achieved_cis_auroc_anchor": best["anchor_metrics"]["cis_auroc"],
        "achieved_plain_auprc_anchor": best["anchor_metrics"]["plain_auprc"],
        "achieved_cis_auprc_anchor": best["anchor_metrics"]["cis_auprc"],
        "objective": best["objective"],
        "plain_err": best["plain_err"],
        "comp_err": best["comp_err"],
    }

    for b, frac in zip([1,2,3,4,5], target_probs):
        row[f"target_Q{b}"] = frac
    for b, frac in zip([1,2,3,4,5], best["observed_bin_frac"]):
        row[f"observed_Q{b}"] = frac

    cohort_search_rows.append(row)

cohort_search_df = pd.DataFrame(cohort_search_rows)
print("Results for Table 2")
display(cohort_search_df)

# Evaluate all selected models on all sampled cohorts

all_eval_rows = []

for cohort_name, info in sampled_cohort_store.items():
    sampled_ids = info["sampled_ids"]
    sampled_df = info["sampled_df"]

    for model_name, cache_entry in prediction_cache.items():
        m = evaluate_sampled_cohort(
            cache_entry, sampled_ids, sampled_df, df_train_ref, pe_train_ref
        )

        all_eval_rows.append({
            "cohort": cohort_name,
            "model": model_name,
            **m
        })

eval_df = pd.DataFrame(all_eval_rows)

summary_cols = [
    "cohort", "model",
    "plain_auroc", "cis_auroc",
    "plain_auprc", "cis_auprc",
    "plain_brier", "cis_brier"
]

print("All models across all cohorts:")
print("Results for Table 3")
display(eval_df[summary_cols].sort_values(["model", "cohort"]).reset_index(drop=True))

print("LSTM only:")
display(
    eval_df[eval_df["model"] == "LSTM"][summary_cols]
    .sort_values("cohort")
    .reset_index(drop=True)
)

# Cohort-composition figures
#   1) complexity composition
#   2) LSTM: grouped by score type, bars are cohorts
#   3) all models: same grouping style


cohort_order = list(cohort_target_probs.keys())
model_order = ["LSTM", "LR", "RF", "GBDT"]

# Cohort styling for grouped plots
# Same purple hue with different intensity levels for cohorts

purple_base = "#7B1FA2"  # base purple
cohort_facecolors = [
    to_rgba(purple_base, 0.20),  # 10%
    to_rgba(purple_base, 0.40),  # 20%
    to_rgba(purple_base, 0.60),  # 30%
    to_rgba(purple_base, 0.80),  # 40%
    to_rgba(purple_base, 1.0),  # 50%
]
cohort_edgecolor = "#4A148C"

# Figure 1: stacked cohort composition
# tiers remain color-coded
comp_rows = []
for cohort_name, info in sampled_cohort_store.items():
    fr = info["observed_bin_frac"]
    for b, frac in zip([1, 2, 3, 4, 5], fr):
        comp_rows.append({
            "cohort": cohort_name,
            "complexity_bin": f"Q{b}",
            "fraction": float(frac)
        })

comp_df = pd.DataFrame(comp_rows)
comp_pivot = comp_df.pivot(index="cohort", columns="complexity_bin", values="fraction")
comp_pivot = comp_pivot.reindex(cohort_order)
comp_pivot = comp_pivot[[f"Q{i}" for i in [1, 2, 3, 4, 5]]]

fig, ax = plt.subplots(figsize=(8.8, 4.6))
x = np.arange(comp_pivot.shape[0])
bottom = np.zeros(comp_pivot.shape[0])
tier_colors = {
    "Q1": "#50C878",
    "Q2": "#98FF98",
    "Q3": "#00FFFF",
    "Q4": "#4E95D9",
    "Q5": "#512888",
}

for q in comp_pivot.columns:
    vals = comp_pivot[q].to_numpy()
    ax.bar(x, vals, bottom=bottom, label=q, color=tier_colors[q])
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(comp_pivot.index)
ax.set_ylabel("Fraction of patients")
ax.legend(title="Complexity bin", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig('Figs/Figure4_cohort.pdf')

# x-axis groups = plain / CIS-weighted
# bars inside each group = cohorts

# Figure 2-3: LSTM only
plot_one_model_grouped(
    eval_df, model_name="LSTM", fname="Figure4_AUROC.pdf",
    metric_base="auroc", zoom_pad=0.015
)
