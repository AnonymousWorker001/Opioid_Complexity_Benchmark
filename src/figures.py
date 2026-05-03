"""Figure helpers used by the benchmark scripts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _natural_cohort_key(value: str):
    """Sort cohort names such as C1, C2, ..., C10 in numeric order."""
    value = str(value)
    match = re.search(r"(\d+)$", value)
    if match:
        return value[: match.start()], int(match.group(1))
    return value, 0


def _get_cohort_order(eval_df: pd.DataFrame, cohort_order=None):
    """Return a valid cohort order for plotting."""
    available = list(pd.unique(eval_df["cohort"]))

    if cohort_order is None:
        return sorted(available, key=_natural_cohort_key)

    # Keep only cohorts that are actually present, preserving requested order.
    return [c for c in cohort_order if c in set(available)]


def _get_cohort_colors(n_cohorts: int, cohort_facecolors=None):
    """Return enough colors for cohort bars."""
    if cohort_facecolors is not None:
        return cohort_facecolors

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = [None]

    return [color_cycle[i % len(color_cycle)] for i in range(n_cohorts)]


def plot_pe_tier_metrics(metrics_a, metrics_b, label_a, label_b, outpath):
    tiers = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    auroc_a = [metrics_a.get(t, {}).get("auroc", np.nan) for t in tiers]
    auroc_b = [metrics_b.get(t, {}).get("auroc", np.nan) for t in tiers]

    x = np.arange(len(tiers))
    plt.figure(figsize=(8, 4))
    plt.plot(x, auroc_a, marker="o", label=label_a)
    plt.plot(x, auroc_b, marker="o", label=label_b)
    plt.xticks(x, tiers)
    plt.xlabel("Permutation Entropy Tier")
    plt.ylabel("AUROC (week-level)")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_loss_curves(history, title=None, outpath=None, best_epoch=15):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    ax.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation",
        linewidth=2.2,
        marker="s",
        markersize=4,
    )

    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            linestyle=":",
            linewidth=2.0,
            color="black",
            label="Optimal epoch",
        )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Cross-entropy loss", fontsize=14)

    if title is not None:
        ax.set_title(title, fontsize=15)

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_one_model_grouped(
    eval_df,
    model_name,
    fname,
    metric_base="auroc",
    zoom_pad=0.01,
    cohort_order=None,
    cohort_facecolors=None,
    cohort_edgecolor="black",
    out_dir="Figs",
):
    dfm = eval_df[eval_df["model"] == model_name].copy()

    if dfm.empty:
        raise ValueError(f"No rows found for model_name={model_name!r}.")

    cohort_order = _get_cohort_order(dfm, cohort_order)
    cohort_facecolors = _get_cohort_colors(len(cohort_order), cohort_facecolors)

    dfm = dfm.set_index("cohort").loc[cohort_order].reset_index()

    if metric_base == "auroc":
        score_groups = [("Plain", "plain_auroc"), ("CIS-weighted", "cis_auroc")]
        ylab = "AUROC"
    elif metric_base == "auprc":
        score_groups = [("Plain", "plain_auprc"), ("CIS-weighted", "cis_auprc")]
        ylab = "AUPRC"
    else:
        raise ValueError("metric_base must be either 'auroc' or 'auprc'.")

    vals = []
    for _, col in score_groups:
        vals.extend(dfm[col].to_numpy().tolist())

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise ValueError("No finite metric values available for plotting.")

    ymin = max(0.0, float(vals.min()) - zoom_pad)
    ymax = min(1.0, float(vals.max()) + zoom_pad)

    x = np.arange(len(score_groups))
    n_cohorts = len(cohort_order)
    width = 0.82 / n_cohorts

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for j, cohort_name in enumerate(cohort_order):
        heights = [
            dfm.loc[dfm["cohort"] == cohort_name, col].values[0]
            for _, col in score_groups
        ]

        ax.bar(
            x + (j - (n_cohorts - 1) / 2) * width,
            heights,
            width=width,
            label=cohort_name,
            color=cohort_facecolors[j % len(cohort_facecolors)],
            edgecolor=cohort_edgecolor,
            linewidth=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([group_name for group_name, _ in score_groups])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Cohorts", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    outpath = Path(out_dir) / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lstm_grouped_auroc(
    eval_df,
    fname="Figure4_AUROC.pdf",
    zoom_pad=0.01,
    cohort_order=None,
    cohort_facecolors=None,
    cohort_edgecolor="black",
    out_dir="Figs",
):
    dfm = eval_df[eval_df["model"] == "LSTM"].copy()

    if dfm.empty:
        raise ValueError("No rows found for model_name='LSTM'.")

    cohort_order = _get_cohort_order(dfm, cohort_order)
    cohort_facecolors = _get_cohort_colors(len(cohort_order), cohort_facecolors)

    dfm = dfm.set_index("cohort").loc[cohort_order].reset_index()

    vals = np.concatenate(
        [
            dfm["plain_auroc"].to_numpy(dtype=float),
            dfm["cis_auroc"].to_numpy(dtype=float),
        ]
    )
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise ValueError("No finite AUROC values available for plotting.")

    ymin = max(0.0, float(vals.min()) - zoom_pad)
    ymax = min(1.0, float(vals.max()) + zoom_pad)

    x = np.arange(2)  # Plain, CIS-weighted
    n_cohorts = len(cohort_order)
    width = 0.82 / n_cohorts

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for j, cohort_name in enumerate(cohort_order):
        row = dfm.loc[dfm["cohort"] == cohort_name].iloc[0]
        heights = [row["plain_auroc"], row["cis_auroc"]]

        ax.bar(
            x + (j - (n_cohorts - 1) / 2) * width,
            heights,
            width=width,
            label=cohort_name,
            color=cohort_facecolors[j % len(cohort_facecolors)],
            edgecolor=cohort_edgecolor,
            linewidth=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Plain", "CIS-weighted"])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("AUROC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Cohorts", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    outpath = Path(out_dir) / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
