"""Model definitions, training loops, and prediction collectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from .complexity import compute_patient_complexity_from_weekly, compute_cis_weights, compute_sample_prefix_cis_weights
from .metrics import compute_metrics_binary

class TimeDependentLSTM(nn.Module):
    def __init__(self, input_size1, hidden_size, num_static_features, input_size2=7):
        super(TimeDependentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size1, hidden_size, batch_first=True)
        self.fc_time_series = nn.Linear(hidden_size, 32)
        self.tlstm = nn.LSTM(input_size2, hidden_size, batch_first=True)
        self.fc_t_time_series = nn.Linear(hidden_size, 32)
        self.fc_static = nn.Linear(num_static_features, 32)
        self.fc_combined_all = nn.Linear(96, 16)
        self.fc_combined = nn.Linear(64, 16)
        self.fc_single = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x_time_series, x_static, t_time_series):
        lstm_out, _ = self.lstm(x_time_series)
        x_time_series = torch.relu(self.fc_time_series(lstm_out))

        if (not x_static is None) and (not t_time_series is None):
            # Expand static features across the time axis
            x_static_expanded = x_static.unsqueeze(1).expand(-1, x_time_series.size(1), -1)
            x_static = torch.relu(self.fc_static(x_static_expanded))

            tlstm_out, _ = self.tlstm(t_time_series)
            t_time_series = torch.relu(self.fc_t_time_series(tlstm_out))

            # Combine time-series and static features
            combined = torch.cat((x_time_series, t_time_series, x_static), dim=2)
            combined = torch.relu(self.fc_combined_all(combined))

            output = self.output(combined).squeeze(-1)
        elif not x_static is None:
            # Expand static features across the time axis
            x_static_expanded = x_static.unsqueeze(1).expand(-1, x_time_series.size(1), -1)
            x_static = torch.relu(self.fc_static(x_static_expanded))
            # Combine time-series and static features
            combined = torch.cat((x_time_series, x_static), dim=2)
            combined = torch.relu(self.fc_combined(combined))
            output = self.output(combined).squeeze(-1)
        elif not t_time_series is None:
            tlstm_out, _ = self.tlstm(t_time_series)
            t_time_series = torch.relu(self.fc_t_time_series(tlstm_out))
            combined = torch.cat((x_time_series, t_time_series), dim=2)
            combined = torch.relu(self.fc_combined(combined))
            output = self.output(combined).squeeze(-1)
        else:
            single = torch.relu(self.fc_single(x_time_series))
            output = self.output(single).squeeze(-1)

        return output


def load_LSTM_model_pars(train_dataset, hidden_size=64, lr=0.0005, device="cpu"):
    model = TimeDependentLSTM(
        train_dataset[0][0].shape[1],
        hidden_size,
        train_dataset[0][1].shape[0]
    ).to(device)

    classes = np.unique(train_dataset[:][2].numpy())
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=train_dataset[:][2].numpy()
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def fit_model(model, train_loader, val_loader, criterion, optimizer, no_static=False, include_tlstm_treat=False, n_epochs=15, device="cpu", return_history=False):
# def fit_model(model, train_loader, val_loader, criterion, optimizer, no_static=False, include_tlstm_treat=False, n_epochs=15, device="cpu"):
    model.to(device)
    best_val = -1
    best_state = None
    best_epoch = -1

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        val_loss_sum = 0.0
        val_n = 0
        for batch in train_loader:
            if include_tlstm_treat:
                x_ts, x_static, y, sample_idx, week_index, t_ts = batch
                t_ts = t_ts.to(device)
            else:
                x_ts, x_static, y, sample_idx, week_index = batch
                t_ts = None

            x_ts = x_ts.to(device)
            x_static = x_static.to(device)
            y = y.to(device)
            week_index = week_index.to(device)

            optimizer.zero_grad()
            logits_all = model(x_ts, None if no_static else x_static, t_ts)
            logits = logits_all[torch.arange(logits_all.size(0), device=device), week_index, :]
            loss = criterion(logits, y)
            train_loss_sum += loss.item() * y.size(0)
            train_n += y.size(0)
            loss.backward()
            optimizer.step()

        # quick val AUROC
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if include_tlstm_treat:
                    x_ts, x_static, y, sample_idx, week_index, t_ts = batch
                    t_ts = t_ts.to(device)
                else:
                    x_ts, x_static, y, sample_idx, week_index = batch
                    t_ts = None

                x_ts = x_ts.to(device)
                x_static = x_static.to(device)
                y = y.to(device)
                week_index = week_index.to(device)

                logits_all = model(x_ts, None if no_static else x_static, t_ts)
                logits = logits_all[torch.arange(logits_all.size(0), device=device), week_index, :]
                val_loss = criterion(logits, y)

                val_loss_sum += val_loss.item() * y.size(0)
                val_n += y.size(0)

        val_metrics = evaluate_week_level(model, val_loader, no_static=no_static, include_tlstm_treat=include_tlstm_treat, device=device)
        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        print(f"[epoch {epoch+1}] val AUROC={val_metrics['auroc']:.3f} val F1={val_metrics['f1']:.3f}")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss_sum / max(train_n, 1))
        history["val_loss"].append(val_loss_sum / max(val_n, 1))

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch


def evaluate_week_level(model, loader, no_static=False, include_tlstm_treat=False, device="cpu"):
    model.eval()
    ys = []
    ps = []

    with torch.no_grad():
        for batch in loader:
            if include_tlstm_treat:
                x_ts, x_static, y, sample_idx, week_index, t_ts = batch
                t_ts = t_ts.to(device)
            else:
                x_ts, x_static, y, sample_idx, week_index = batch
                t_ts = None

            x_ts = x_ts.to(device)
            x_static = x_static.to(device)
            y = y.to(device)
            week_index = week_index.to(device)

            logits_all = model(x_ts, None if no_static else x_static, t_ts)
            logits = logits_all[torch.arange(logits_all.size(0), device=device), week_index, :]
            prob = F.softmax(logits, dim=1)[:, 1]  # P(relapse)

            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)

    auroc = roc_auc_score(y_all, p_all)
    auprc = average_precision_score(y_all, p_all)

    pred = (p_all >= 0.5).astype(int)
    f1 = f1_score(y_all, pred)

    return {"auroc": auroc, "auprc": auprc, "f1": f1}


def evaluate_by_pe_tier(model, df_test, test_dataset, no_static=False, include_tlstm_treat=False, device="cpu"):
    """
    df_test rows correspond to patients; test_dataset contains many (patient, week) samples.
    We'll use sample_index to map week-level samples back to patient PE tier.
    """
    loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    model.eval()

    tiers = df_test["Target_Permutation_Entropy"].values
    tier_y = {}
    tier_p = {}

    with torch.no_grad():
        for batch in loader:
            if include_tlstm_treat:
                x_ts, x_static, y, sample_idx, week_index, t_ts = batch
                t_ts = t_ts.to(device)
            else:
                x_ts, x_static, y, sample_idx, week_index = batch
                t_ts = None

            x_ts = x_ts.to(device)
            x_static = x_static.to(device)
            week_index = week_index.to(device)

            logits_all = model(x_ts, None if no_static else x_static, t_ts)
            logits = logits_all[torch.arange(logits_all.size(0), device=device), week_index, :]
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            sidx = sample_idx.detach().cpu().numpy()

            for yi, pi, si in zip(y_np, p, sidx):
                tier = int(tiers[int(si)])
                tier_y.setdefault(tier, []).append(int(yi))
                tier_p.setdefault(tier, []).append(float(pi))

    out = {}
    for tier in sorted(tier_y.keys()):
        yv = np.array(tier_y[tier])
        pv = np.array(tier_p[tier])
        if len(np.unique(yv)) < 2:
            out[tier] = {"n": len(yv), "auroc": np.nan, "auprc": np.nan}
        else:
            out[tier] = {
                "n": len(yv),
                "auroc": roc_auc_score(yv, pv),
                "auprc": average_precision_score(yv, pv),
            }
    return out


def collect_week_level_predictions(model, dataset, include_tlstm_treat=False, device="cpu"):
    """Collect y_true, y_prob, and sample_index from a TensorDataset."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    model.to(device)
    ys, ps, sidxs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if include_tlstm_treat:
                x_ts, x_static, y, sample_idx, week_idx, t_ts = batch
            else:
                x_ts, x_static, y, sample_idx, week_idx = batch
                t_ts = None
            x_ts = x_ts.to(device)
            x_static = x_static.to(device)
            y = y.long().to(device)
            sample_idx = sample_idx.to(device)
            week_idx = week_idx.to(device)
            if t_ts is not None:
                t_ts = t_ts.to(device)
            out = model(x_ts, x_static, t_ts)
            b = torch.arange(out.size(0), device=device)
            out = out[b, week_idx, :]
            prob = F.softmax(out, dim=1)[:, 1]
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())
            sidxs.append(sample_idx.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(sidxs)


def cis_weighted_metrics_for_dataset(model, df_train_ref, df_eval, dataset_eval, include_tlstm_treat=False, device="cpu"):
    y_true, y_prob, sample_idx = collect_week_level_predictions(model, dataset_eval, include_tlstm_treat=include_tlstm_treat, device=device)
    pe_train = compute_patient_complexity_from_weekly(df_train_ref)
    pe_eval = compute_patient_complexity_from_weekly(df_eval)
    cis_eval = compute_cis_weights(df_eval)
    w = cis_eval[sample_idx.astype(int)]
    return compute_metrics_binary(y_true, y_prob, sample_weight=w)


def collect_week_level_probs_with_pid(model, loader, include_tlstm_treat, device="cpu"):
    """
    Collect week-level predictions aligned with labels AND patient ids (sample_idx).
    Handles y shapes (B,) or (B,1).
    """
    model.eval()
    ys, ps, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x_ts, x_static, y, sample_idx, week_index = batch
                t_ts = None
            else:
                x_ts, x_static, y, sample_idx, week_index, t_ts = batch

            x_ts = x_ts.to(device)
            x_static = x_static.to(device)
            y = y.to(device)
            sample_idx = sample_idx.to(device)
            week_index = week_index.to(device)
            if include_tlstm_treat and t_ts is not None:
                t_ts = t_ts.to(device)

            out = model(x_ts, x_static, t_ts if include_tlstm_treat else None)
            bi = torch.arange(out.size(0), device=device)
            out = out[bi, week_index, :]  # (B, 2)
            prob_pos = F.softmax(out, dim=1)[:, 1]  # P(y=1)

            y_np = y.detach().cpu().numpy().reshape(-1).astype(int)
            ys.append(y_np)
            ps.append(prob_pos.detach().cpu().numpy().reshape(-1))
            pids.append(sample_idx.detach().cpu().numpy().reshape(-1).astype(int))

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    pid_all = np.concatenate(pids)
    assert len(y_all) == len(p_all) == len(pid_all)
    return y_all, p_all, pid_all


def collect_week_level_predictions_with_week(model, dataset, include_tlstm_treat=False, device="cpu"):
    """Collect y_true, y_prob, sample_index, and week_index from a TensorDataset."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    model.to(device)
    ys, ps, sidxs, widxs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if include_tlstm_treat:
                x_ts, x_static, y, sample_idx, week_idx, t_ts = batch
            else:
                x_ts, x_static, y, sample_idx, week_idx = batch
                t_ts = None
            x_ts = x_ts.to(device)
            x_static = x_static.to(device)
            y = y.long().to(device)
            sample_idx = sample_idx.to(device)
            week_idx = week_idx.to(device)
            if t_ts is not None:
                t_ts = t_ts.to(device)
            out = model(x_ts, x_static, t_ts)
            b = torch.arange(out.size(0), device=device)
            out = out[b, week_idx, :]
            prob = F.softmax(out, dim=1)[:, 1]
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())
            sidxs.append(sample_idx.detach().cpu().numpy())
            widxs.append(week_idx.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(sidxs), np.concatenate(widxs)


def prefix_cis_weighted_metrics_for_dataset(model, df_train_ref, df_eval, dataset_eval, include_tlstm_treat=False, device="cpu"):
    y_true, y_prob, sample_idx, week_idx = collect_week_level_predictions_with_week(
        model, dataset_eval, include_tlstm_treat=include_tlstm_treat, device=device
    )
    w, prefix_pe = compute_sample_prefix_cis_weights(
        df_train_ref, df_eval, sample_idx, week_idx
    )
    return compute_metrics_binary(y_true, y_prob, sample_weight=w)
