"""Dataset construction and tabular feature helpers."""

from __future__ import annotations

import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


from .complexity import (
    calculate_permutation_entropy,
    compute_prefix_patient_complexity_from_weekly,
    hamming_distance,
)


def generate_dataset_from_dataframe(
    df, miss_value, random_seed=42, future_window=4, n_timesteps=24,
    daily_treatment=True, restrict_length=-1, select_class=None, prev_week_mode=1,
    include_treat_stats=True,
    include_tlstm_treat=True
):

    np.random.seed(random_seed)

    static_features = df[['treatment_group','age', 'race', 'is_male', 'heroin_inject', 'TLFB_Heroin',
                          'TLFB_THC', 'TLFB_Alcohol', 'TLFB_Cocaine', 'TLFB_Methadone',
                          'TLFB_Amphetamine', 'UDS_Benzodiazepine', 'UDS_Opioid',
                          'UDS_Thc', 'UDS_Buprenorphine', 'UDS_Methadone', 'UDS_Cocaine',
                          'UDS_Alcohol', 'UDS_Amphetamine', 'UDS_Sedatives', 'UDS_Mdma/Hallucinogen']].values

    treat_features = df['treat_wks'].tolist()

    # ----- Weekly time-series features -----
    time_series_raw = df.filter(regex='week').values
    time_series_raw[time_series_raw == -1] = miss_value

    N = time_series_raw.shape[0]
    n_orig_features = time_series_raw.shape[1] // n_timesteps
    orig_weekly = (
        time_series_raw
        .reshape(N, n_orig_features, n_timesteps)
        .transpose(0, 2, 1)
    )  # (N, 24, n_orig_features)

    # ----- Optional treatment dose summaries appended to weekly features -----
    if include_treat_stats:
        week_len = 7

        def parse_daily_seq(colname):
            seq_list = df[colname].tolist()
            daily_arr = []
            for s in seq_list:
                if not isinstance(s, str) or s.strip() == "":
                    arr = np.zeros(n_timesteps * week_len, dtype=float)
                else:
                    vals = s.split(',')
                    if len(vals) < n_timesteps * week_len:
                        vals = vals + ['0'] * (n_timesteps * week_len - len(vals))
                    elif len(vals) > n_timesteps * week_len:
                        vals = vals[:n_timesteps * week_len]
                    arr = np.array([float(x) for x in vals], dtype=float)
                daily_arr.append(arr)
            daily_arr = np.stack(daily_arr, axis=0)  # (N, 168)
            return daily_arr.reshape(N, n_timesteps, week_len)  # (N, 24, 7)

        def weekly_stats(daily_3d):
            wk_min  = daily_3d.min(axis=2, keepdims=True)
            wk_max  = daily_3d.max(axis=2, keepdims=True)
            wk_mean = daily_3d.mean(axis=2, keepdims=True)
            return np.concatenate([wk_min, wk_max, wk_mean], axis=2)  # (N, 24, 3)

        bup_stats = weekly_stats(parse_daily_seq('treat_Buprenorphine_amt'))
        met_stats = weekly_stats(parse_daily_seq('treat_Methadone_amt'))
        ntx_stats = weekly_stats(parse_daily_seq('treat_Naltrexone_amt'))

        treat_weekly_stats = np.concatenate([bup_stats, met_stats, ntx_stats], axis=2)  # (N, 24, 9)
        time_series_features = np.concatenate([orig_weekly, treat_weekly_stats], axis=2)
    else:
        time_series_features = orig_weekly

    # --- Variable-length sample creation ---
    def create_variable_length_sequences_with_labels(static_features, time_series_features, treat_features):
        X_static_list, X_time_series_list, y_list = [], [], []
        sample_index, week_index, T_time_series_list = [], [], []
        week_len = 7
        start_week = 3

        for i in range(len(static_features)):
            treat_rec = np.array([float(ele) for ele in treat_features[i].split(',')])

            for t in range(start_week, n_timesteps - future_window):
                if (select_class is not None) and (time_series_features[i, t, 0] != select_class):
                    continue
                if restrict_length != -1 and (t + 1 != restrict_length):
                    continue

                # TLSTM treatment input: ONLY if include_tlstm_treat
                if include_tlstm_treat:
                    treat_ts = treat_rec.reshape(n_timesteps, week_len)
                    if prev_week_mode == 1:
                        T_time_series_list.append(treat_ts[:t+1, :])
                    elif prev_week_mode == 2:
                        T_time_series_list.append(treat_ts[t-3:t+1, :])
                    elif prev_week_mode == 3:
                        temp_vec = treat_ts[:t+1, :].copy()
                        temp_vec[:-3] = -3
                        T_time_series_list.append(temp_vec)

                X_static_list.append(static_features[i])

                if prev_week_mode == 1:
                    X_time_series_list.append(time_series_features[i, :t+1, :])
                    week_index.append(t)
                elif prev_week_mode == 2:
                    X_time_series_list.append(time_series_features[i, t-3:t+1, :])
                    week_index.append(3)
                elif prev_week_mode == 3:
                    temp_vec = time_series_features[i, :t+1, :].copy()
                    temp_vec[:-3] = -3
                    X_time_series_list.append(temp_vec)
                    week_index.append(t)

                sample_index.append(i)

                labels = time_series_features[i, t+1, 0]
                labels = 1 if labels != 0 else 0
                y_list.append(labels)

        return X_static_list, X_time_series_list, y_list, sample_index, week_index, T_time_series_list

    X_static, X_time_series, y, sample_index, week_index, T_time_series = create_variable_length_sequences_with_labels(
        static_features, time_series_features, treat_features
    )

    # tensors + padding
    X_static = torch.tensor(X_static, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # better: long for CrossEntropy
    X_time_series = [torch.tensor(seq, dtype=torch.float32) for seq in X_time_series]
    X_time_series = nn.utils.rnn.pad_sequence(X_time_series, batch_first=True)
    sample_index = torch.tensor(sample_index, dtype=torch.int64)
    week_index = torch.tensor(week_index, dtype=torch.int64)

    if include_tlstm_treat:
        T_time_series = [torch.tensor(seq, dtype=torch.float32) for seq in T_time_series]
        T_time_series = nn.utils.rnn.pad_sequence(T_time_series, batch_first=True)
        dataset = TensorDataset(X_time_series, X_static, y, sample_index, week_index, T_time_series)
    else:
        dataset = TensorDataset(X_time_series, X_static, y, sample_index, week_index)

    return dataset


def split_train_test_stratify_permutation_entropy(csv_file='data/processed/static_timeSeries_new.csv', test_ratio = 0.2, val_ratio=None, bins=5, no_filter=False):
######################## Function to generate traing, validation, test, test_shuffle data sets
    random_seed = 42
    n_timesteps = 24  # number of time points

    df = pd.read_csv(csv_file)

    start_idx = df.columns.tolist().index('Opioid_week0')
    if df.columns[start_idx+n_timesteps-1] != 'Opioid_week23':
        print("The columns of original data is not correct! Please check the static_timeSeries_new.csv file!")
        return
    print(f"Raw data: {df.shape[0]}")
    miss_origin = list(set(np.unique(df.iloc[:,start_idx:(start_idx+n_timesteps)].values))-set([0,1]))[0]
    df_miss = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)==n_timesteps*miss_origin]
    df_positive = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)==n_timesteps] # remove all positive
    df_negative = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)==0] # remove all negative

    df = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)!=n_timesteps*miss_origin]
    print(f"Filter all missing: {df.shape[0]}")
    df = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)!=n_timesteps] # remove all positive
    print(f"Filter all positive: {df.shape[0]}")
    df = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)!=0] # remove all negative
    print(f"Filter all negative: {df.shape[0]}")

    perm_ent = []
    for i in range(df.shape[0]):
        perm_ent.append(calculate_permutation_entropy(df.iloc[i,start_idx:(start_idx+n_timesteps)]))

    perm_ent = np.array(perm_ent)
    perm_ent_lv = np.zeros_like(perm_ent)

    res = plt.hist(perm_ent,bins=5)
    plt.close()# Do not show the plot

    bins_res = res[1]
    for bin in bins_res[:-1]:
        perm_ent_lv[perm_ent>=bin] += 1

    df['Target_Permutation_Entropy'] = perm_ent_lv
    print(np.unique(perm_ent_lv))

    if val_ratio is not None:
        val_size = int(df.shape[0]*val_ratio)
    else:
        val_size = int(df.shape[0]*test_ratio)

    test_size = int(df.shape[0]*test_ratio)

    df_rem, df_test = train_test_split(
        df,
        test_size=test_size,  # specifying the exact size for the test set
        stratify=df['Target_Permutation_Entropy'].astype(str),
        random_state=random_seed  # for reproducibility
    )
    df_rem = df_rem.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if no_filter:
        df_rem = pd.concat([df_rem, df_positive, df_negative], ignore_index=True)

    df_train, df_val = train_test_split(
        df_rem,
        test_size=val_size,  # specifying the exact size for the test set
        stratify=df_rem['Target_Permutation_Entropy'].astype(str),
        random_state=random_seed  # for reproducibility
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # process temporal-shuffled test
    df_test_random = df_test.copy()
    random.seed(random_seed)
    rand_index = list(range(n_timesteps))
    random.shuffle(rand_index)
    rand_index = [ele+n_timesteps for ele in rand_index]
    for i in range(df_test_random.shape[0]):
        df_test_random.iloc[i,start_idx:(start_idx+n_timesteps)] = df_test_random.iloc[i,rand_index]

    perm_ent_shuf = []
    for i in range(df_test.shape[0]):
        perm_ent_shuf.append(hamming_distance(df_test.iloc[i,start_idx:(start_idx+n_timesteps)],df_test_random.iloc[i,start_idx:(start_idx+n_timesteps)]))

    perm_ent_shuf = np.array(perm_ent_shuf)
    perm_ent_shuf_lv = np.zeros_like(perm_ent_shuf)
    if bins == 5:
        bins_shuf = [0, 0.2, 0.4, 0.6, 0.8, 1]
    elif bins == 4:
        bins_shuf = [0, 0.2, 0.4, 0.6, 1]
    for bin in bins_shuf[:-1]:
        perm_ent_shuf_lv[perm_ent_shuf>=bin] += 1
    df_test_random['Target_Permutation_Entropy'] = perm_ent_shuf_lv

    return df_rem, df_val, df_test, df_train, bins_res


def split_train_test_stratify_prefix_entropy(csv_file='data/processed/static_timeSeries_new.csv', test_ratio = 0.2, val_ratio=None, bins=5, no_filter=False):
######################## Function to generate traing, validation, test, test_shuffle data sets
    random_seed = 42
    n_timesteps = 24  # number of time points

    df = pd.read_csv(csv_file)

    start_idx = df.columns.tolist().index('Opioid_week0')
    if df.columns[start_idx+n_timesteps-1] != 'Opioid_week23':
        print("The columns of original data is not correct! Please check the static_timeSeries_new.csv file!")
        return
    print(f"Raw data: {df.shape[0]}")
    miss_origin = list(set(np.unique(df.iloc[:,start_idx:(start_idx+n_timesteps)].values))-set([0,1]))[0]
    df_miss = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)==n_timesteps*miss_origin]
    df_positive = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)==n_timesteps] # remove all positive
    df_negative = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)==0] # remove all negative

    df = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)!=n_timesteps*miss_origin]
    print(f"Filter all missing: {df.shape[0]}")
    df = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)!=n_timesteps] # remove all positive
    print(f"Filter all positive: {df.shape[0]}")
    df = df[df.iloc[:,start_idx:(start_idx+n_timesteps)].sum(axis=1)!=0] # remove all negative
    print(f"Filter all negative: {df.shape[0]}")

    # Prefix-complexity summary used for benchmark split construction.
    perm_ent = compute_prefix_patient_complexity_from_weekly(
        df,
        opioid_prefix="Opioid_week",
        n_timesteps=n_timesteps,
        start_week=3,
        future_window=1,
        summary="mean",
    )

    perm_ent = np.array(perm_ent, dtype=float)
    perm_ent_lv = np.zeros_like(perm_ent)

    res = plt.hist(perm_ent,bins=bins)
    plt.close()

    bins_res = res[1]
    for bin in bins_res[:-1]:
        perm_ent_lv[perm_ent>=bin] += 1

    df['Target_Permutation_Entropy'] = perm_ent_lv.astype(int)
    print(np.unique(perm_ent_lv))

    if val_ratio is not None:
        val_size = int(df.shape[0]*val_ratio)
    else:
        val_size = int(df.shape[0]*test_ratio)

    test_size = int(df.shape[0]*test_ratio)

    df_rem, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['Target_Permutation_Entropy'].astype(str),
        random_state=random_seed
    )
    df_rem = df_rem.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if no_filter:
        df_rem = pd.concat([df_rem, df_positive, df_negative], ignore_index=True)

    df_train, df_val = train_test_split(
        df_rem,
        test_size=val_size,
        stratify=df_rem['Target_Permutation_Entropy'].astype(str),
        random_state=random_seed
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # process temporal-shuffled test
    df_test_random = df_test.copy()
    random.seed(random_seed)
    rand_index = list(range(n_timesteps))
    random.shuffle(rand_index)
    rand_index = [ele+n_timesteps for ele in rand_index]
    for i in range(df_test_random.shape[0]):
        df_test_random.iloc[i,start_idx:(start_idx+n_timesteps)] = df_test_random.iloc[i,rand_index]

    perm_ent_shuf = []
    for i in range(df_test.shape[0]):
        perm_ent_shuf.append(hamming_distance(df_test.iloc[i,start_idx:(start_idx+n_timesteps)],df_test_random.iloc[i,start_idx:(start_idx+n_timesteps)]))

    perm_ent_shuf = np.array(perm_ent_shuf)
    perm_ent_shuf_lv = np.zeros_like(perm_ent_shuf)
    if bins == 5:
        bins_shuf = [0, 0.2, 0.4, 0.6, 0.8, 1]
    elif bins == 4:
        bins_shuf = [0, 0.2, 0.4, 0.6, 1]
    else:
        bins_shuf = np.linspace(0, 1, bins + 1)
    for bin in bins_shuf[:-1]:
        perm_ent_shuf_lv[perm_ent_shuf>=bin] += 1
    df_test_random['Target_Permutation_Entropy'] = perm_ent_shuf_lv.astype(int)

    return df_rem, df_val, df_test, df_test_random, bins_res


def _get_weekly_prefixes(df, n_timesteps=24):
    prefixes = []
    for c in df.columns:
        m = re.match(r"(.+)_week(\d+)$", c)
        if m and int(m.group(2)) == 0:
            prefix = m.group(1)
            ok = all(f"{prefix}_week{i}" in df.columns for i in range(n_timesteps))
            if ok:
                prefixes.append(prefix)
    prefixes = sorted(prefixes)
    if "Opioid" in prefixes:
        prefixes.remove("Opioid")
        prefixes = ["Opioid"] + prefixes
    return prefixes


def build_week_level_tabular(df, n_timesteps=24, window=4, start_week=3,
                             include_treat=False, include_dose=False):
    """
    Fixed-length week-level dataset:
      - features from weeks (t-window+1..t)
      - label: opioid at t+1 (non-zero => 1; 0 => 0)
    Returns: X, y, pid_idx, week_idx
    """
    prefixes = _get_weekly_prefixes(df, n_timesteps=n_timesteps)

    static_cols = [
        "treatment_group","age","race","is_male","heroin_inject",
        "TLFB_Heroin","TLFB_THC","TLFB_Alcohol","TLFB_Cocaine","TLFB_Methadone","TLFB_Amphetamine",
        "UDS_Benzodiazepine","UDS_Opioid","UDS_Thc","UDS_Buprenorphine","UDS_Methadone","UDS_Cocaine",
        "UDS_Alcohol","UDS_Amphetamine","UDS_Sedatives","UDS_Mdma/Hallucinogen"
    ]
    static_cols = [c for c in static_cols if c in df.columns]
    X_static = df[static_cols].astype(float).values

    week_len = 7

    def _parse_daily(colname):
        if colname not in df.columns:
            return None
        out = []
        need = n_timesteps * week_len
        for v in df[colname].values:
            # robust handling of NaN / None / empty
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals = ["0"] * need
            else:
                s = str(v).strip()
                if s == "" or s.lower() == "nan" or s.lower() == "none":
                    vals = ["0"] * need
                else:
                    vals = s.split(",")
            vals = (vals + ["0"] * max(0, need - len(vals)))[:need]
            arr = np.array([float(x) if str(x).strip() not in ("", "nan", "None") else 0.0 for x in vals], dtype=float)
            out.append(arr.reshape(n_timesteps, week_len))
        return np.stack(out, axis=0)  # (N, 24, 7)

    treat_days = _parse_daily("treat_wks") if include_treat else None
    bup = _parse_daily("treat_Buprenorphine_amt") if include_dose else None
    met = _parse_daily("treat_Methadone_amt") if include_dose else None
    ntx = _parse_daily("treat_Naltrexone_amt") if include_dose else None

    X_list, y_list, pid_list, week_list = [], [], [], []
    n_pat = df.shape[0]

    # Pre-fetch weekly columns for speed (optional but helps)
    weekly_cols_by_prefix = {p: [f"{p}_week{i}" for i in range(n_timesteps)] for p in prefixes}

    for pid in range(n_pat):
        for t in range(start_week, n_timesteps - 1):
            feats = []

            # windowed weekly history features
            for prefix in prefixes:
                seq = df.loc[pid, weekly_cols_by_prefix[prefix]].astype(float).values
                feats.extend(seq[t - window + 1:t + 1].tolist())

            # static
            feats.extend(X_static[pid].tolist())

            # optional: treatment summaries at week t
            if include_treat and treat_days is not None:
                feats.append(float((treat_days[pid, t] > 0).mean()))  # fraction of days with any tx

            if include_dose:
                feats.append(float(bup[pid, t].mean()) if bup is not None else 0.0)
                feats.append(float(met[pid, t].mean()) if met is not None else 0.0)
                feats.append(float(ntx[pid, t].mean()) if ntx is not None else 0.0)

            # label from next week opioid status (non-zero => 1)
            y_next = float(df.loc[pid, f"Opioid_week{t+1}"])
            y = 1 if y_next != 0 else 0

            X_list.append(feats)
            y_list.append(y)
            pid_list.append(pid)
            week_list.append(t)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=int)
    pid_idx = np.asarray(pid_list, dtype=int)
    week_idx = np.asarray(week_list, dtype=int)
    return X, y, pid_idx, week_idx


def enumerate_prediction_samples(df, n_timesteps=24, future_window=1, start_week=3):
    patient_ids = []
    actual_weeks = []
    for pid in range(df.shape[0]):
        for t in range(start_week, n_timesteps - future_window):
            patient_ids.append(pid)
            actual_weeks.append(t)
    return np.asarray(patient_ids, dtype=int), np.asarray(actual_weeks, dtype=int)
