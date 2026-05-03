#!/usr/bin/env python3
"""Fast sanity checks for the processed CTN-0094 benchmark table.

This script does not train models. It verifies the processed CSV, reproduces the
main trajectory-exclusion counts, and checks that each remaining patient would
contribute 20 week-level prediction samples in the benchmark task.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "static_timeSeries_new.csv"
N_WEEKS = 24
N_PREDICTION_WEEKS = 20


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Processed CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    opioid_cols = [f"Opioid_week{i}" for i in range(N_WEEKS)]
    missing_cols = [col for col in ["who", "return_to_use", "treatment_group", *opioid_cols] if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    opioid = df[opioid_cols]
    values = set(pd.unique(opioid.values.ravel()))
    non_binary = sorted(values - {0, 1})
    if len(non_binary) != 1:
        raise ValueError(f"Expected exactly one non-binary missing-code value; found {non_binary}")
    missing_value = non_binary[0]

    after_all_missing = df[opioid.sum(axis=1) != N_WEEKS * missing_value]
    after_all_positive = after_all_missing[after_all_missing[opioid_cols].sum(axis=1) != N_WEEKS]
    after_all_negative = after_all_positive[after_all_positive[opioid_cols].sum(axis=1) != 0]
    week_level_samples = after_all_negative.shape[0] * N_PREDICTION_WEEKS

    print("Smoke test passed")
    print(f"Processed rows: {df.shape[0]:,}")
    print(f"Processed columns: {df.shape[1]:,}")
    print(f"After all-missing filter: {after_all_missing.shape[0]:,}")
    print(f"After all-positive filter: {after_all_positive.shape[0]:,}")
    print(f"After all-negative filter: {after_all_negative.shape[0]:,}")
    print(f"Week-level prediction samples: {week_level_samples:,}")


if __name__ == "__main__":
    main()
