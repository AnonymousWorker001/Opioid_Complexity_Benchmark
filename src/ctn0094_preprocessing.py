#!/usr/bin/env python3
"""Build the analysis-ready CTN-0094 benchmark table.

This script recreates ``data/processed/static_timeSeries_new.csv`` from the
public CTN-0094 release plus the weekly UDS Excel file included in this
repository. It is the command-line version of the preprocessing notebook, but
is written as a normal Python script so the data-generation logic can be read
and reviewed without notebook cell state.

The script does not redistribute the raw CTN-0094 ``.rda`` files. To run it,
place the source tables from ``public.ctn0094data-main`` and
``public.ctn0094data-extra`` under ``data/source/public.ctn0094data-main/data``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyreadr
except ImportError:  # optional until full preprocessing is run
    pyreadr = None


WEEK_DAYS = 7
N_WEEKS = 24
FOLLOWUP_DAYS = WEEK_DAYS * N_WEEKS

OPIOID_SUBSTANCES = {
    "Morphine",
    "Oxycodone",
    "Fentanyl",
    "Opioid",
    "Heroin",
    "Opium",
    "Buprenorphine",
    "Methadone",
    "Hydromorphone",
    "Hydrocodone",
    "Tramadol",
    "Propoxyphene",
    "Oxymorphone",
    "Codeine",
    "Merperidine",
    "Nalbuphine",
}

PRESCRIBED_OPIOIDS = {
    "Outpatient BUP + EMM": "Buprenorphine",
    "Outpatient BUP + SMM": "Buprenorphine",
    "Outpatient BUP": "Buprenorphine",
    "Inpatient BUP": "Buprenorphine",
    "Methadone": "Methadone",
    "Inpatient NR-NTX": "Naltrexone",
}

CANONICAL_MOUD_TYPES = ["Buprenorphine", "Methadone", "Naltrexone"]

WEEKLY_DRUG_NAME_MAP = {
    "o": "Opioid",
    "amp": "Amphetamine",
    "ben": "Benzodiazepine",
    "bup": "Buprenorphine",
    "can": "Cannabis",
    "coc": "Cocaine",
    "her": "Heroin",
    "met": "Methadone",
    "methamp": "Methamphetamine",
    "oxy": "Oxycodone",
    "propoxy": "Propoxyphene",
}

STATIC_COLUMNS = [
    "who",
    "return_to_use",
    "treatment_group",
    "treat_wks",
    "age",
    "race",
    "is_male",
    "heroin_inject",
]

REQUIRED_RDA_TABLES = {
    "everybody",
    "treatment",
    "all_drugs",
    "uds",
    "uds_temp",
    "visit",
    "tlfb",
    "randomization",
    "demographics",
    "rbs_iv",
}

OPTIONAL_RDA_TABLES = {
    "withdrawal",
    "screening_date",
    "outcomesCTN0094",
    "site_masked",
}


def repo_root() -> Path:
    """Return the repository root when called from root or scripts/."""
    current = Path.cwd().resolve()
    if (current / "data").exists():
        return current
    if (current.parent / "data").exists():
        return current.parent
    return current


def load_rda_tables(source_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load every ``.rda`` table in ``source_dir`` into a dictionary.

    The CTN release stores each table in a separate R data file. For the files
    used here, the filename stem is the table name expected downstream.
    """
    if not source_dir.exists():
        raise FileNotFoundError(
            f"CTN source directory not found: {source_dir}\n"
            "Download public.ctn0094data-main and public.ctn0094data-extra, "
            "then place their .rda files in this folder."
        )

    if pyreadr is None:
        raise ImportError(
            "pyreadr is required for preprocessing from CTN .rda files. "
            "Install repository dependencies with `pip install -r requirements.txt`."
        )

    tables: Dict[str, pd.DataFrame] = {}
    for path in sorted(source_dir.glob("*.rda")):
        result = pyreadr.read_r(str(path))
        if not result:
            continue
        # The CTN files used in this project contain one object per .rda file.
        tables[path.stem] = next(iter(result.values()))

    missing = sorted(REQUIRED_RDA_TABLES - set(tables))
    if missing:
        raise FileNotFoundError(
            "Missing required CTN source tables: "
            + ", ".join(missing)
            + f"\nChecked folder: {source_dir}"
        )

    return tables


def patient_randomization_row(
    patient_id: int,
    everybody: pd.DataFrame,
    randomization: pd.DataFrame,
) -> Optional[pd.Series]:
    """Return the randomization row used as treatment entry for one patient.

    Project 30 has a phase-I randomization stage that was removed to match the
    prior CTN analysis convention used in this work.
    """
    project_values = everybody.loc[everybody["who"] == patient_id, "project"].values
    if len(project_values) == 0:
        return None

    rows = randomization[(randomization["who"] == patient_id) & randomization["when"].notna()].copy()
    if project_values[0] == "30" and "which" in rows.columns:
        rows = rows[rows["which"] != "1"]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_treatment_entry_maps(
    tables: Mapping[str, pd.DataFrame],
    miss_flag: str = "visit",
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, str], Dict[int, int]]:
    """Construct weekly opioid outcome strings and treatment-entry maps.

    Returns
    -------
    outcomes:
        Patient-level 24-character strings using ``+`` for non-prescribed opioid
        positive, ``-`` for negative, ``o`` for missing, and ``*`` for mixed
        weekly evidence.
    return_to_use:
        Indicator for at least four consecutive risk-positive weeks between
        weeks 3 and 12 after randomization.
    treatment_entry:
        Randomization day used to align treatment and follow-up time.
    prescribed_opioid:
        Medication expected from treatment assignment and removed from the
        non-prescribed opioid urine-screen target.
    """
    everybody = tables["everybody"]
    randomization = tables["randomization"]
    uds = tables["uds"]
    uds_temp = tables["uds_temp"]
    visit = tables["visit"]

    outcomes: Dict[int, str] = {}
    return_to_use: Dict[int, int] = {}
    treatment_entry: Dict[int, int] = {}
    prescribed_opioid: Dict[int, str] = {}

    cnt_pos = 0
    cnt_neg = 0

    for patient_id in randomization["who"].dropna().unique():
        pid = int(patient_id)
        rand_row = patient_randomization_row(pid, everybody, randomization)
        if rand_row is None:
            continue

        treatment_name = rand_row["treatment"]
        if treatment_name not in PRESCRIBED_OPIOIDS:
            continue

        random_day = int(rand_row["when"])
        range_start = random_day
        range_stop = random_day + FOLLOWUP_DAYS
        output_start = random_day + WEEK_DAYS * 3
        output_stop = random_day + WEEK_DAYS * 12

        prescribed = PRESCRIBED_OPIOIDS[treatment_name]
        prescribed_opioid[pid] = prescribed
        treatment_entry[pid] = random_day
        return_to_use[pid] = 0

        weekly_states: List[str] = []
        consecutive_risk_weeks = 0

        for week_start_day in range(range_start, range_stop, WEEK_DAYS):
            start = week_start_day + 1
            stop = min(week_start_day + WEEK_DAYS + 1, range_stop + 1)
            in_return_window = output_start <= week_start_day < output_stop

            visits_this_week = visit[
                (visit["who"] == pid)
                & (visit["when"] >= start)
                & (visit["when"] < stop)
                & (visit["what"].isin(["visit", "final"]))
            ]
            uds_ok_this_week = uds_temp[
                (uds_temp["who"] == pid)
                & (uds_temp["when"] >= start)
                & (uds_temp["when"] < stop)
                & (uds_temp["was_temp_ok"] == "1")
            ]
            uds_this_week = uds[
                (uds["who"] == pid)
                & (uds["when"] >= start)
                & (uds["when"] < stop)
                & (uds["what"] != prescribed)
            ]

            observed_count = len(uds_ok_this_week)
            if miss_flag == "visit":
                observed_count += len(visits_this_week)

            if observed_count == 0:
                weekly_state = "o"
            else:
                observed_days = uds_ok_this_week["when"].tolist()
                if miss_flag == "visit":
                    observed_days += visits_this_week["when"].tolist()
                observed_days = sorted(set(observed_days))

                daily_states = []
                for day in observed_days:
                    detected = set(uds_this_week.loc[uds_this_week["when"] == day, "what"].values)
                    daily_states.append("+" if detected & OPIOID_SUBSTANCES else "-")

                if not daily_states:
                    weekly_state = "-"
                elif len(set(daily_states)) > 1:
                    weekly_state = "*"
                else:
                    weekly_state = daily_states[0]

            weekly_states.append(weekly_state)

            if in_return_window:
                if weekly_state in {"+", "o", "*"}:
                    consecutive_risk_weeks += 1
                else:
                    consecutive_risk_weeks = 0
                if consecutive_risk_weeks >= 4:
                    return_to_use[pid] = 1

        outcomes[pid] = "".join(weekly_states)
        if return_to_use[pid] == 1:
            cnt_pos += 1
        else:
            cnt_neg += 1

    print(f"Return-to-use labels from weekly UDS logic: {cnt_pos} positive, {cnt_neg} negative")
    return outcomes, return_to_use, treatment_entry, prescribed_opioid


def four_week_return_label(ctn94wk: pd.DataFrame, patient_ids: Iterable[int]) -> List[int]:
    """Replicate the notebook's four-consecutive-risk-week label from Excel."""
    labels: List[int] = []
    for patient_id in patient_ids:
        row = ctn94wk.loc[ctn94wk["who"] == patient_id].values[0]
        risk_weeks = (row[3:12] != 0).astype(int)
        label = 0
        for start in range(6):
            if risk_weeks[start : start + 4].sum() == 4:
                label = 1
                break
        labels.append(label)
    return labels


def encode_treatment_group(project: str, treatment_name: str) -> int:
    """Map CTN project/treatment assignment to the numeric group used downstream."""
    if project == "30":
        return 3
    if treatment_name == "Inpatient NR-NTX":
        return 5
    if project == "51":
        return 4
    if treatment_name == "Methadone":
        return 2
    return 1


def daily_treatment_series(entry_day: Optional[int], records: List[List[float]]) -> str:
    """Return a 168-day binary treatment-exposure string for one patient."""
    series = ["0"] * FOLLOWUP_DAYS
    if entry_day is None:
        return ",".join(series)
    for _amount, day in records:
        aligned_day = int(day - entry_day + WEEK_DAYS)
        if 0 <= aligned_day < FOLLOWUP_DAYS:
            series[aligned_day] = "1"
    return ",".join(series)


def medication_amount_series(
    entry_day: Optional[int],
    records: List[List[float]],
    treatment_name: str,
) -> Dict[str, str]:
    """Return separate 168-day amount strings for BUP, methadone, and XR-NTX."""
    per_type = {drug: ["0"] * FOLLOWUP_DAYS for drug in CANONICAL_MOUD_TYPES}
    canonical = PRESCRIBED_OPIOIDS.get(treatment_name)
    if entry_day is not None and canonical in per_type:
        for amount, day in records:
            aligned_day = int(day - entry_day + WEEK_DAYS)
            if 0 <= aligned_day < FOLLOWUP_DAYS:
                per_type[canonical][aligned_day] = str(float(amount))
    return {drug: ",".join(values) for drug, values in per_type.items()}


def extend_effective_exposure(values: List[str], window: int = 29) -> str:
    """Extend a non-zero daily exposure value forward for ``window`` days.

    This is used for injectable naltrexone so one administration is represented
    as an approximately 30-day effective exposure window rather than a one-day
    event.
    """
    extended = list(values)
    for idx, value in enumerate(values):
        if value != "0":
            for offset in range(1, window + 1):
                target = idx + offset
                if target < len(extended):
                    extended[target] = value
    return ",".join(extended)


def extract_weekly_drug_columns(ctn94wk: pd.DataFrame, patient_ids: List[int]) -> Dict[str, List[float]]:
    """Convert the Excel wide weekly UDS table into named ``Drug_week#`` columns."""
    weekly_drugs: Dict[str, List[float]] = {}
    for start_col in range(0, ctn94wk.shape[1] - 1, N_WEEKS):
        prefix = ctn94wk.columns[start_col][:-6]
        drug_name = WEEKLY_DRUG_NAME_MAP.get(prefix)
        if drug_name is None:
            raise KeyError(f"Unexpected weekly UDS column prefix: {prefix!r}")

        for patient_id in patient_ids:
            row = ctn94wk.loc[ctn94wk["who"] == patient_id].values[0]
            for week in range(N_WEEKS):
                col_name = f"{drug_name}_week{week}"
                weekly_drugs.setdefault(col_name, []).append(row[start_col + week])
    return weekly_drugs


def choose_site_column(site_masked: pd.DataFrame) -> Optional[str]:
    """Find the masked site column across small naming differences in releases."""
    candidates = ["site_masked", "site", "site_id", "where", "center", "node", "masked_site"]
    for column in candidates:
        if column in site_masked.columns:
            return column
    non_id_columns = [c for c in site_masked.columns if c != "who"]
    return non_id_columns[0] if non_id_columns else None


def build_predictor_table(
    ctn94wk: pd.DataFrame,
    tables: MutableMapping[str, pd.DataFrame],
    treatment_entry: Mapping[int, int],
    prescribed_opioid: Mapping[int, str],
) -> pd.DataFrame:
    """Assemble the patient-level static and 24-week time-series table."""
    ctn94wk = ctn94wk.copy().fillna(-1)
    patient_ids = sorted(ctn94wk["who"].astype(int).tolist())

    labels = four_week_return_label(ctn94wk, patient_ids)

    demographics = tables["demographics"].sort_values("who").reset_index(drop=True)
    demo = demographics.loc[
        demographics["who"].isin(patient_ids),
        ["age", "is_hispanic", "race", "job", "is_living_stable", "education", "marital", "is_male"],
    ]
    age = demo["age"].tolist()
    race = [value if value in {"Black", "White"} else "Other" for value in demo["race"].tolist()]
    race = [{"Black": 1, "White": 2, "Other": 3}[value] for value in race]
    is_male = [1 if value == "Yes" else 0 for value in demo["is_male"].tolist()]

    everybody = tables["everybody"].sort_values("who").reset_index(drop=True)
    project = everybody.loc[everybody["who"].isin(patient_ids), "project"].values.tolist()

    randomization = tables["randomization"][["who", "treatment"]].drop_duplicates()
    randomization = randomization.sort_values("who").reset_index(drop=True)
    treatment_names = [randomization.loc[randomization["who"] == pid, "treatment"].values[0] for pid in patient_ids]
    treatment_group = [encode_treatment_group(proj, trt) for proj, trt in zip(project, treatment_names)]

    treatment = tables["treatment"]
    treatment_records = [
        treatment.loc[treatment["who"] == pid, ["amount", "when"]].values.tolist()
        if pid in treatment["who"].tolist()
        else []
        for pid in patient_ids
    ]
    entry_days = [treatment_entry.get(pid) for pid in patient_ids]

    treat_wks = [daily_treatment_series(entry, records) for entry, records in zip(entry_days, treatment_records)]

    amount_columns = {drug: [] for drug in CANONICAL_MOUD_TYPES}
    for entry, records, trt in zip(entry_days, treatment_records, treatment_names):
        per_type = medication_amount_series(entry, records, trt)
        for drug in CANONICAL_MOUD_TYPES:
            amount_columns[drug].append(per_type[drug])

    rbs_iv = tables["rbs_iv"].sort_values("who").reset_index(drop=True)
    rbs_iv_rec = rbs_iv.loc[
        rbs_iv["who"].isin(patient_ids) & (rbs_iv["heroin_inject_days"] > 0),
        ["who", "heroin_inject_days"],
    ]
    heroin_inject = [1 if pid in rbs_iv_rec["who"].values else 0 for pid in patient_ids]

    uds_drugs = {drug: [] for drug in tables["uds"]["what"].unique().tolist()}
    tlfb_drugs = {"Heroin": [], "THC": [], "Alcohol": [], "Cocaine": [], "Methadone": [], "Amphetamine": []}

    for pid in patient_ids:
        entry_day = treatment_entry[pid]
        tlfb_start_day = entry_day - WEEK_DAYS * 4

        for drug in tlfb_drugs:
            tab = tables["tlfb"][
                (tables["tlfb"]["who"] == pid)
                & (tables["tlfb"]["what"] == drug)
                & (tables["tlfb"]["when"] > tlfb_start_day)
                & (tables["tlfb"]["when"] <= entry_day)
            ]
            tlfb_drugs[drug].append(tab.shape[0])

        for drug in uds_drugs:
            tab = tables["all_drugs"][
                (tables["all_drugs"]["who"] == pid)
                & (tables["all_drugs"]["what"] == drug)
                & (tables["all_drugs"]["when"] <= entry_day)
                & (tables["all_drugs"]["when"] >= 0)
            ]
            uds_drugs[drug].append(1 if not tab.empty else 0)

        _ = prescribed_opioid[pid]  # Retained as an explicit input check.

    result_columns: Dict[str, List[object]] = {
        "who": patient_ids,
        "return_to_use": labels,
        "treatment_group": treatment_group,
        "treat_wks": treat_wks,
        "age": age,
        "race": race,
        "is_male": is_male,
        "heroin_inject": heroin_inject,
    }

    for drug in CANONICAL_MOUD_TYPES:
        result_columns[f"treat_{drug}_amt"] = amount_columns[drug]
    for drug, values in tlfb_drugs.items():
        result_columns[f"TLFB_{drug}"] = values
    for drug, values in uds_drugs.items():
        result_columns[f"UDS_{drug}"] = values
    result_columns.update(extract_weekly_drug_columns(ctn94wk, patient_ids))

    result = pd.DataFrame(result_columns)

    site_masked = tables.get("site_masked")
    if site_masked is not None:
        site_col = choose_site_column(site_masked)
        if site_col is not None:
            site_map = site_masked[["who", site_col]].drop_duplicates().rename(columns={site_col: "site"})
            result = result.merge(site_map, on="who", how="left")

    result["rand_day"] = result["who"].map(treatment_entry)

    return result


def extend_xr_ntx_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 30-day effective exposure convention to XR-NTX patients."""
    out = df.copy()
    ntx_mask = out["treatment_group"] == 5
    for row_idx in out.index[ntx_mask]:
        out.at[row_idx, "treat_wks"] = extend_effective_exposure(str(out.at[row_idx, "treat_wks"]).split(","))
        out.at[row_idx, "treat_Naltrexone_amt"] = extend_effective_exposure(
            str(out.at[row_idx, "treat_Naltrexone_amt"]).split(",")
        )
    return out


def validate_output(df: pd.DataFrame) -> None:
    """Run lightweight checks that catch common preprocessing/path mistakes."""
    missing = [column for column in STATIC_COLUMNS if column not in df.columns]
    opioid_cols = [f"Opioid_week{i}" for i in range(N_WEEKS)]
    missing += [column for column in opioid_cols if column not in df.columns]
    if missing:
        raise ValueError("Processed table is missing required columns: " + ", ".join(missing))
    if df["who"].duplicated().any():
        raise ValueError("Processed table contains duplicated patient IDs in the `who` column.")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Generate static_timeSeries_new.csv from CTN-0094 source tables.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=root / "data" / "source" / "public.ctn0094data-main" / "data",
        help="Folder containing CTN-0094 .rda files from public.ctn0094data-main and public.ctn0094data-extra.",
    )
    parser.add_argument(
        "--weekly-uds-xlsx",
        type=Path,
        default=root / "data" / "external" / "CTN94WKUDS_E_0330_0820.xlsx",
        help="Weekly UDS Excel file used by the original preprocessing notebook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "static_timeSeries_new.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--miss-flag",
        choices=["visit", "UDS"],
        default="visit",
        help="Observation rule used in the weekly opioid outcome construction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tables = load_rda_tables(args.source_dir)
    available_optional = sorted(OPTIONAL_RDA_TABLES & set(tables))
    print(f"Loaded {len(tables)} CTN source tables from {args.source_dir}")
    if available_optional:
        print("Optional tables available: " + ", ".join(available_optional))

    if not args.weekly_uds_xlsx.exists():
        raise FileNotFoundError(f"Weekly UDS Excel file not found: {args.weekly_uds_xlsx}")

    ctn94wk = pd.read_excel(args.weekly_uds_xlsx)
    _, return_to_use, treatment_entry, prescribed_opioid = build_treatment_entry_maps(
        tables, miss_flag=args.miss_flag
    )
    processed = build_predictor_table(ctn94wk, tables, treatment_entry, prescribed_opioid)
    processed = extend_xr_ntx_exposure(processed)
    validate_output(processed)

    processed.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"Rows: {processed.shape[0]:,}; columns: {processed.shape[1]:,}")
    print(f"Return-to-use positives in processed table: {int(processed['return_to_use'].sum()):,}")


if __name__ == "__main__":
    main()
