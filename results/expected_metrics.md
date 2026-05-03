# Expected metrics

This file records the printed outputs used as checks when running the notebooks. Values are copied from the uploaded executed notebook outputs.

## Processed CSV

- Rows: 2,199
- Columns: 293
- File: `data/processed/static_timeSeries_new.csv`

## Main filtering counts

| Check | Patients |
|---|---:|
| Raw data | 2,199 |
| After all-missing filter | 2,030 |
| After all-positive filter | 2,027 |
| After all-negative filter | 1,929 |

## Plain and CIS-weighted metrics

| Model | Plain AUROC | CIS AUROC | Plain AUPRC | CIS AUPRC | Plain Brier | CIS Brier | AUROC gap | AUPRC gap | Brier gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.896804 | 0.835580 | 0.946442 | 0.902896 | 0.126618 | 0.165408 | 0.061224 | 0.043546	 | 0.038790 |
| GBDT | 0.895096 | 0.832300 | 0.946251 | 0.902005 | 0.124497 | 0.161136 | 0.062796 | 0.044245 | 0.036639 |
| LR | 0.887538 | 0.823141 | 0.938824 | 0.892611 | 0.133810 | 0.173653 | 0.064397 | 0.046212 | 0.039843 |
| RF | 0.883073 | 0.818261 | 0.940094 | 0.892293 | 0.131603 | 0.167120 | 0.064812 | 0.047800 | 0.035518 |

## LSTM tier-specific performance

| Tier | Week-level samples | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| Q1 | 300 | 0.676667 | 0.987482 | 0.992520 | 0.019580 |
| Q2 | 1,800 | 0.710556 | 0.968873 | 0.986842 | 0.046485 |
| Q3 | 1,640 | 0.676220 | 0.938875 | 0.973502 | 0.091919 |
| Q4 | 2,440 | 0.595492 | 0.840818 | 0.899361 | 0.159884 |
| Q5 | 1,520 | 0.624342 | 0.700970 | 0.799690 | 0.226676 |
