# Data card

## Dataset in this repository

The repository includes a processed patient-level table:

```text
data/processed/static_timeSeries_new.csv
```

This file is generated from public CTN-0094 opioid use disorder treatment trial data and an additional weekly UDS Excel source file included at:

```text
data/external/CTN94WKUDS_E_0330_0820.xlsx
```

The processed CSV is provided so that the modeling and benchmark scripts can run without requiring reviewers to download and preprocess the CTN source tables first.

## Hosting and access

The processed benchmark table used in the paper is hosted on Kaggle for reviewer access:

```text
https://www.kaggle.com/datasets/anonymousreview/opioid-complexity-benchmark
```

The same processed file is included in this repository at:

```text
data/processed/static_timeSeries_new.csv
```

The benchmark is derived from existing public CTN-0094 source data. The original CTN-0094 `.rda` source files are not redistributed here. Users who wish to regenerate the processed benchmark table should obtain the public `public.ctn0094data` and `public.ctn0094extra` source packages separately and run the preprocessing pipeline provided in this repository.

The processed benchmark table is provided for reproducibility of the submitted benchmark. Reuse should follow the terms of the original CTN-0094 data release and the license specified on the Kaggle dataset page.

## Source-data workflow

The preprocessing code expects `.rda` files from `public.ctn0094data` and `public.ctn0094extra` to be placed under:

```text
data/source/public.ctn0094data-main/data/
```

The preprocessing code reads all `.rda` files in that directory and uses tables including `everybody`, `treatment`, `all_drugs`, `uds`, `uds_temp`, `visit`, `tlfb`, `randomization`, `withdrawal`, `screening_date`, `outcomesCTN0094`, `site_masked`, `demographics`, and `rbs_iv`, depending on which objects are present in the public source release.

## Benchmark construction summary

The processed CSV contains 2,199 patient-level rows. The main benchmark analysis excludes degenerate opioid trajectories before constructing the complexity-stratified benchmark split:

- all-missing trajectories are excluded;
- all-positive trajectories are excluded;
- all-negative trajectories are excluded.

The resulting benchmark cohort used in the main analyses contains 1,929 patients. Each patient contributes 20 week-level prediction samples in the main modeling task, producing 38,580 week-level prediction samples.

## Outcome coding

The weekly opioid-use trajectory is encoded from urine drug screen information after removing the participant's prescribed opioid from the urine-screen signal. In the benchmark analyses, missing opioid observation is treated as behaviorally informative and risk-positive for the binary week-ahead target.

## Intended use

This repository is intended for retrospective algorithm development, benchmark evaluation, and reproducibility review. It is not a clinical decision-support tool and should not be used for direct clinical deployment.

## Limitations

The benchmark is derived from a single public opioid treatment data resource and uses a binary week-ahead target. Trial follow-up occurred on structured schedules, which may differ from observational care settings.
