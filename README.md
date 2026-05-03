# Complexity-aware benchmark for OUD treatment-response prediction

This repository contains the code and analysis notebooks used for the complexity-aware benchmark described in the paper. The benchmark is derived from public CTN-0094 opioid use disorder treatment trial data and evaluates week-ahead risk prediction under behavioral-complexity stratification and Complexity Impact Score (CIS)-weighted metrics.

The repository supports two workflows:

1. **Fast benchmark workflow.** Start from the processed CSV included in `data/processed/static_timeSeries_new.csv` and run the modeling/evaluation code directly.
2. **Full preprocessing workflow.** Recreate `static_timeSeries_new.csv` from the public CTN-0094 source data after downloading the source `.rda` files and placing them in the expected folder.

The fast workflow is the intended path for most reviewers. The preprocessing notebook is included for provenance and transparency, and the preprocessing logic is also provided as ordinary Python source code under `src/`.

## Repository layout

```text
.
├── README.md
├── DATA_CARD.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── processed/
│   │   └── static_timeSeries_new.csv
│   ├── external/
│   │   └── CTN94WKUDS_E_0330_0820.xlsx
│   └── source/
│       └── public.ctn0094data-main/
│           └── data/              # place CTN-0094 .rda files here for preprocessing
├── notebooks/
│   ├── 00_preprocess_ctn0094.ipynb
│   ├── 01_main_benchmark_analysis.ipynb
│   └── 02_prefix_cis_figure_s4.ipynb
├── src/
│   ├── __init__.py
│   ├── ctn0094_preprocessing.py
│   ├── utils.py
│   ├── dataset.py
│   ├── complexity.py
│   ├── metrics.py
│   ├── models.py
│   ├── cohorts.py
│   └── figures.py
├── scripts/
│   ├── 00_preprocess_ctn0094.py
│   ├── 01_main_benchmark_analysis.py
│   ├── 02_prefix_cis_figure_s4.py
│   ├── check_data_files.py
│   ├── smoke_test.py
│   ├── run_preprocessing.sh
│   ├── run_main_benchmark.sh
│   ├── run_prefix_cis.sh
│   ├── run_preprocessing_notebook.sh
│   ├── run_main_benchmark_notebook.sh
│   └── run_prefix_cis_notebook.sh
├── results/
│   ├── expected_metrics.json
│   └── expected_metrics.md
└── Figs/                          # generated figures are written here
```

## Setup

Create a clean Python environment, then install the dependencies using either `pip` or `conda`.

### Option 1: Python virtual environment (need 6GB storage for virtural environment, tested on Ubuntu)

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Conda environment

```bash
conda env create -f environment.yml
conda activate oudpred
```

To use the conda environment as a Jupyter kernel, run:

```bash
python -m ipykernel install --user --name oudpred --display-name "Python (oudpred)"
```

The `environment.yml` file creates a Python 3.10 conda environment and installs the package list from `requirements.txt`. The preprocessing code uses `pyreadr` to read `.rda` files and `openpyxl` through pandas to read the Excel source file. The analysis code uses PyTorch, scikit-learn, pandas, NumPy, and matplotlib.

## Quick check

From the repository root, run:

```bash
python scripts/check_data_files.py
python scripts/smoke_test.py
```

The first command verifies that required files are present. The smoke test reproduces the main trajectory-exclusion counts and the week-level sample count without training a model.

## Script-based reproduction

The repository includes both executed notebooks and regular Python scripts. The notebooks retain the outputs used during manuscript preparation. The scripts provide a cleaner command-line path for review and reproduction:

```bash
bash scripts/run_main_benchmark.sh
bash scripts/run_prefix_cis.sh
```

The main script runs `scripts/01_main_benchmark_analysis.py`, which imports reusable code from `src/` and writes generated figures to `Figs/`. It performs filtering, split construction, LSTM evaluation, tabular baselines, tier-specific evaluation, CIS-weighted metrics, cohort-composition stress tests, and ablations.

The prefix-based comparison script runs `scripts/02_prefix_cis_figure_s4.py` and generates the supplementary prefix-/sample-level CIS AUROC comparison.

Notebook execution remains available for reviewers who prefer to reproduce the executed notebooks directly (**to 100% reproduce the results in the manuscript, please run notebooks**):

```bash
bash scripts/run_main_benchmark_notebook.sh
bash scripts/run_prefix_cis_notebook.sh
bash scripts/run_preprocessing_notebook.sh
```

## Fast benchmark workflow

The processed benchmark source table is already included:

```text
data/processed/static_timeSeries_new.csv
```

This is the direct input to the modeling and benchmark scripts, so reviewers can run the benchmark without first downloading and preprocessing the raw CTN-0094 source files.

## Full preprocessing workflow

The raw CTN-0094 `.rda` files are not redistributed in this repository. To regenerate the processed CSV from source data:

1. Download the public `public.ctn0094data` release from the CTN-0094 project.
2. Download the accompanying `public.ctn0094extra` release from the CTN-0094 project.
3. Place the required `.rda` files (in folder `data/`) from both releases under:

```text
data/source/public.ctn0094data-main/data/
```

The preprocessing code expects that folder to contain the CTN-0094 source tables used in the analysis, including objects such as `everybody`, `treatment`, `all_drugs`, `uds`, `uds_temp`, `visit`, `tlfb`, `randomization`, `demographics`, `rbs_iv`, and, when available, `withdrawal`, `screening_date`, `outcomesCTN0094`, and `site_masked`.

For reference, the public source packages are documented at:

```text
https://ctn-0094.github.io/public.ctn0094data/
https://ctn-0094.github.io/public.ctn0094extra/
```

Then run:

```bash
python scripts/00_preprocess_ctn0094.py
# or
bash scripts/run_preprocessing.sh
```

This reads the source `.rda` files and the included Excel file, then writes:

```text
data/processed/static_timeSeries_new.csv
```

## Expected checks from the included processed CSV

The included processed CSV has:

- 2,199 patient-level rows
- 293 columns
- weekly opioid-state columns from `Opioid_week0` through `Opioid_week23`

The main analysis then excludes degenerate trajectories before benchmark construction. The expected counts are:

- raw data: 2,199 patients
- after all-missing filter: 2,030 patients
- after all-positive filter: 2,027 patients
- after all-negative filter: 1,929 patients

## Data access, hosting, and license

This benchmark is derived from the public CTN-0094 data release. The original CTN-0094 `.rda` source files are not redistributed in this repository. Users who wish to regenerate the processed benchmark table should obtain the public `public.ctn0094data` and `public.ctn0094extra` source packages separately and follow the full preprocessing workflow above.

For reviewer convenience, the processed benchmark table used in the paper is also hosted on Kaggle:

```text
https://www.kaggle.com/datasets/anonymousreview/opioid-complexity-benchmark
```

The Kaggle dataset contains the processed file:

```text
static_timeSeries_new.csv
```

A copy of the same processed table is included in this repository at:

```text
data/processed/static_timeSeries_new.csv
```

This allows reviewers to run the modeling and benchmark analyses without first downloading and preprocessing the raw CTN-0094 source files.

The processed benchmark table is provided for reproducibility of the submitted benchmark. Reuse of the processed table should follow the terms of the original CTN-0094 data release and the license specified on the Kaggle dataset page. The original CTN-0094 source data remain governed by their own public release terms.

This anonymized review version does not include author-identifying repository links. A de-anonymized public repository and dataset page will be provided after review if the paper is accepted.

For the NeurIPS Evaluations & Datasets submission, the Kaggle dataset URL and completed Croissant metadata file are provided through OpenReview.

## Runtime note

The quick data check runs in seconds. The main benchmark script trains the LSTM and tabular baselines and then runs tier-specific evaluation, CIS-weighted evaluation, cohort-composition stress tests, and ablations. Runtime depends on hardware; CPU execution may take substantially longer than the quick data check, but usually the time won't exceed 15 minutes. The executed notebooks and `results/expected_metrics.*` files provide reference outputs for sanity checking.
