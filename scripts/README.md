# Scripts

The scripts in this folder are command-line companions to the notebooks. They call reusable modules under `src/` rather than storing all analysis code in notebook-export form.

Recommended commands from the repository root:

```bash
python scripts/check_data_files.py
python scripts/smoke_test.py
bash scripts/run_main_benchmark.sh
bash scripts/run_prefix_cis.sh
```

To regenerate `data/processed/static_timeSeries_new.csv` from the public CTN-0094 source files, first place the required `.rda` files from `public.ctn0094data` and `public.ctn0094extra` under `data/source/public.ctn0094data-main/data/`, then run:

```bash
bash scripts/run_preprocessing.sh
```

The notebook runner scripts are also kept for reviewers who prefer to reproduce the executed notebooks directly:

```bash
bash scripts/run_main_benchmark_notebook.sh
bash scripts/run_prefix_cis_notebook.sh
bash scripts/run_preprocessing_notebook.sh
```
