from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
processed = ROOT / "data" / "processed" / "static_timeSeries_new.csv"
excel = ROOT / "data" / "external" / "CTN94WKUDS_E_0330_0820.xlsx"
source_dir = ROOT / "data" / "source" / "public.ctn0094data-main" / "data"

print("Repository root:", ROOT)
print("Processed CSV:", processed, "FOUND" if processed.exists() else "MISSING")
print("Excel source:", excel, "FOUND" if excel.exists() else "MISSING")
print("CTN source directory:", source_dir, "FOUND" if source_dir.exists() else "MISSING")

if processed.exists():
    df = pd.read_csv(processed)
    print("Processed CSV shape:", df.shape)
    required = ["who", "return_to_use", "treatment_group"] + [f"Opioid_week{i}" for i in range(24)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
    else:
        print("Required benchmark columns: OK")
