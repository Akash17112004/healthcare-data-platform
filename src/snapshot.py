"""Copies the small outputs the dashboard needs into dashboard/data/ (committed to GitHub).

Why: the full pipeline output in data/ is git-ignored (it is rebuilt by run_pipeline.py and needs Spark). A hosted
dashboard (e.g. Streamlit Community Cloud) has no Spark and no data/ folder, so the app falls back to this snapshot.
It runs automatically at the end of the model stage; run it alone with `python -m src.snapshot`.
"""
import shutil

import pandas as pd

from . import config as C

SNAPSHOT = C.ROOT / "dashboard" / "data"
SPARK_TABLES = [("gold", "ml_dataset"), ("gold", "vitals_alert_rates"), ("predictions", "readmission_predictions")]
SMALL_FILES = ["model_comparison.parquet", "feature_importance.parquet", "roc_curve.parquet"]


def _read(path):
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run() -> None:
    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)
    for layer, name in SPARK_TABLES:
        out = SNAPSHOT / layer / name
        out.mkdir(parents=True)
        _read(C.DATA / layer / name).to_parquet(out / "part-0.parquet", index=False)   # one clean file per table
    for f in SMALL_FILES:
        shutil.copy(C.PREDICTIONS / f, SNAPSHOT / "predictions" / f)
    size = sum(p.stat().st_size for p in SNAPSHOT.rglob("*") if p.is_file()) / 1024
    print(f"[snapshot] dashboard/data refreshed ({size:,.0f} KB) - commit it so the hosted dashboard shows the latest results")


if __name__ == "__main__":
    run()
