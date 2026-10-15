"""Central configuration: paths and constants shared by every pipeline stage."""
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
PROCESSED = DATA / "processed"
QUARANTINE = DATA / "quarantine"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
REF_DATE = datetime(2026, 9, 30)   # snapshot date of the simulated extract
START = datetime(2024, 10, 1)      # start of the two-year history
N_PATIENTS = 1500
PAGE_SIZE = 3000                   # rows per page of the mock vitals API


def reset_workspace() -> None:
    """Delete and recreate every data folder so each run is reproducible."""
    for folder in (RAW, BRONZE, SILVER, PROCESSED, QUARANTINE):
        if folder.exists():
            shutil.rmtree(folder)
    for folder in (RAW / "vitals_api", BRONZE, SILVER, PROCESSED, QUARANTINE, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
