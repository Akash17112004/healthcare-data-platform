"""Validation tests for the finished pipeline (run `python run_pipeline.py` first, then `pytest`).

They check the guarantees the pipeline promises: nothing lost, no duplicate keys, referential
integrity, valid ranges, and a well-formed prediction label.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "processed").exists(), reason="run run_pipeline.py first")


def load(layer, name):
    folder = DATA / layer / name
    files = sorted(folder.rglob("*.parquet"))  # explicit file list: pyarrow would skip '_ingest_date=...' folders
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def test_bronze_matches_source_counts():
    expected = json.loads((DATA / "raw" / "_source_counts.json").read_text())
    for name in ("patients", "admissions", "vitals"):
        assert len(load("bronze", name)) == expected[name]


def test_bronze_has_lineage_columns():
    cols = set(load("bronze", "admissions").columns)
    assert {"_source_system", "_source_file", "_ingested_at", "_batch_id", "_row_hash"} <= cols


def test_reconciliation_bronze_equals_silver_plus_dups_plus_quarantine():
    recon = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "reconciliation.csv")
    assert (recon["bronze"] == recon["silver"] + recon["duplicates"] + recon["quarantined"]).all()


def test_quarantine_matches_reconciliation():
    recon = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "reconciliation.csv").set_index("entity")
    folder = DATA / "quarantine" / "rejected_records"
    counts = {d.name.split("=", 1)[1]: sum(len(pd.read_parquet(f)) for f in d.rglob("*.parquet"))
              for d in folder.glob("entity=*")}
    for entity in recon.index:
        assert counts.get(entity, 0) == recon.loc[entity, "quarantined"]


def test_silver_primary_keys_are_unique_and_not_null():
    for name, key in (("patients", "patient_id"), ("admissions", "admission_id"), ("vitals", "reading_id")):
        df = load("silver", name)
        assert df[key].notna().all() and df[key].is_unique, f"{name}.{key}"


def test_silver_referential_integrity():
    pat, adm, vit = load("silver", "patients"), load("silver", "admissions"), load("silver", "vitals")
    assert adm["patient_id"].isin(pat["patient_id"]).all()
    assert vit["admission_id"].isin(adm["admission_id"]).all()
    assert vit["patient_id"].isin(pat["patient_id"]).all()


def test_silver_admissions_are_valid():
    adm = load("silver", "admissions")
    assert (adm["discharge_ts"] >= adm["admit_ts"]).all()
    assert (adm["total_bill"] > 0).all() and (adm["length_of_stay_days"] >= 1).all()
    assert adm["severity"].between(1, 5).all()
    assert set(adm["department"]) <= {"Cardiology", "Endocrinology", "Pulmonology", "Orthopedics",
                                      "Neurology", "General Medicine", "Nephrology"}
    assert set(adm["admission_type"]) <= {"Emergency", "Elective", "Urgent", "Unknown"}


def test_silver_patients_are_standardised():
    pat = load("silver", "patients")
    assert set(pat["gender"]) <= {"Male", "Female", "Other", "Unknown"}
    assert pat["age"].between(0, 110).all()
    assert pat["has_diabetes"].dtype == bool and pat["has_hypertension"].dtype == bool


def test_silver_vitals_within_clinical_ranges():
    v = load("silver", "vitals")
    assert v["heart_rate"].between(30, 220).all()
    assert v["spo2"].between(50, 100).all()
    assert v["temperature_c"].between(33, 43).all()
    assert (v["systolic_bp"] > v["diastolic_bp"]).all()
    assert v.drop(columns=["_batch_id", "_ingested_at"]).notna().all().all()


def test_processed_label_is_binary_and_consistent():
    feat = load("processed", "admission_features")
    assert set(feat["readmitted_within_30d"].unique()) <= {0, 1}
    assert 0.05 < feat["readmitted_within_30d"].mean() < 0.60
    assert len(feat) == len(load("silver", "admissions"))
    assert feat["admission_id"].is_unique
    # a re-admission flag on visit N+1 must line up with the label on visit N
    f = feat.sort_values(["patient_id", "admit_ts"])
    nxt = f.groupby("patient_id")["is_readmission_30d"].shift(-1).fillna(0)
    assert (nxt == f["readmitted_within_30d"]).all()
