"""Validation tests for the finished pipeline (run `python run_pipeline.py` first, then `pytest`).

They check the guarantees the pipeline promises: nothing lost, no duplicate keys, referential
integrity, valid ranges, a well-formed prediction label, a leakage-free Gold layer, and a model whose
predictions are well-formed and genuinely better than chance.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "processed").exists(), reason="run run_pipeline.py first")


def load(layer, name):
    folder = DATA / layer / name
    files = [folder] if folder.is_file() else sorted(folder.rglob("*.parquet"))  # explicit file list: pyarrow would skip '_ingest_date=...' folders
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


# ───────────────────────── Gold layer (step 6) ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_gold_dataset_covers_every_silver_admission_and_has_no_leaky_features():
    from src import gold
    ml = load("gold", "ml_dataset")
    assert len(ml) == len(load("silver", "admissions")) and ml["admission_id"].is_unique
    assert not (gold.LEAKY & set(gold.FEATURE_NUMERIC + gold.FEATURE_CATEGORICAL))
    assert set(gold.FEATURE_NUMERIC + gold.FEATURE_CATEGORICAL) <= set(ml.columns)
    assert ml[gold.FEATURE_NUMERIC].notna().all().all(), "model features must be fully imputed"


def test_gold_modelling_population_excludes_deceased_and_open_followup_windows():
    from src import config
    ml = load("gold", "ml_dataset")
    pop = ml[ml["in_model_population"]]
    assert (pop["outcome"] != "Deceased").all()
    assert (pop["discharge_ts"] <= pd.Timestamp(config.REF_DATE) - pd.Timedelta(days=30)).all()
    assert 0.05 < pop["readmitted_within_30d"].mean() < 0.60


def test_gold_kpis_agree_with_silver():
    kpi = load("gold", "kpi_summary").iloc[0]
    adm = load("silver", "admissions")
    assert kpi["admissions"] == len(adm) and kpi["patients"] == adm["patient_id"].nunique()
    assert load("gold", "department_monthly")["admissions"].sum() == len(adm)


# ───────────────────────── Model + predictions (steps 7-8) ─────────────────────────
def test_predictions_are_well_formed_and_cover_every_admission():
    pred = load("predictions", "readmission_predictions")
    assert len(pred) == len(load("silver", "admissions")) and pred["admission_id"].is_unique
    assert pred["probability"].between(0, 1).all()
    assert set(pred["risk_band"]) <= {"Low", "Medium", "High"}
    assert set(pred["split"]) <= {"train", "validation", "test", "not_modelled"}
    assert set(pred["predicted_readmit"]) <= {0, 1}


def test_train_validation_test_split_is_patient_disjoint():
    pred = load("predictions", "readmission_predictions")
    sets = {s: set(pred.loc[pred["split"] == s, "patient_id"]) for s in ("train", "validation", "test")}
    assert not sets["train"] & sets["test"] and not sets["train"] & sets["validation"] and not sets["validation"] & sets["test"]


def test_model_beats_chance_on_held_out_test_patients():
    rep = json.loads((ROOT / "reports" / "model_report.json").read_text())
    assert rep["test_roc_auc"] > 0.55
    assert rep["test_pr_auc"] > rep["test_pr_auc_baseline"]
    assert rep["top20pct_lift"] > 1.0


def test_higher_predicted_risk_means_higher_actual_readmission_rate():
    test = load("predictions", "readmission_predictions").query("split == 'test'")
    rate = test.groupby("risk_band")["actual_readmitted_30d"].mean()
    assert rate["High"] > rate["Medium"] > rate["Low"]


def test_saved_model_exists():
    assert (DATA / "models" / "readmission_pipeline" / "metadata").exists()


# ───────────────────────── Dashboard (step 9) ─────────────────────────
def test_dashboard_renders_from_pipeline_outputs():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert {"Admissions", "Test ROC-AUC", "Patients to follow up"} <= {m.label for m in at.metric}
