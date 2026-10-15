# Smart Healthcare Data Analytics and Prediction Platform

Big Data mid-term minor project. A PySpark data pipeline for a hospital group, built in the
**medallion** style: three messy source systems → **Bronze** (raw, traceable) → **Silver** (clean,
validated, rejects quarantined) → **PySpark processing** (KPIs, trends, vitals alerts and a
30-day readmission label for a later prediction model).

> All data is **simulated** with a fixed random seed. No real patient information is used.

## Overview
Hospitals keep registration, admission and bedside-monitoring data in separate systems, each with its own
formats and errors. This project shows how to turn that raw, inconsistent data into something an analyst can
trust. It ingests a CSV export, a database table and a paginated REST API, preserves every raw record with full
lineage, cleans and validates it (while keeping every rejected record for review), and uses PySpark to produce
department KPIs, monthly trends, clinical vitals alerts and a **30-day readmission label** that a prediction
model can learn from.

**Key features**
- Three source types in one pipeline: CSV, relational database, paginated REST API
- Bronze → Silver → processed layers stored as Parquet, with `_source_system`, `_batch_id`, `_ingested_at`, `_row_hash` lineage
- Data-quality handling: duplicates, nulls, wrong types, inconsistent values, invalid and orphan records
- Quarantine table with the reason each record was rejected, and a reconciliation check: `bronze = silver + duplicates + quarantine`
- PySpark processing: broadcast joins, aggregations and pivot, Spark SQL, window functions (`lag`, `lead`, `row_number`, `dense_rank`, rolling averages)
- 10 automated validation tests, a one-command runner and an Airflow DAG

```
 SOURCES (simulated)                  BRONZE                  SILVER                          PROCESSED (PySpark)
 patients_registry.csv   (CSV)   ─┐   data/bronze/patients ─┐ data/silver/patients   ─┐
 hospital_his.db         (DB)    ─┼─► data/bronze/admissions┼► data/silver/admissions┼─► admission_features · dept_kpis
 vitals API, 12 JSON pages (API) ─┘   data/bronze/vitals   ─┘ data/silver/vitals     ─┘   monthly_trend · top_diagnoses
                                      raw text + lineage       typed + standardised         patient_summary
                                                               data/quarantine/ (rejects + reason)
```

## Problem
Patient registration, admissions and bedside vitals live in three systems with inconsistent formats
(mixed date formats, bills like `₹12,500` / `12500 INR`, °C and °F, duplicates, orphan records, sensor
glitches). Reports built on this are unreliable and avoidable 30-day readmissions go unnoticed.
**Users:** hospital administrators, ward doctors, quality team.

## What each stage does

| Stage | Code | What happens |
|---|---|---|
| Sources | `src/generate_sources.py` | Simulates a CSV export, a SQLite database table and a paginated REST API, with realistic data-quality problems injected on purpose |
| Bronze | `src/bronze.py` | Stores everything exactly as received (all columns as text) as Parquet, plus `_source_system`, `_source_file`, `_ingested_at`, `_batch_id`, `_row_hash`. Row counts are reconciled with the sources |
| Silver | `src/silver.py` | Handles missing values, duplicates, wrong types, inconsistent values and invalid records. Rejected rows go to `data/quarantine/` with the reason and the raw record. Asserts `bronze = silver + duplicates + quarantine` |
| Processing | `src/processing.py` | Filtering, broadcast join, transformations, aggregations + pivot, Spark SQL (CTE, window `RANK`) and window functions (`row_number`, `lag`, `lead`, running totals, rolling averages, `dense_rank`) |

Helpers: `src/utils.py` (cleaning functions and lookup tables), `src/config.py` (paths/constants),
`src/spark_session.py`.

## Run it

Requirements: **Python 3.9+** and **Java 17 or 21** (Spark 4).

```bash
git clone <your-repo-url> && cd healthcare-data-platform
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py                 # all stages (~2 minutes)
python run_pipeline.py --stage silver  # or a single stage
python -m pytest -q tests              # validate the outputs
```

Windows note: Spark writing Parquet sometimes needs Hadoop's `winutils`; WSL or Google Colab avoids that.
In Colab: `!apt-get install -y openjdk-17-jdk-headless -qq`, set `JAVA_HOME`, `!pip install pyspark`, upload the folder, run.

Outputs appear under `data/` (git-ignored, rebuilt deterministically), `reports/` and `reports/figures/`.
`notebooks/Smart_Healthcare_Pipeline_Minor_Project.ipynb` is the same pipeline as a single narrated notebook with outputs saved.

## Results of a run

Row accounting (`reports/reconciliation.csv`) - every Bronze row is accounted for:

| Entity | Bronze | Silver | Duplicates removed | Quarantined | Kept |
|---|---:|---:|---:|---:|---:|
| patients | 1,566 | 1,500 | 60 | 6 | 95.8% |
| admissions | 6,083 | 5,464 | 148 | 471 | 89.8% |
| vitals | 33,229 | 29,132 | 967 | 3,130 | 87.7% |

The full list of cleaning actions is in `reports/silver_cleaning_report.csv`. Quarantined vitals are
mostly readings that belong to admissions that were themselves rejected (2,854).

`admission_features` (5,464 rows × 48 columns) carries the target label `readmitted_within_30d`
(25.3% positive), computed with `lead()` over each patient's admission history, plus patient,
admission and vitals-alert features.

![silver](reports/figures/silver_reconciliation.png)
![trends](reports/figures/admissions_and_readmissions.png)

## Project layout

```
healthcare-data-platform/
├── run_pipeline.py            # entry point: runs the stages in order
├── requirements.txt
├── src/                       # one module per stage + helpers
├── tests/                     # validation tests for the finished pipeline
├── airflow/dags/              # DAG: generate >> bronze >> silver >> process >> validate
├── notebooks/                 # narrated notebook version with saved outputs
├── reports/                   # reconciliation + cleaning report + figures
└── data/                      # generated layers (git-ignored)
```

## Testing and automation
- `tests/test_pipeline_outputs.py` (10 tests): Bronze counts equal source counts, lineage columns present,
  reconciliation holds, unique non-null Silver keys, referential integrity (vitals → admissions → patients),
  valid dates/bills/severity, clinical value ranges, systolic > diastolic, and a binary, consistent readmission label.
- `airflow/dags/healthcare_pipeline_dag.py` chains the stages and the tests daily. It has only been syntax-checked
  here; Airflow itself was not run.

## Design decisions
- **Nothing is silently deleted.** Rejects are quarantined with a reason; duplicates and rejects are counted and reconciled.
- **Bronze stays raw** (strings only) so cleaning logic can change without re-ingesting.
- **Imputation is flagged** (`age_imputed`, `bill_imputed`, `severity_imputed`, `n_values_imputed`).
- **Blood pressure is imputed as a pair** so systolic can never fall below diastolic (found by the tests).

## Limitations / scope
- Simulated data; the database source is SQLite read through Python (on a cluster this would be `spark.read.jdbc`).
- Scope covers sources → Bronze → Silver → PySpark processing, plus the repo, tests and DAG. A Gold layer,
  the MLlib readmission model, a Streamlit dashboard and Airflow deployment are the natural next steps; the
  label and feature table needed for the model already exist in `data/processed/admission_features`.
