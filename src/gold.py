"""Stage 5 - Gold layer: business-ready and model-ready datasets built from the processed tables.

Outputs (data/gold/):
  ml_dataset          one row per admission: only information known at discharge + the label
  kpi_summary         one-row headline KPIs for the dashboard
  department_monthly  department x month metrics (admissions, bill, LOS, readmission rate)
  vitals_alert_rates  share of vitals readings raising each clinical alert, per department

Label hygiene (this is what makes the model honest):
  * `label_observable` is false when the 30-day follow-up window had not finished by the snapshot date,
    because a missing return visit then proves nothing. Those rows are never used for training/testing.
  * Deceased patients cannot be readmitted, so they are excluded from the modelling population.
  * No column that is derived from the *next* admission is allowed in the feature list.
"""
from datetime import timedelta

from pyspark.sql import functions as F

from . import config as C

FEATURE_NUMERIC = [
    "age", "severity", "length_of_stay_days", "total_bill", "cost_per_day", "admission_seq", "prior_admissions",
    "is_chronic", "has_diabetes_i", "has_hypertension_i", "is_readmission_30d", "days_since_last_discharge",
    "has_vitals", "n_readings", "avg_hr", "max_hr", "min_spo2", "max_temp_c", "max_systolic_bp", "avg_glucose",
    "total_alerts", "alerts_per_reading", "hr_change_first_to_last",
]
FEATURE_CATEGORICAL = ["gender", "insurance_type", "department", "admission_type", "age_group", "primary_diagnosis"]
LABEL = "readmitted_within_30d"
# columns that would leak the answer if they ever sneaked into the feature list
LEAKY = {"next_admit_ts", "readmitted_within_30d"}


def run(spark) -> None:
    feat = spark.read.parquet(str(C.PROCESSED / "admission_features"))
    vit = spark.read.parquet(str(C.SILVER / "vitals"))
    cutoff = C.REF_DATE - timedelta(days=30)

    # ── ml_dataset ───────────────────────────────────────────────────────────
    vitals_cols = ["n_readings", "avg_hr", "max_hr", "min_spo2", "max_temp_c", "max_systolic_bp", "avg_glucose",
                   "total_alerts", "alerts_per_reading", "hr_change_first_to_last"]
    ml = (feat
        .withColumn("has_vitals", F.col("n_readings").isNotNull().cast("int"))
        .withColumn("prior_admissions", F.col("admission_seq") - 1)
        .withColumn("has_diabetes_i", F.col("has_diabetes").cast("int"))
        .withColumn("has_hypertension_i", F.col("has_hypertension").cast("int"))
        .withColumn("days_since_last_discharge", F.coalesce(F.col("days_since_last_discharge"), F.lit(-1.0)))
        .withColumn("label_observable", (F.col("discharge_ts") <= F.lit(cutoff)).cast("boolean"))
        .withColumn("in_model_population", ((F.col("outcome") != "Deceased") & F.col("label_observable")).cast("boolean")))

    # median-impute vitals aggregates for admissions that have no readings (flagged by has_vitals)
    medians = ml.select([F.percentile_approx(c, 0.5).alias(c) for c in vitals_cols]).first().asDict()
    for c in vitals_cols:
        ml = ml.withColumn(c, F.coalesce(F.col(c).cast("double"), F.lit(float(medians[c]))))

    keep = (["admission_id", "patient_id", "admit_ts", "discharge_ts", "outcome", "admit_month"]
            + FEATURE_NUMERIC + FEATURE_CATEGORICAL + [LABEL, "label_observable", "in_model_population"])
    ml = ml.select(*dict.fromkeys(keep))
    assert not (LEAKY - {LABEL}) & set(FEATURE_NUMERIC + FEATURE_CATEGORICAL), "leaky column in feature list"
    ml.write.mode("overwrite").parquet(str(C.GOLD / "ml_dataset"))

    # ── department x month ───────────────────────────────────────────────────
    dept_monthly = (feat.groupBy("department", "admit_month").agg(
            F.count("*").alias("admissions"),
            F.round(F.sum("total_bill"), 0).alias("billing_inr"),
            F.round(F.avg("length_of_stay_days"), 2).alias("avg_los_days"),
            F.round(100 * F.avg(F.col("outcome").eqNullSafe("Deceased").cast("int")), 2).alias("mortality_pct"),
            F.round(100 * F.avg("is_readmission_30d"), 2).alias("readmission_visit_pct"))
        .orderBy("department", "admit_month"))
    dept_monthly.write.mode("overwrite").parquet(str(C.GOLD / "department_monthly"))

    # ── vitals alert rates per department ────────────────────────────────────
    adm_dept = feat.select("admission_id", "department")
    alerts = (vit.join(adm_dept, "admission_id")
        .groupBy("department").agg(
            F.count("*").alias("readings"),
            F.round(100 * F.avg((F.col("heart_rate") > 100).cast("int")), 2).alias("tachycardia_pct"),
            F.round(100 * F.avg((F.col("spo2") < 92).cast("int")), 2).alias("hypoxemia_pct"),
            F.round(100 * F.avg(((F.col("systolic_bp") >= 140) | (F.col("diastolic_bp") >= 90)).cast("int")), 2).alias("high_bp_pct"),
            F.round(100 * F.avg((F.col("temperature_c") >= 38.0).cast("int")), 2).alias("fever_pct"),
            F.round(100 * F.avg((F.col("blood_glucose") >= 180).cast("int")), 2).alias("hyperglycemia_pct")))
    alerts.write.mode("overwrite").parquet(str(C.GOLD / "vitals_alert_rates"))

    # ── headline KPIs (one row) ──────────────────────────────────────────────
    kpi = feat.agg(
        F.count("*").alias("admissions"), F.countDistinct("patient_id").alias("patients"),
        F.round(F.avg("length_of_stay_days"), 2).alias("avg_los_days"), F.round(F.avg("total_bill"), 0).alias("avg_bill_inr"),
        F.round(F.sum("total_bill") / 1e7, 2).alias("total_billing_crore_inr"),
        F.round(100 * F.avg(F.col("outcome").eqNullSafe("Deceased").cast("int")), 2).alias("mortality_pct"),
        F.round(100 * F.avg("is_readmission_30d"), 2).alias("readmission_visit_pct"))
    kpi.write.mode("overwrite").parquet(str(C.GOLD / "kpi_summary"))

    n = ml.count()
    pop = ml.filter("in_model_population")
    n_pop = pop.count()
    n_pos = pop.filter(F.col(LABEL) == 1).count()
    print(f"[gold] ml_dataset: {n:,} admissions | modelling population {n_pop:,} "
          f"(excluded {n - n_pop:,}: deceased or follow-up window not complete) | positive rate {100 * n_pos / n_pop:.1f}%")
    for name in ("ml_dataset", "department_monthly", "vitals_alert_rates", "kpi_summary"):
        print(f"[gold] data/gold/{name:<20} {spark.read.parquet(str(C.GOLD / name)).count():>6,} rows")
