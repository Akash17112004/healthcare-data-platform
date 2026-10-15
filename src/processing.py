"""Stage 4 - PySpark processing on the Silver tables.

Filtering, joins (incl. broadcast), transformations, aggregations + pivot, window functions and
Spark SQL. Produces the analysis-ready tables in data/processed/ and the charts in reports/figures/.
The key derived column is `readmitted_within_30d` (lead() over each patient's admission history).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from . import config as C  # noqa: E402
from .utils import to_pandas  # noqa: E402


def run(spark) -> None:
    pat = spark.read.parquet(str(C.SILVER / "patients"))
    adm = spark.read.parquet(str(C.SILVER / "admissions"))
    vit = spark.read.parquet(str(C.SILVER / "vitals"))
    print(f"[process] loaded silver: patients={pat.count():,} admissions={adm.count():,} vitals={vit.count():,}")

    # ── 1. filtering ─────────────────────────────────────────────────────────
    critical_emerg = adm.filter((F.col("admission_type") == "Emergency") & (F.col("severity") >= 4))
    seniors_chronic = pat.filter((F.col("age") >= 60) & (F.col("has_diabetes") | F.col("has_hypertension")))
    critical_vitals = vit.filter((F.col("spo2") < 90) | (F.col("temperature_c") >= 39.5) | (F.col("heart_rate") > 130))
    print(f"[process] emergency admissions with severity>=4: {critical_emerg.count():,} | "
          f"senior chronic patients: {seniors_chronic.count():,} | critical-vitals readings: {critical_vitals.count():,}")

    # ── 2. transformations + broadcast join (admissions x patient dimension) ──
    age_group = (F.when(F.col("age") < 13, "Child (0-12)").when(F.col("age") < 18, "Teen (13-17)")
                  .when(F.col("age") < 40, "Adult (18-39)").when(F.col("age") < 60, "Middle-aged (40-59)")
                  .otherwise("Senior (60+)"))
    age_sort = (F.when(F.col("age") < 13, 1).when(F.col("age") < 18, 2).when(F.col("age") < 40, 3)
                 .when(F.col("age") < 60, 4).otherwise(5))
    pat_dim = pat.select("patient_id", "gender", "age", age_group.alias("age_group"), age_sort.alias("age_group_sort"),
                         "city", "blood_group", "insurance_type", "has_diabetes", "has_hypertension")

    enriched = (adm.join(F.broadcast(pat_dim), "patient_id", "inner")
        .withColumn("admit_month", F.date_trunc("month", "admit_ts").cast("date"))
        .withColumn("admit_weekday", F.date_format("admit_ts", "E"))
        .withColumn("cost_per_day", F.round(F.col("total_bill") / F.col("length_of_stay_days"), 0))
        .withColumn("los_bucket", F.when(F.col("length_of_stay_days") <= 2, "1. Short (1-2d)")
                                   .when(F.col("length_of_stay_days") <= 5, "2. Medium (3-5d)")
                                   .when(F.col("length_of_stay_days") <= 9, "3. Long (6-9d)")
                                   .otherwise("4. Extended (10d+)"))
        .withColumn("is_chronic", (F.col("has_diabetes") | F.col("has_hypertension")).cast("int")))

    # ── 3. window functions: readmission detection per patient ───────────────
    w_pat = Window.partitionBy("patient_id").orderBy("admit_ts")
    adm_w = (enriched
        .withColumn("admission_seq", F.row_number().over(w_pat))
        .withColumn("prev_discharge_ts", F.lag("discharge_ts").over(w_pat))
        .withColumn("days_since_last_discharge", F.datediff("admit_ts", "prev_discharge_ts"))
        .withColumn("is_readmission_30d", F.when(F.col("days_since_last_discharge").between(0, 30), 1).otherwise(0))
        .withColumn("next_admit_ts", F.lead("admit_ts").over(w_pat))
        .withColumn("readmitted_within_30d",
                    F.when(F.datediff("next_admit_ts", "discharge_ts").between(0, 30), 1).otherwise(0))  # <- ML label
        .withColumn("cumulative_bill", F.sum("total_bill").over(w_pat.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
        .withColumn("visits_so_far", F.count("*").over(w_pat.rowsBetween(Window.unboundedPreceding, Window.currentRow))))

    # ── 4. vitals: alert flags, rolling window, per-admission aggregation ────
    flags = (vit
        .withColumn("tachycardia", (F.col("heart_rate") > 100).cast("int"))
        .withColumn("hypoxemia", (F.col("spo2") < 92).cast("int"))
        .withColumn("hypertension", ((F.col("systolic_bp") >= 140) | (F.col("diastolic_bp") >= 90)).cast("int"))
        .withColumn("fever", (F.col("temperature_c") >= 38.0).cast("int"))
        .withColumn("hyperglycemia", (F.col("blood_glucose") >= 180).cast("int"))
        .withColumn("alert_count", F.col("tachycardia") + F.col("hypoxemia") + F.col("hypertension")
                                    + F.col("fever") + F.col("hyperglycemia")))
    w_v = Window.partitionBy("admission_id").orderBy("recorded_ts")
    flags = (flags.withColumn("reading_no", F.row_number().over(w_v))
                  .withColumn("hr_rolling_avg_3", F.round(F.avg("heart_rate").over(w_v.rowsBetween(-2, 0)), 1)))
    vitals_by_admission = flags.groupBy("admission_id").agg(
        F.count("*").alias("n_readings"),
        F.round(F.avg("heart_rate"), 1).alias("avg_hr"), F.max("heart_rate").alias("max_hr"),
        F.min("spo2").alias("min_spo2"), F.round(F.max("temperature_c"), 1).alias("max_temp_c"),
        F.max("systolic_bp").alias("max_systolic_bp"), F.round(F.avg("blood_glucose"), 0).alias("avg_glucose"),
        F.sum("alert_count").alias("total_alerts"), F.round(F.avg("alert_count"), 2).alias("alerts_per_reading"),
        (F.max_by("heart_rate", "recorded_ts") - F.min_by("heart_rate", "recorded_ts")).alias("hr_change_first_to_last"))

    # ── 5. join + aggregations -> analysis-ready datasets ────────────────────
    admission_features = adm_w.join(vitals_by_admission, "admission_id", "left").cache()
    n_rows = admission_features.count()
    n_pos = admission_features.filter("readmitted_within_30d = 1").count()

    dept_kpis = (admission_features.groupBy("department").agg(
            F.count("*").alias("admissions"), F.countDistinct("patient_id").alias("unique_patients"),
            F.round(F.avg("length_of_stay_days"), 2).alias("avg_los_days"), F.round(F.avg("total_bill"), 0).alias("avg_bill_inr"),
            F.round(100 * F.avg((F.col("outcome") == "Deceased").cast("int")), 2).alias("mortality_pct"),
            F.round(100 * F.avg("readmitted_within_30d"), 2).alias("readmission_30d_pct"),
            F.round(F.avg("total_alerts"), 2).alias("avg_vital_alerts"))
        .withColumn("cost_rank", F.dense_rank().over(Window.orderBy(F.desc("avg_bill_inr"))))
        .orderBy(F.desc("admissions")))

    w_m = Window.orderBy("admit_month")
    monthly_trend = (admission_features.groupBy("admit_month").agg(
            F.count("*").alias("admissions"), F.round(F.sum("total_bill") / 1e5, 2).alias("billing_lakh_inr"))
        .withColumn("rolling_3m_avg", F.round(F.avg("admissions").over(w_m.rowsBetween(-2, 0)), 1))
        .withColumn("mom_change_pct", F.round(100 * (F.col("admissions") - F.lag("admissions").over(w_m))
                                              / F.lag("admissions").over(w_m), 1))
        .orderBy("admit_month"))

    w_d = Window.partitionBy("department")
    top_dx = (admission_features.filter(F.col("primary_diagnosis") != "Unspecified")
        .groupBy("department", "primary_diagnosis")
        .agg(F.count("*").alias("cases"), F.round(F.avg("length_of_stay_days"), 1).alias("avg_los"))
        .withColumn("share_of_dept_pct", F.round(100 * F.col("cases") / F.sum("cases").over(w_d), 1))
        .withColumn("rank_in_dept", F.row_number().over(w_d.orderBy(F.desc("cases"))))
        .filter("rank_in_dept <= 3").orderBy("department", "rank_in_dept"))

    age_by_type = (admission_features.groupBy("age_group_sort", "age_group")
                   .pivot("admission_type", ["Emergency", "Urgent", "Elective"]).count()
                   .orderBy("age_group_sort").drop("age_group_sort"))

    patient_summary = (admission_features.groupBy("patient_id").agg(
            F.count("*").alias("total_admissions"), F.round(F.sum("total_bill"), 0).alias("total_bill_inr"),
            F.round(F.avg("length_of_stay_days"), 1).alias("avg_los_days"), F.sum("is_readmission_30d").alias("readmissions_30d"),
            F.sum((F.col("admission_type") == "Emergency").cast("int")).alias("emergency_admissions"),
            F.max("admit_ts").alias("last_admit_ts")).join(pat_dim, "patient_id", "inner"))

    print(f"[process] admission_features: {n_rows:,} rows x {len(admission_features.columns)} columns; "
          f"30-day readmission label: {n_pos:,} positive ({100 * n_pos / n_rows:.1f}%)")
    print("\n[process] Department KPIs")
    dept_kpis.show(truncate=False)
    print("[process] Top-3 diagnoses per department")
    top_dx.show(21, truncate=False)
    print("[process] Admissions by age group x admission type (pivot)")
    age_by_type.show(truncate=False)

    # ── 6. Spark SQL ─────────────────────────────────────────────────────────
    admission_features.createOrReplaceTempView("admission_features")
    pat_dim.createOrReplaceTempView("patients_dim")
    print("[process] SQL Q1 - department performance report")
    spark.sql("""
        SELECT department, COUNT(*) AS admissions, ROUND(AVG(length_of_stay_days), 2) AS avg_los_days,
               SUM(CASE WHEN admission_type = 'Emergency' THEN 1 ELSE 0 END) AS emergency_cases,
               ROUND(100.0 * SUM(CASE WHEN admission_type = 'Emergency' THEN 1 ELSE 0 END) / COUNT(*), 1) AS emergency_pct,
               ROUND(100.0 * AVG(readmitted_within_30d), 2) AS readmission_pct
        FROM admission_features GROUP BY department ORDER BY admissions DESC
    """).show(truncate=False)
    print("[process] SQL Q2 - high-risk chronic patients (CTE + JOIN + RANK window)")
    spark.sql("""
        WITH patient_stats AS (
            SELECT patient_id, COUNT(*) AS admissions, SUM(is_readmission_30d) AS readmissions, SUM(total_bill) AS total_bill
            FROM admission_features GROUP BY patient_id)
        SELECT s.patient_id, p.age, p.gender, p.city, s.admissions, s.readmissions, ROUND(s.total_bill) AS total_bill_inr,
               RANK() OVER (ORDER BY s.readmissions DESC, s.total_bill DESC) AS risk_rank
        FROM patient_stats s JOIN patients_dim p ON s.patient_id = p.patient_id
        WHERE (p.has_diabetes OR p.has_hypertension) AND s.admissions >= 3
        ORDER BY risk_rank LIMIT 10
    """).show(truncate=False)
    print("[process] SQL Q3 - longest-stay diagnosis per department (RANK() OVER PARTITION BY)")
    spark.sql("""
        SELECT department, primary_diagnosis, cases, avg_los_days FROM (
            SELECT department, primary_diagnosis, COUNT(*) AS cases, ROUND(AVG(length_of_stay_days), 1) AS avg_los_days,
                   RANK() OVER (PARTITION BY department ORDER BY AVG(length_of_stay_days) DESC) AS los_rank
            FROM admission_features WHERE primary_diagnosis <> 'Unspecified'
            GROUP BY department, primary_diagnosis) t
        WHERE los_rank = 1 ORDER BY avg_los_days DESC
    """).show(truncate=False)

    # ── 7. persist processed tables ──────────────────────────────────────────
    outputs = {"admission_features": admission_features, "dept_kpis": dept_kpis, "monthly_trend": monthly_trend,
               "top_diagnoses": top_dx, "patient_summary": patient_summary}
    for name, df in outputs.items():
        df.write.mode("overwrite").parquet(str(C.PROCESSED / name))
        print(f"[process] data/processed/{name:<20} {spark.read.parquet(str(C.PROCESSED / name)).count():>6,} rows")

    _make_figures(spark, monthly_trend, dept_kpis)


def _make_figures(spark, monthly_trend, dept_kpis) -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    # figure 1: where did every bronze row go + why were rows quarantined
    recon = pd.read_csv(C.REPORTS / "reconciliation.csv")
    q = to_pandas(spark.read.parquet(str(C.QUARANTINE / "rejected_records")).groupBy("entity", "reject_reason").count()) \
        .sort_values("count")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    recon.set_index("entity")[["silver", "duplicates", "quarantined"]].plot.barh(
        stacked=True, ax=axes[0], color=["#2F8F73", "#E0972E", "#D9666B"])
    axes[0].set_title("Where did every Bronze row go?")
    axes[0].set_xlabel("rows")
    axes[0].set_ylabel("")
    axes[1].barh(q["entity"] + " · " + q["reject_reason"].str.replace("_", " "), q["count"], color="#3E6AE1")
    axes[1].set_title("Quarantine: why records were rejected")
    axes[1].tick_params(axis="y", labelsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(C.FIGURES / "silver_reconciliation.png", dpi=130)
    plt.close(fig)

    # figure 2: monthly trend + department readmission
    mt = to_pandas(monthly_trend)
    mt["admit_month"] = pd.to_datetime(mt["admit_month"])
    dk = to_pandas(dept_kpis).sort_values("admissions")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(mt["admit_month"], mt["admissions"], width=20, color="#BFD3F5", label="monthly admissions")
    axes[0].plot(mt["admit_month"], mt["rolling_3m_avg"], color="#3E6AE1", linewidth=2.2, label="rolling 3-month avg")
    axes[0].set_ylim(0, mt["admissions"].max() * 1.3)
    axes[0].set_title("Hospital admissions per month")
    axes[0].legend(frameon=False, loc="upper center", ncol=2)
    axes[0].tick_params(axis="x", rotation=45)
    bars = axes[1].barh(dk["department"], dk["readmission_30d_pct"], color="#E0972E")
    for bar, pct, los in zip(bars, dk["readmission_30d_pct"], dk["avg_los_days"]):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%  ·  avg stay {los:.1f} d", va="center", fontsize=9)
    axes[1].set_xlim(0, dk["readmission_30d_pct"].max() * 1.6)
    axes[1].set_xlabel("30-day readmission rate (%)")
    axes[1].set_title("Readmission rate by department")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(C.FIGURES / "admissions_and_readmissions.png", dpi=130)
    plt.close(fig)
    print(f"[process] figures saved to {C.FIGURES}")
