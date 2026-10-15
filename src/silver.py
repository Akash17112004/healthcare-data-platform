"""Stage 3 - Silver layer.

Turns raw Bronze text into typed, standardised, validated data. Every Bronze row ends up in exactly
one place, which is asserted at the end:

        bronze rows = silver rows + duplicates removed + quarantined rows

Rejected rows are not deleted: they go to `data/quarantine/` with the rule they failed and the
original raw record.
"""
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from . import config as C
from .utils import (BLOOD_GROUPS, BOOL_MAP, DEPT_MAP, DEVICE_MAP, GENDER_MAP, OUTCOME_MAP, SEVERITY_MAP,
                    TYPE_MAP, VITAL_RANGES, clean_str, lookup, parse_ts, ref_date, ref_ts, to_pandas,
                    to_quarantine)


# ═════════════════════════ PATIENTS ═════════════════════════
def _silver_patients(spark, log):
    cols = ["patient_id", "gender", "dob", "city", "blood_group", "insurance_type", "has_diabetes",
            "has_hypertension", "registration_date", "last_updated"]
    p0 = spark.read.parquet(str(C.BRONZE / "patients"))
    n_bronze = p0.count()
    p = (p0.withColumn("_raw", F.to_json(F.struct(*cols)))  # keep the raw record for the quarantine table
           .select("_raw", "_ingested_at", "_batch_id", *[clean_str(c).alias(c) for c in cols]))

    # (a) no primary key -> cannot be trusted -> quarantine
    nokey = p.filter(F.col("patient_id").isNull()).withColumn("reject_reason", F.lit("missing_patient_id"))
    keyed = p.filter(F.col("patient_id").isNotNull()).withColumn(
        "last_updated_ts", F.to_timestamp("last_updated", "yyyy-MM-dd HH:mm:ss"))

    # (b) de-duplicate: keep the most recently updated version of each patient (window function)
    w = Window.partitionBy("patient_id").orderBy(F.col("last_updated_ts").desc_nulls_last(), F.col("_ingested_at").desc())
    ranked = keyed.withColumn("_rn", F.row_number().over(w))
    n_dups = ranked.filter("_rn > 1").count()
    latest = ranked.filter("_rn = 1").drop("_rn")

    # (c) date of birth -> date, validate, derive age
    dob = F.coalesce(F.to_date("dob", "yyyy-MM-dd"), F.to_date("dob", "dd/MM/yyyy"))
    latest = (latest.withColumn("dob_d", dob)
                    .withColumn("age_calc", F.floor(F.months_between(ref_date(), F.col("dob_d")) / 12)))
    dob_ok = F.col("dob_d").isNotNull() & (F.col("dob_d") <= ref_date()) & (F.col("age_calc") <= 110)
    median_age = int(latest.filter(dob_ok).approxQuantile("age_calc", [0.5], 0.01)[0])

    n_dob_bad = latest.filter(~dob_ok).count()
    n_gender_fx = latest.filter(F.col("gender").isNotNull() & ~F.col("gender").isin("Male", "Female", "Other")).count()
    n_city_null = latest.filter(F.col("city").isNull()).count()

    silver = latest.select(
        "patient_id",
        lookup("gender", GENDER_MAP, "Unknown").alias("gender"),
        F.when(dob_ok, F.col("dob_d")).alias("date_of_birth"),
        F.when(dob_ok, F.col("age_calc")).otherwise(F.lit(median_age)).cast("int").alias("age"),
        (~dob_ok).alias("age_imputed"),
        F.coalesce(F.initcap("city"), F.lit("Unknown")).alias("city"),
        F.when(F.upper("blood_group").isin(BLOOD_GROUPS), F.upper("blood_group")).otherwise("Unknown").alias("blood_group"),
        F.coalesce(F.initcap("insurance_type"), F.lit("Unknown")).alias("insurance_type"),
        lookup("has_diabetes", BOOL_MAP, False).alias("has_diabetes"),
        lookup("has_hypertension", BOOL_MAP, False).alias("has_hypertension"),
        F.to_date("registration_date", "yyyy-MM-dd").alias("registration_date"),
        "last_updated_ts", "_batch_id", "_ingested_at",
    ).cache()

    quarantine = to_quarantine(nokey, "patients", "patient_id")
    n_q, n_silver = quarantine.count(), silver.count()
    log("patients", "bronze rows", n_bronze)
    log("patients", "quarantined: missing patient_id", n_q)
    log("patients", "duplicates removed (older versions)", n_dups)
    log("patients", "gender spellings standardized", n_gender_fx)
    log("patients", "invalid/missing DOB -> age imputed", n_dob_bad)
    log("patients", "missing city -> 'Unknown'", n_city_null)
    log("patients", "silver rows", n_silver)
    print(f"[silver] patients   bronze {n_bronze:>6,} -> silver {n_silver:>6,}  (duplicates {n_dups}, quarantined {n_q}, median age {median_age})")
    return silver, quarantine, dict(bronze=n_bronze, silver=n_silver, dups=n_dups, quarantined=n_q)


# ═════════════════════════ ADMISSIONS ═════════════════════════
def _silver_admissions(spark, log, silver_patients):
    cols = ["admission_id", "patient_id", "department", "doctor_id", "admission_type", "primary_diagnosis",
            "severity", "admit_date", "discharge_date", "total_bill", "outcome"]
    a0 = spark.read.parquet(str(C.BRONZE / "admissions"))
    n_bronze = a0.count()
    a = (a0.withColumn("_raw", F.to_json(F.struct(*cols)))
           .select("_raw", "_ingested_at", "_batch_id", *[clean_str(c).alias(c) for c in cols]))

    nokey = a.filter(F.col("admission_id").isNull()).withColumn("reject_reason", F.lit("missing_admission_id"))
    keyed = a.filter(F.col("admission_id").isNotNull())
    ranked = keyed.withColumn("_rn", F.row_number().over(
        Window.partitionBy("admission_id").orderBy(F.col("_ingested_at").desc())))
    n_dups = ranked.filter("_rn > 1").count()
    latest = ranked.filter("_rn = 1").drop("_rn")

    # type fixing & standardisation
    bill_txt = F.regexp_replace(F.regexp_replace(F.col("total_bill"), r"(?i)(rs\.?|inr|₹)", ""), r"[,\s]", "")
    a_p = (latest
           .withColumn("admit_ts", parse_ts("admit_date"))
           .withColumn("discharge_ts", parse_ts("discharge_date"))
           .withColumn("dept_c", lookup("department", DEPT_MAP))
           .withColumn("sev_c", lookup("severity", SEVERITY_MAP).cast("int"))
           .withColumn("bill_num", bill_txt.cast("double")))

    # missing values: severity -> overall median; bill -> median of same department & severity (window)
    sev_median = int(round(a_p.filter(F.col("sev_c").isNotNull()).approxQuantile("sev_c", [0.5], 0.01)[0]))
    a_p = (a_p.withColumn("severity_imputed", F.col("sev_c").isNull())
              .withColumn("severity_f", F.coalesce("sev_c", F.lit(sev_median)))
              .withColumn("_med_bill", F.percentile_approx(F.when(F.col("bill_num") > 0, F.col("bill_num")), 0.5)
                                        .over(Window.partitionBy("dept_c", "severity_f")))
              .withColumn("bill_imputed", F.col("bill_num").isNull())
              .withColumn("bill_f", F.coalesce("bill_num", F.round("_med_bill", 0))))

    # referential integrity: the patient must exist in the Silver registry
    sp_ids = silver_patients.select("patient_id").distinct().withColumn("_p_ok", F.lit(True))
    a_p = a_p.join(sp_ids, "patient_id", "left")

    reject = (F.when(F.col("patient_id").isNull(), "missing_patient_id")
               .when(F.col("admit_ts").isNull(), "missing_or_unparseable_admit_date")
               .when(F.col("admit_ts") > ref_ts(), "admit_date_in_future")
               .when(F.col("discharge_ts").isNull(), "missing_or_unparseable_discharge_date")
               .when(F.col("discharge_ts") < F.col("admit_ts"), "discharge_before_admit")
               .when(F.col("dept_c").isNull(), "unknown_department")
               .when(F.col("bill_num") <= 0, "non_positive_bill")
               .when(F.col("_p_ok").isNull(), "orphan_patient_not_in_registry"))
    flagged = a_p.withColumn("reject_reason", reject).cache()
    bad, good = flagged.filter(F.col("reject_reason").isNotNull()), flagged.filter(F.col("reject_reason").isNull())

    silver = good.select(
        "admission_id", "patient_id", F.col("dept_c").alias("department"),
        F.coalesce("doctor_id", F.lit("Unassigned")).alias("doctor_id"),
        lookup("admission_type", TYPE_MAP, "Unknown").alias("admission_type"),
        F.coalesce("primary_diagnosis", F.lit("Unspecified")).alias("primary_diagnosis"),
        F.col("severity_f").cast("int").alias("severity"), "severity_imputed",
        "admit_ts", "discharge_ts",
        F.greatest(F.lit(1), F.ceil((F.unix_timestamp("discharge_ts") - F.unix_timestamp("admit_ts")) / 86400))
         .cast("int").alias("length_of_stay_days"),
        lookup("outcome", OUTCOME_MAP, "Unknown").alias("outcome"),
        F.round("bill_f", 0).cast("double").alias("total_bill"), "bill_imputed", "_batch_id", "_ingested_at",
    ).cache()

    quarantine = to_quarantine(nokey, "admissions", "admission_id").unionByName(to_quarantine(bad, "admissions", "admission_id"))
    n_q, n_silver = quarantine.count(), silver.count()
    log("admissions", "bronze rows", n_bronze)
    log("admissions", "duplicates removed", n_dups)
    for reason, n in bad.groupBy("reject_reason").count().collect():
        log("admissions", f"quarantined: {reason}", n)
    log("admissions", "severity missing -> imputed", silver.filter("severity_imputed").count())
    log("admissions", "bill missing -> imputed (dept+severity median)", silver.filter("bill_imputed").count())
    log("admissions", "silver rows", n_silver)
    print(f"[silver] admissions bronze {n_bronze:>6,} -> silver {n_silver:>6,}  (duplicates {n_dups}, quarantined {n_q})")
    return silver, quarantine, dict(bronze=n_bronze, silver=n_silver, dups=n_dups, quarantined=n_q)


# ═════════════════════════ VITALS ═════════════════════════
def _silver_vitals(spark, log, silver_admissions):
    cols = ["reading_id", "patient_id", "admission_id", "recorded_at", "heart_rate", "systolic_bp", "diastolic_bp",
            "temperature", "temp_unit", "spo2", "blood_glucose", "respiration_rate", "source_device"]
    v0 = spark.read.parquet(str(C.BRONZE / "vitals"))
    n_bronze = v0.count()
    v = (v0.withColumn("_raw", F.to_json(F.struct(*cols)))
           .select("_raw", "_ingested_at", "_batch_id", *[clean_str(c).alias(c) for c in cols]))

    nokey = v.filter(F.col("reading_id").isNull()).withColumn("reject_reason", F.lit("missing_reading_id"))
    keyed = v.filter(F.col("reading_id").isNotNull())
    ranked = keyed.withColumn("_rn", F.row_number().over(
        Window.partitionBy("reading_id").orderBy(F.col("_ingested_at").desc())))
    n_dups = ranked.filter("_rn > 1").count()
    latest = ranked.filter("_rn = 1").drop("_rn")

    # parse text -> numbers, unify units
    def num(c):  # "78 bpm" -> 78.0 ; "97%" -> 97.0
        return F.regexp_extract(F.col(c), r"(-?\d+\.?\d*)", 1).cast("double")

    unit = F.upper(F.coalesce("temp_unit", F.lit("C")))
    t_raw = num("temperature")
    temp_c = F.when((unit == "F") | (t_raw > 45), (t_raw - 32) * 5 / 9).otherwise(t_raw)  # F (or obvious decimal slip) -> C

    v_p = (latest
           .withColumn("recorded_ts", F.coalesce(
               F.to_timestamp("recorded_at", "yyyy-MM-dd'T'HH:mm:ss"),
               F.when(F.col("recorded_at").rlike(r"^\d{9,11}$"), F.timestamp_seconds(F.col("recorded_at").cast("long")))))
           .withColumn("heart_rate_raw", num("heart_rate")).withColumn("systolic_bp_raw", num("systolic_bp"))
           .withColumn("diastolic_bp_raw", num("diastolic_bp")).withColumn("temperature_c_raw", temp_c)
           .withColumn("spo2_raw", num("spo2")).withColumn("blood_glucose_raw", num("blood_glucose"))
           .withColumn("respiration_rate_raw", num("respiration_rate")))

    # clinical plausibility: anything outside the range is a sensor error -> NULL
    bp_bad = F.coalesce(F.col("systolic_bp_raw") <= F.col("diastolic_bp_raw"), F.lit(False))  # systolic must exceed diastolic
    for k, (lo, hi) in VITAL_RANGES.items():
        cond = F.col(f"{k}_raw").between(lo, hi)
        if k in ("systolic_bp", "diastolic_bp"):
            cond = cond & ~bp_bad
        v_p = v_p.withColumn(f"{k}_v", F.when(cond, F.col(f"{k}_raw")))
    invalid = v_p.agg(*[F.sum((F.col(f"{k}_raw").isNotNull() & F.col(f"{k}_v").isNull()).cast("int")).alias(k)
                        for k in VITAL_RANGES]).first().asDict()

    # repair patient_id from the admission, then referential integrity on the admission
    adm_link = silver_admissions.select("admission_id", F.col("patient_id").alias("adm_patient_id"))
    v_p = v_p.join(adm_link, "admission_id", "left")
    n_repaired = v_p.filter(F.col("patient_id").isNull() & F.col("adm_patient_id").isNotNull()).count()
    v_p = v_p.withColumn("patient_id", F.coalesce("patient_id", "adm_patient_id"))

    n_valid = sum(F.col(f"{k}_v").isNotNull().cast("int") for k in VITAL_RANGES)
    reject = (F.when(F.col("adm_patient_id").isNull(), "orphan_or_missing_admission")
               .when(F.col("recorded_ts").isNull(), "missing_or_unparseable_timestamp")
               .when(n_valid < 3, "fewer_than_3_valid_vitals"))
    flagged = v_p.withColumn("reject_reason", reject).cache()
    bad, good = flagged.filter(F.col("reject_reason").isNotNull()), flagged.filter(F.col("reject_reason").isNull())

    # blood pressure is a PAIR: if either value is missing/invalid, drop both and impute them together,
    # otherwise independent imputation could produce systolic <= diastolic
    pair_ok = F.col("systolic_bp_v").isNotNull() & F.col("diastolic_bp_v").isNotNull()
    good = (good.withColumn("systolic_bp_v", F.when(pair_ok, F.col("systolic_bp_v")))
                .withColumn("diastolic_bp_v", F.when(pair_ok, F.col("diastolic_bp_v"))))

    # impute remaining NULLs: the patient's own median (window), else the global median
    glob = good.approxQuantile([f"{k}_v" for k in VITAL_RANGES], [0.5], 0.01)
    glob_med = {k: g[0] for k, g in zip(VITAL_RANGES, glob)}
    w_pat = Window.partitionBy("patient_id")
    for k in VITAL_RANGES:
        good = (good.withColumn(f"{k}_imp", F.col(f"{k}_v").isNull())
                    .withColumn(k, F.coalesce(F.col(f"{k}_v"), F.percentile_approx(F.col(f"{k}_v"), 0.5).over(w_pat),
                                              F.lit(glob_med[k]))))
    imputed = good.agg(*[F.sum(F.col(f"{k}_imp").cast("int")).alias(k) for k in VITAL_RANGES]).first().asDict()

    silver = good.select(
        "reading_id", "patient_id", "admission_id", "recorded_ts",
        F.round("heart_rate").cast("int").alias("heart_rate"), F.round("systolic_bp").cast("int").alias("systolic_bp"),
        F.round("diastolic_bp").cast("int").alias("diastolic_bp"), F.round("temperature_c", 1).alias("temperature_c"),
        F.round("spo2").cast("int").alias("spo2"), F.round("blood_glucose").cast("int").alias("blood_glucose"),
        F.round("respiration_rate").cast("int").alias("respiration_rate"),
        lookup("source_device", DEVICE_MAP, "Unknown").alias("source_device"),
        sum(F.col(f"{k}_imp").cast("int") for k in VITAL_RANGES).alias("n_values_imputed"),
        "_batch_id", "_ingested_at",
    ).cache()

    quarantine = to_quarantine(nokey, "vitals", "reading_id").unionByName(to_quarantine(bad, "vitals", "reading_id"))
    n_q, n_silver = quarantine.count(), silver.count()
    log("vitals", "bronze rows", n_bronze)
    log("vitals", "duplicates removed (API replays)", n_dups)
    for reason, n in bad.groupBy("reject_reason").count().collect():
        log("vitals", f"quarantined: {reason}", n)
    log("vitals", "patient_id repaired from admission link", n_repaired)
    for k, n in invalid.items():
        log("vitals", f"impossible {k} -> NULL (sensor error)", n or 0)
    log("vitals", "remaining NULL cells imputed (patient/global median)", sum(imputed.values()))
    log("vitals", "silver rows", n_silver)
    print(f"[silver] vitals     bronze {n_bronze:>6,} -> silver {n_silver:>6,}  (duplicates {n_dups}, quarantined {n_q}, patient_id repaired {n_repaired})")
    return silver, quarantine, dict(bronze=n_bronze, silver=n_silver, dups=n_dups, quarantined=n_q)


# ═════════════════════════ ORCHESTRATION ═════════════════════════
def run(spark) -> pd.DataFrame:
    report = []

    def log(entity, check, n):
        report.append((entity, check, int(n)))

    s_pat, q_pat, st_pat = _silver_patients(spark, log)
    s_adm, q_adm, st_adm = _silver_admissions(spark, log, s_pat)
    s_vit, q_vit, st_vit = _silver_vitals(spark, log, s_adm)

    s_pat.write.mode("overwrite").parquet(str(C.SILVER / "patients"))
    s_adm.write.mode("overwrite").parquet(str(C.SILVER / "admissions"))
    s_vit.write.mode("overwrite").parquet(str(C.SILVER / "vitals"))
    (q_pat.unionByName(q_adm).unionByName(q_vit)
          .write.mode("overwrite").partitionBy("entity").parquet(str(C.QUARANTINE / "rejected_records")))

    C.REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(report, columns=["entity", "check", "rows"]).sort_values(["entity", "rows"], ascending=[True, False]) \
      .to_csv(C.REPORTS / "silver_cleaning_report.csv", index=False)

    # reconciliation: bronze = silver + duplicates + quarantine
    recon = []
    for entity, st in (("patients", st_pat), ("admissions", st_adm), ("vitals", st_vit)):
        assert st["bronze"] == st["silver"] + st["dups"] + st["quarantined"], f"{entity}: rows are unaccounted for"
        recon.append((entity, st["bronze"], st["silver"], st["dups"], st["quarantined"],
                      round(100 * st["silver"] / st["bronze"], 1)))
    recon_df = pd.DataFrame(recon, columns=["entity", "bronze", "silver", "duplicates", "quarantined", "kept_%"])
    recon_df.to_csv(C.REPORTS / "reconciliation.csv", index=False)
    print("\n[silver] RECONCILIATION (bronze = silver + duplicates + quarantine)")
    print(recon_df.to_string(index=False))
    print("[silver] OK - every bronze row is accounted for\n")
    return recon_df
