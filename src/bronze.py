"""Stage 2 - Bronze layer.

Land the raw data exactly as received (every column as text, duplicates and errors included) and
attach lineage columns so any downstream number can be traced back to its source row.
"""
import json
import sqlite3
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from . import config as C

VITAL_FIELDS = ["reading_id", "patient_id", "admission_id", "recorded_at", "heart_rate", "systolic_bp",
                "diastolic_bp", "temperature", "temp_unit", "spo2", "blood_glucose", "respiration_rate",
                "source_device"]


def _add_lineage(df, source_system, batch_id):
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return (df.withColumn("_source_system", F.lit(source_system))
              .withColumn("_ingested_at", F.current_timestamp())
              .withColumn("_batch_id", F.lit(batch_id))
              .withColumn("_ingest_date", F.current_date())
              .withColumn("_row_hash", F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c), F.lit("<null>")) for c in data_cols]), 256)))


def _write(df, name):
    df.write.mode("overwrite").partitionBy("_ingest_date").parquet(str(C.BRONZE / name))


def _basename():
    return F.element_at(F.split(F.col("_metadata.file_path"), "/"), -1)


def run(spark) -> str:
    batch_id = "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) CSV source
    patients = (spark.read.option("header", True).option("inferSchema", False)
                .csv(str(C.RAW / "patients_registry.csv")).withColumn("_source_file", _basename()))
    _write(_add_lineage(patients, "registration_desk_csv", batch_id), "patients")

    # 2) Database source (SQLite here; on a cluster this would be spark.read.jdbc(...))
    con = sqlite3.connect(C.RAW / "hospital_his.db")
    cur = con.execute("SELECT * FROM admissions")
    cols = [c[0] for c in cur.description]
    rows = [tuple(None if v is None else str(v) for v in r) for r in cur.fetchall()]
    con.close()
    schema = StructType([StructField(c, StringType(), True) for c in cols])
    admissions = spark.createDataFrame(rows, schema).withColumn("_source_file", F.lit("hospital_his.db::admissions"))
    _write(_add_lineage(admissions, "hospital_his_database", batch_id), "admissions")

    # 3) REST API source (JSON pages) - explicit all-string schema so nothing is silently coerced
    row_schema = StructType([StructField(c, StringType(), True) for c in VITAL_FIELDS])
    resp_schema = StructType([StructField("status", StringType()), StructField("page", StringType()),
                              StructField("total_pages", StringType()), StructField("data", ArrayType(row_schema))])
    vitals = (spark.read.schema(resp_schema).option("multiLine", True).json(str(C.RAW / "vitals_api"))
              .select(F.explode("data").alias("r"), _basename().alias("_source_file"))
              .select("r.*", "_source_file"))
    _write(_add_lineage(vitals, "vitals_rest_api", batch_id), "vitals")

    # reconciliation: nothing lost on the way in
    expected = json.loads((C.RAW / "_source_counts.json").read_text())
    for name in ("patients", "admissions", "vitals"):
        n = spark.read.parquet(str(C.BRONZE / name)).count()
        assert n == expected[name], f"bronze/{name}: {n} rows but source delivered {expected[name]}"
        print(f"[bronze] {name:<11} {n:>7,} rows  (matches source)")
    print(f"[bronze] batch id = {batch_id}")
    return batch_id
