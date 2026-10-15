"""Reusable cleaning helpers and standard-value lookup tables used by the Silver stage."""
import pandas as pd
from pyspark.sql import functions as F

NULL_TOKENS = ["", "na", "n/a", "null", "none", "nan", "-", "--"]

GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female",
              "o": "Other", "other": "Other", "others": "Other"}
BOOL_MAP = {"y": True, "yes": True, "1": True, "true": True, "t": True,
            "n": False, "no": False, "0": False, "false": False, "f": False}
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
DEPT_MAP = {"cardiology": "Cardiology", "cardio": "Cardiology",
            "endocrinology": "Endocrinology", "endo": "Endocrinology",
            "pulmonology": "Pulmonology", "pulmo": "Pulmonology",
            "orthopedics": "Orthopedics", "orthopaedics": "Orthopedics", "ortho": "Orthopedics",
            "neurology": "Neurology", "neuro": "Neurology",
            "general medicine": "General Medicine", "gen med": "General Medicine",
            "nephrology": "Nephrology", "nephro": "Nephrology"}
TYPE_MAP = {"emergency": "Emergency", "er": "Emergency", "elective": "Elective", "urgent": "Urgent"}
OUTCOME_MAP = {"discharged": "Discharged", "recovered": "Discharged",
               "transferred": "Transferred", "deceased": "Deceased"}
SEVERITY_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "low": 1, "mild": 2,
                "moderate": 3, "medium": 3, "high": 4, "critical": 5, "severe": 5}
DEVICE_MAP = {"bedside monitor": "Bedside Monitor", "wearable": "Wearable", "manual entry": "Manual Entry"}

# clinical plausibility ranges: a value outside is a sensor error -> NULL
VITAL_RANGES = {"heart_rate": (30, 220), "systolic_bp": (60, 260), "diastolic_bp": (30, 150),
                "temperature_c": (33, 43), "spo2": (50, 100), "blood_glucose": (30, 600),
                "respiration_rate": (5, 60)}


def ref_date():
    return F.lit("2026-09-30").cast("date")


def ref_ts():
    return F.lit("2026-09-30 23:59:59").cast("timestamp")


def clean_str(c):
    """Trim whitespace and turn placeholder strings into real NULLs."""
    t = F.trim(F.col(c))
    return F.when(F.lower(t).isin(NULL_TOKENS), F.lit(None)).otherwise(t)


def lookup(col, mapping, default=None):
    """Standardise a categorical column through a dictionary (case/space-insensitive)."""
    m = F.create_map(*[F.lit(x) for kv in mapping.items() for x in kv])
    out = m[F.lower(F.trim(F.col(col)))]
    return F.coalesce(out, F.lit(default)) if default is not None else out


def parse_ts(c):
    """Parse a timestamp column that arrives in two different formats."""
    return F.coalesce(F.to_timestamp(c, "yyyy-MM-dd HH:mm:ss"), F.to_timestamp(c, "dd/MM/yyyy HH:mm"))


def to_quarantine(df, entity, key):
    """Shape rejected rows into the common quarantine schema."""
    return df.select(F.lit(entity).alias("entity"), F.col(key).cast("string").alias("record_key"),
                     "reject_reason", F.col("_raw").alias("raw_record"), "_batch_id")


def to_pandas(df) -> pd.DataFrame:
    """Small Spark DataFrame -> pandas without needing Arrow."""
    return pd.DataFrame([r.asDict() for r in df.collect()], columns=df.columns)
