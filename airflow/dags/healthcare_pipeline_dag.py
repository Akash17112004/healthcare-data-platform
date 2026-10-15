"""Airflow DAG that automates the pipeline end to end (one task per stage, then the validation tests).

Copy this file into your Airflow `dags/` folder and set the PROJECT_DIR environment variable (or edit
the default below) to the folder that contains `run_pipeline.py`.
"""
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/healthcare-data-platform")

with DAG(
    dag_id="smart_healthcare_medallion_pipeline",
    description="Sources -> Bronze -> Silver -> PySpark processing -> validation",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["healthcare", "medallion", "pyspark"],
) as dag:
    def stage(name):
        return BashOperator(task_id=name, bash_command=f"cd {PROJECT_DIR} && python run_pipeline.py --stage {name}")

    generate = stage("generate")   # in production this step is replaced by real source extraction
    bronze = stage("bronze")
    silver = stage("silver")
    process = stage("process")
    validate = BashOperator(task_id="validate_outputs", bash_command=f"cd {PROJECT_DIR} && python -m pytest -q tests")

    generate >> bronze >> silver >> process >> validate
