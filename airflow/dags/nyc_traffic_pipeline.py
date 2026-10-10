
from datetime import datetime, timedelta
import os
import re
import subprocess
import tempfile
from urllib.request import urlopen

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


RAW = "hdfs://namenode:8020/data/nyc/raw/"
STAGING_ROOT = "hdfs://namenode:8020/data/nyc/staging/runs/"
SPARK_MASTER = "spark://spark-master:7077"


def output_base(run_id):
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._-")
    if not safe_id:
        raise ValueError("Invalid Airflow run_id")
    return f"{STAGING_ROOT}{safe_id}/"


def check_service(name, url):
    try:
        with urlopen(url, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"{name} returned HTTP {response.status}")
        print(f"{name} is reachable: {url}")
    except Exception as exc:
        raise RuntimeError(f"{name} health check failed: {exc}") from exc


def check_hdfs():
    check_service("HDFS NameNode", "http://namenode:9870/")


def check_spark():
    check_service("Spark Master", "http://spark-master:8080/")


def run_spark_check(script_text, app_name):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8",
            delete=False, dir="/tmp"
        ) as script:
            script.write(script_text)
            temp_path = script.name

        subprocess.run([
            "spark-submit",
            "--master", SPARK_MASTER,
            "--deploy-mode", "client",
            "--conf", "spark.driver.host=airflow",
            "--conf", "spark.driver.bindAddress=0.0.0.0",
            "--name", app_name,
            temp_path,
        ], check=True, timeout=1800)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def preflight_staging(**context):
    base = output_base(context["run_id"])
    script = f'''
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("nyc-staging-preflight").getOrCreate()
try:
    conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(conf)
    targets = ["{base}trips", "{base}zone_hour"]

    existing = [
        target for target in targets
        if fs.exists(spark._jvm.org.apache.hadoop.fs.Path(target))
    ]
    if existing:
        raise RuntimeError("Run output already exists: " + ", ".join(existing))

    if spark.read.parquet("{RAW}").limit(1).count() == 0:
        raise RuntimeError("Raw input contains no readable rows.")

    print("PREFLIGHT_SUCCESS")
    print("ETL_OUTPUT_BASE={base}")
finally:
    spark.stop()
'''
    run_spark_check(script, "nyc-staging-preflight")


def validate_staging(**context):
    base = output_base(context["run_id"])
    script = f'''
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("nyc-staging-validation").getOrCreate()
try:
    outputs = {{
        "trips": "{base}trips",
        "zone_hour": "{base}zone_hour",
    }}
    for name, path in outputs.items():
        df = spark.read.parquet(path)
        if df.limit(1).count() == 0:
            raise RuntimeError(f"Output {{name}} is empty")
        print(f"VALIDATED_OUTPUT={{name}}; COLUMNS={{df.columns}}")

    print("STAGING_VALIDATION_SUCCESS")
finally:
    spark.stop()
'''
    run_spark_check(script, "nyc-staging-validation")


with DAG(
    dag_id="nyc_traffic_pipeline",
    description="NYC traffic ETL with isolated per-run staging outputs",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "nyc-traffic-team", "retries": 0},
    tags=["nyc", "big-data", "etl"],
) as dag:

    check_hdfs_task = PythonOperator(
        task_id="check_hdfs",
        python_callable=check_hdfs,
    )

    check_spark_task = PythonOperator(
        task_id="check_spark",
        python_callable=check_spark,
    )

    preflight_task = PythonOperator(
        task_id="preflight_staging",
        python_callable=preflight_staging,
    )

    run_etl_task = BashOperator(
        task_id="run_etl",
        bash_command=(
            "export ETL_OUTPUT_BASE='"
            "{{ 'hdfs://namenode:8020/data/nyc/staging/runs/' "
            "+ (run_id | replace(':', '_') | replace('+', '_') "
            "| replace('/', '_') | replace(' ', '_')) + '/' }}"
            "' && spark-submit "
            "--master spark://spark-master:7077 "
            "--deploy-mode client "
            "--conf spark.driver.host=airflow "
            "--conf spark.driver.bindAddress=0.0.0.0 "
            "/opt/airflow/etl_clean_staging.py"
        ),
        execution_timeout=timedelta(hours=3),
    )

    validate_task = PythonOperator(
        task_id="validate_staging",
        python_callable=validate_staging,
    )

    check_hdfs_task >> check_spark_task >> preflight_task
    preflight_task >> run_etl_task >> validate_task
