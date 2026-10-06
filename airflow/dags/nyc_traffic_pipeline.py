from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def check_pipeline():
    print("NYC Traffic Analytics pipeline is ready.")
    print("Airflow DAG execution test passed.")


with DAG(
    dag_id="nyc_traffic_pipeline",
    description="Initial DAG for NYC Traffic Analytics",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["nyc", "big-data", "test"],
) as dag:

    pipeline_check = PythonOperator(
        task_id="check_pipeline",
        python_callable=check_pipeline,
    )
