from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime

with DAG('test_connexion', start_date=datetime(2026, 1, 1), schedule_interval=None, catchup=False) as dag:
    task = EmptyOperator(task_id='tache_test')