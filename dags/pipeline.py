from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops_team',
    'start_date': datetime(2026, 6, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'real_estate_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    # 1. Ingestion (Appelle load.py)
    t1 = BashOperator(
        task_id='ingest_data', 
        bash_command='python /app/src/data/load.py --filename full2025.csv'
    )

    # 2. Nettoyage (Appelle preprocess.py)
    t2 = BashOperator(
        task_id='clean_data', 
        bash_command='python /app/src/data/preprocess.py'
    )

    # 3. Entraînement (Appelle production_train.py)
    t3 = BashOperator(
        task_id='train_model', 
        bash_command='python /app/models/production_train.py'
    )

    # Définition de l'ordre d'exécution
    t1 >> t2 >> t3