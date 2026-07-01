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

    # 1. Ingestion
    t1 = BashOperator(
        task_id='ingest_data', 
        bash_command='python /app/src/data/load.py --filename full2025.csv'
    )

    # 2. Nettoyage
    t2 = BashOperator(
        task_id='clean_data', 
        bash_command='python /app/src/data/preprocess.py'
    )

    # 3. Entraînement avec injection explicite de l'URI
    # On passe MLFLOW_TRACKING_URI dans le dictionnaire 'env' pour garantir sa visibilité
    t3 = BashOperator(
        task_id='train_model', 
        bash_command='python /app/src/models/production_train.py',
        env={
            "PYTHONPATH": "/app",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000"
        }
    )

    # 4. Notifier l'API
    t4 = BashOperator(
        task_id='reload_api_model',
        bash_command='curl -f -X POST http://api:8000/reload_model' # -f fait échouer la tâche si l'API répond 404/500
    )

    # Définition de l'ordre d'exécution
    t1 >> t2 >> t3 >> t4