from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops_team',
    'start_date': datetime(2026, 6, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
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

    # 3. Entraînement et Tracking MLflow
    # On ajoute des variables d'environnement pour MLflow
    t3 = BashOperator(
        task_id='train_model', 
        bash_command='export MLFLOW_TRACKING_URI=http://mlflow:5000 && python /app/src/models/production_train.py',
        env={"PYTHONPATH": "/app"}
    )

    # 4. Nouvelle Tâche : Notifier l'API de recharger le nouveau modèle
    # 4. Tâche de rechargement ultra-robuste
    t4 = BashOperator(
        task_id='reload_api_model',
        # On utilise le nom du service 'api' et on force le POST
        bash_command='curl -X POST http://api:8000/reload_model'
    )
