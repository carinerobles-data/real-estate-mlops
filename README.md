Real Estate MLOps Project

Ce projet propose une solution MLOps complète pour prédire les prix immobiliers à partir des données DVF (Demande de Valeurs Foncières). L'objectif est de garantir un cycle de vie robuste, de l'ingestion des données brutes jusqu'au déploiement en production.

Architecture & Composants
Le système repose sur une architecture modulaire orchestrée par Docker Compose :

MLflow : Suivi des expériences et versioning des modèles.

FastAPI : API d'inférence performante.

Streamlit : Interface utilisateur interactive.

Nginx : Load Balancer pour la gestion des instances API.

Prometheus : Monitoring des métriques en temps réel.

Docker/Docker Compose : Orchestration des microservices.


Pipelines
1. Data Engineering (Préparation)
Ce pipeline garantit la qualité et la traçabilité de la donnée :

Ingestion (load.py) : Conversion des CSV bruts vers le format binaire optimisé Parquet.

Nettoyage (clean.py) : Feature Engineering et filtrage métier.

Versionnage (DVC) : Gestion et suivi des gros volumes de données.

2. MLOps (Modélisation & Inférence)
Ce pipeline automatise la livraison du modèle :

Tracking (MLflow) : Enregistrement systématique des paramètres et artefacts.

Entraînement (production_train.py) : Refit automatisé sur données traitées.

Inférence : API haute performance avec chargement dynamique via volumes partagés.


Commandes & Utilisation
Prérequis
Docker et Docker Desktop installés.

Lancement du système Lancement de tout l'écosystème (API, MLflow, Monitoring, Streamlit)
docker-compose up -d --build

Action,Commande
Data Engineering,make load (Ingestion) / make clean (Nettoyage)
Training,python models/production_train.py
API/Proxy,docker-compose up

Accès aux Services
API (Swagger/Docs) : http://localhost:8000/docs

MLflow UI : http://localhost:5000

Monitoring (Prometheus) : http://localhost:9090

Interface (Streamlit) : http://localhost:8501

Structure du dépôt
/src : Code source de l'API.

/models : Stockage des modèles de production (montage en volume).

/experiments : Scripts d'entraînement et notebooks d'analyse.

docker-compose.yml : Orchestration des conteneurs.

Ce projet illustre une approche moderne du MLOps en transformant un pipeline de données isolé en un service de production scalable et monitoré. En combinant DVC pour la donnée, MLflow pour l'expérimentation et Docker/Nginx pour l'inférence, cette architecture garantit non seulement la performance du modèle, mais aussi sa fiabilité et sa maintenabilité sur le long terme.
