import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Chargement
model_path = os.path.join("models", "model_production.pkl")
model = joblib.load(model_path)
print("Modèle chargé avec succès !")

# 2. Extraction des importances
feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
sorted_importances = feature_importances.sort_values(ascending=True) # Trié pour le graphique

# 3. Affichage console (Tableau de bord rapide)
print("\n--- Top 10 des variables les plus importantes ---")
print(sorted_importances.tail(10))

# 4. Génération du graphique
plt.figure(figsize=(10, 6))
sorted_importances.tail(15).plot(kind='barh', color='skyblue')
plt.title('Top 15 : Importance des variables (Feature Importance)')
plt.xlabel('Score d\'importance')
plt.ylabel('Variables')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig("models/feature_importance.png")
print("\nGraphique sauvegardé sous : models/feature_importance.png")
plt.show()