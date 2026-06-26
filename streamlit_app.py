import streamlit as st
import requests

# =========================
# CONFIG
# =========================
API_URL = "http://api:8000/predict"

st.set_page_config(page_title="DVF Predictor", page_icon="🏠")

st.title("🏠 Prédiction Prix Immobilier (MLOps)")
st.write("Entrez les caractéristiques du bien")

# =========================
# INPUTS UTILISATEUR
# =========================

surface = st.number_input("Surface habitable (m²)", min_value=10, max_value=500)
pieces = st.number_input("Nombre de pièces", min_value=1, max_value=20)
terrain = st.number_input("Surface terrain (m²)", min_value=0, max_value=5000)

# =========================
# BUTTON
# =========================

if st.button("Prédire le prix"):

    payload = {
        "data": {
            "surface_reelle_bati": surface,
            "nombre_pieces_principales": pieces,
            "surface_terrain": terrain
        }
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        st.success(f"💰 Prix estimé : {result['prediction']:.2f} € / m²")

    except Exception as e:
        st.error(f"Erreur API : {e}")