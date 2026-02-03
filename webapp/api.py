import streamlit as st
import requests
import pandas as pd

st.title("Prédiction de Churn Client")
st.write("Entrez les informations du client pour prédire le risque de départ.")

# Formulaire pour les variables brutes (X_raw)
with st.form("customer_form"):
    credit_score = st.number_input("Credit Score", value=600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", value=40)
    tenure = st.number_input("Tenure (années)", value=5)
    balance = st.number_input("Balance", value=0.0)
    num_products = st.number_input("Nombre de produits", value=1)
    has_card = st.selectbox("A une carte de crédit ?", [1, 0])
    active = st.selectbox("Membre actif ?", [1, 0])
    salary = st.number_input("Salaire estimé", value=50000.0)
    
    submitted = st.form_submit_button("Prédire")

if submitted:
    # Préparation du dictionnaire pour l'API
    payload = {
        "features": {
            "CreditScore": credit_score, "Geography": geography,
            "Gender": gender, "Age": age, "Tenure": tenure,
            "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": has_card, "IsActiveMember": active,
            "EstimatedSalary": salary
        }
    }
    
    # Appel de l'API de serving via le réseau Docker [cite: 69]
    try:
        response = requests.post("http://serving-api:8080/predict", json=payload)
        result = response.json()
        
        st.subheader(f"Probabilité de départ : {result['churn_probability']:.2%}")
        if result['prediction'] == 1:
            st.error("Résultat : Risque de départ ÉLEVÉ")
        else:
            st.success("Résultat : Client FIDÈLE")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
