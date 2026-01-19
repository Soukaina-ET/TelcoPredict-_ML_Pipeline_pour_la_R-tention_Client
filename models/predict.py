import joblib
import pandas as pd
import numpy as np

# 1. Charger le modèle final
model = joblib.load('churn_model_xgb_final.pkl')
threshold = 0.37  # Le seuil que nous avons optimisé

def score_customer(customer_df):
    """Calcule le risque et donne une recommandation"""
    # Calcul de la probabilité
    proba = model.predict_proba(customer_df)[:, 1][0]
    
    # Décision basée sur le seuil optimisé
    status = "RISQUE ÉLEVÉ" if proba >= threshold else "RISQUE FAIBLE"
    
    print(f"--- RÉSULTAT DU SCORING ---")
    print(f"Probabilité de départ : {proba:.2%}")
    print(f"Statut : {status}")
    
    if status == "RISQUE ÉLEVÉ":
        print("Recommandation : Envoyer une offre de rétention (Remise ou appel du service client).")
    else:
        print("Recommandation : Maintenir la communication standard.")


# Pour tester, on peut prendre une ligne de X_test
# --- TEST DU SCORING AVEC UN CLIENT DE X_TEST ---

# 1. On recharge les données de test 
from preprocessing.preprocessing import prepare_data
_, X_test, _, _ = prepare_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Sélectionner un client au hasard (ex: le 10ème client de la liste)
sample_customer = X_test.iloc[[10]] 

print("Données du client sélectionné :")
print(sample_customer[['tenure', 'MonthlyCharges', 'Contract']])
print("-" * 30)

# 3. Lancer le scoring
score_customer(sample_customer)

# --- CRÉATION D'UN CLIENT IMAGINAIRE ---

fake_customer_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,                 # Très récent
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic', # Fibre (souvent lié au churn élevé)
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month', # Contrat risqué
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.50,     # Prix élevé
    'TotalCharges': 191.00
}

# Conversion en DataFrame (format attendu par la pipeline)
fake_df = pd.DataFrame([fake_customer_data])

# On applique le feature engineering (calcul des ratios) avant le scoring
from preprocessing.preprocessing import feature_engineering
fake_df_engineered = feature_engineering(fake_df)

print("\n--- TEST SUR CLIENT IMAGINAIRE (NOUVEAU) ---")
score_customer(fake_df_engineered)