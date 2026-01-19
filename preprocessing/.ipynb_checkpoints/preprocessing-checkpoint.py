import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split

def feature_engineering(df):
    """Création de nouvelles variables métier"""
    df = df.copy()
    # 1. Ratio de consommation : Combien le client paie par mois d'ancienneté
    df['Charges_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # 2. Fidélité binaire : Client depuis plus de 2 ans ?
    df['Is_Long_Term'] = (df['tenure'] > 24).astype(int)
    
    # 3. Total de services souscrits
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Total_Services'] = (df[service_cols] == 'Yes').sum(axis=1)
    
    return df

def get_pipeline():
    """Construction de la Pipeline de transformation"""
    
    # Variables par type
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Charges_per_Tenure', 'Total_Services']
    cat_features = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity'] 
    # Note: On ne met que les plus importantes pour rester concis

    # Pipeline Numérique : Imputation + Scaling
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline Catégorielle : OneHotEncoding
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Assemblage final
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    return preprocessor

def prepare_data(filepath):
    """Fonction principale pour charger et transformer"""
    # Chargement et nettoyage rapide
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test