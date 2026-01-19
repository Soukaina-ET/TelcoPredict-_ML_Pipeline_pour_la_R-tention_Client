import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from preprocessing.preprocessing import prepare_data, get_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# 1. Préparation des données via notre script précédent
X_train, X_test, y_train, y_test = prepare_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# 2. Création de la pipeline complète (Preprocessing + Modèle)
full_pipeline = Pipeline(steps=[
    ('preprocessor', get_pipeline()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 3. Entraînement
print("Entraînement du modèle en cours...")
full_pipeline.fit(X_train, y_train)

# 4. Évaluation
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("\n--- RAPPORT DE PERFORMANCE ---")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# 5. Sauvegarde du modèle pour le futur (Déploiement)
joblib.dump(full_pipeline, 'churn_model_v1.pkl')
print("\nModèle sauvegardé sous 'churn_model_v1.pkl'")

# 6. Matrice de Confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de Confusion')
plt.show()

# 7. Importance des caractéristiques
# Extraction de l'importance des variables
def plot_feature_importance(model, X_train):
    # Récupérer les noms des colonnes après transformation
    # On accède au preprocessor de la pipeline
    preprocessor = model.named_steps['preprocessor']
    
    # Noms des colonnes numériques
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Charges_per_Tenure', 'Total_Services', 'Is_Long_Term']
    
    # Noms des colonnes catégorielles après OneHotEncoder
    cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
    all_features = num_cols + list(cat_cols)
    importances = model.named_steps['classifier'].feature_importances_
    
    # Création du DataFrame
    feature_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(15), palette='magma')
    plt.title('Top 15 des facteurs de Churn (Random Forest)')
    plt.tight_layout()
    plt.show()

plot_feature_importance(full_pipeline, X_train)


#XGBOOST MODEL TRAINING
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Initialiser le modèle
xgb = XGBClassifier(random_state=42)

# Créer la pipeline complète
clf = Pipeline(steps=[
    ('preprocessor', get_pipeline()),
    ('classifier', xgb)
])

# Définir la grille de recherche (Hyperparameter Tuning)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5]
}

# Configurer la Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lancer la recherche
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='roc_auc', verbose=1)
grid_search.fit(X_train, y_train)

print(f"Meilleur score de Cross-Validation : {grid_search.best_score_}")
# 1. Récupérer le meilleur modèle après GridSearchCV
best_model = grid_search.best_estimator_

# 2. Évaluation finale sur le jeu de test (le vrai test)
y_pred_xgb = best_model.predict(X_test)
y_proba_xgb = best_model.predict_proba(X_test)[:, 1]

print("\n--- PERFORMANCE XGBOOST (TEST SET) ---")
print(classification_report(y_test, y_pred_xgb))
print(f"Nouvel AUC Score: {roc_auc_score(y_test, y_proba_xgb):.4f}")

# 6. Matrice de Confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de Confusion')
plt.show()

# 3. Sauvegarder ce modèle (c'est celui-ci le plus performant)
joblib.dump(best_model, 'churn_model_xgb_final.pkl')

# 7. Importance des caractéristiques
# Extraction de l'importance des variables
def plot_feature_importance(model, X_train):
    # Récupérer les noms des colonnes après transformation
    # On accède au preprocessor de la pipeline
    preprocessor = model.named_steps['preprocessor']
    
    # Noms des colonnes numériques
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Charges_per_Tenure', 'Total_Services', 'Is_Long_Term']
    
    # Noms des colonnes catégorielles après OneHotEncoder
    cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
    all_features = num_cols + list(cat_cols)
    importances = model.named_steps['classifier'].feature_importances_
    
    # Création du DataFrame
    feature_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(15), palette='magma')
    plt.title('Top 15 des facteurs de Churn (XGBoost)')
    plt.tight_layout()
    plt.show()

plot_feature_importance(full_pipeline, X_train)

# OPTIMISATION DU SEUIL DE DÉCISION
from sklearn.metrics import precision_recall_curve

# 1. Calculer les probabilités
y_scores = best_model.predict_proba(X_test)[:, 1]

# 2. Obtenir la précision et le rappel pour différents seuils
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# 3. Trouver le seuil qui donne un rappel de ~0.75 (par exemple)
# On veut capturer 75% des départs au lieu de 51%
target_recall = 0.70
idx = np.where(recalls >= target_recall)[0][-1]
optimal_threshold = thresholds[idx]

print(f"\n--- OPTIMISATION DU SEUIL ---")
print(f"Seuil suggéré pour capturer {target_recall*100}% des départs : {optimal_threshold:.2f}")

# 4. Appliquer ce nouveau seuil
y_pred_new = (y_scores >= optimal_threshold).astype(int)

print("\n--- NOUVEAU RAPPORT (Seuil Optimisé) ---")
print(classification_report(y_test, y_pred_new))

