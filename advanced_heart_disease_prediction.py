import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

import shap
import joblib

import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
df = pd.read_csv('heart.csv')

# 2. CLEAN DATA
# Replace 0 in Cholesterol and RestingBP with np.nan (as 0 is physiological nonsense)
for col in ["Cholesterol", "RestingBP"]:
    df[col] = df[col].replace(0, np.nan)

# Quick EDA: Show target distribution
plt.figure(figsize=(4,4))
df.HeartDisease.value_counts().plot.pie(autopct='%1.0f%%', labels=['No', 'Yes'])
plt.title("Heart Disease Distribution")
plt.show()

# EDA: Chest pain type by heart disease
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='ChestPainType', hue='HeartDisease')
plt.title("Chest Pain Type vs Heart Disease")
plt.show()

# EDA: Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="HeartDisease")
plt.title('Sex vs Heart Disease')
plt.show()

# EDA: Correlation heatmap
plt.figure(figsize=(10,8))
corr = pd.get_dummies(df).corr()
sns.heatmap(corr, vmax=.3, center=0, square=True)
plt.title('Feature correlation heatmap')
plt.show()

# 3. PREPARE DATA FOR MODELING
TARGET = 'HeartDisease'
cat_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
num_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

X = df[cat_features + num_features]
y = df[TARGET]

# 4. PREPROCESSING PIPELINE
numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy='median')),
    ("scale", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy='most_frequent')),
    ("ohe", OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 6. MODEL PIPELINES
models = {
    "NaiveBayes": Pipeline([
        ("pre", preprocessor), ("clf", GaussianNB())
    ]),
    "KNN": Pipeline([
        ("pre", preprocessor), ("clf", KNeighborsClassifier())
    ]),
    "LogisticRegression": Pipeline([
        ("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))
    ]),
    "RandomForest": Pipeline([
        ("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42))
    ]),
}

# 7. HYPERPARAMETER TUNING (example: Random Forest)
param_grid_rf = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [5, 10, None]
}

search = GridSearchCV(
    models["RandomForest"], param_grid_rf, cv=5, scoring='roc_auc'
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_

# 8. TRAIN AND EVALUATE ALL MODELS
results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print(f"\n{name} Accuracy: {acc:.3f}")
    if auc is not None:
        print(f"{name} ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))
    results[name] = {"model": pipe, "acc": acc, "auc": auc}

# Add best RF results
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:,1]
print("\nTuned Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Tuned Random Forest ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# 9. CONFUSION MATRIX & ROC CURVE
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1,2,2)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='RF ROC Curve')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

# 10. MODEL INTERPRETABILITY (for best model)
explainer = shap.Explainer(best_rf.named_steps["clf"], feature_names=preprocessor.get_feature_names_out())
X_trans = preprocessor.transform(X_test)
shap_values = explainer(X_trans)
shap.summary_plot(shap_values, X_trans, feature_names=preprocessor.get_feature_names_out())

# 11. SAVE BEST MODEL
joblib.dump(best_rf, 'heart_rf_model.pkl')

# 12. PREDICTION ON NEW DATA
# From README example: [50,1,0,145,0,1,1,139,1,0.7,1]
# Map values to column order: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
# Assume order: 50,M,ATA,145,233,1,Normal,139,Y,0.7,Up
sample = pd.DataFrame([[50,'M','ATA',145,233,1,'Normal',139,'Y',0.7,'Up']],
    columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
)
pred = best_rf.predict(sample)
print("Sample Prediction (1=Heart Disease):", pred[0])

##############
# END OF CODE
##############

# To reload and use the model:
# model = joblib.load('heart_rf_model.pkl')
# pred = model.predict(new_df)
