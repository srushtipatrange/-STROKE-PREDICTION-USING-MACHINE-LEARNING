# =========================================
# STROKE PREDICTION USING MACHINE LEARNING
# INT234 - Predictive Analytics
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report, roc_auc_score)

# 1. LOAD DATASET
# =================

data = pd.read_csv("C:/Users/VICTUS/Downloads/archive (2)/healthcare-dataset-stroke-data.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nDataset information:")
print(data.info())

print("\nStatistical summary:")
print(data.describe())

# 2. DATA CLEANING
# ==================

print("\nMissing values before cleaning:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()

# The ID column is only an identifier, so it is not used for prediction.
data = data.drop("id", axis=1)

# Fill missing BMI values with the median.
data["bmi"] = data["bmi"].fillna(data["bmi"].median())

# Clean spaces from text columns.
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# 3. FEATURE ENGINEERING
# =======================

# Create age groups from age.
data["Age_Group"] = pd.cut(
    data["age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["Child", "Young Adult", "Adult", "Middle Age", "Senior"],
    include_lowest=True
)

# Create a simple glucose category.
data["Glucose_Level"] = pd.cut(
    data["avg_glucose_level"],
    bins=[0, 100, 140, 500],
    labels=["Normal", "Elevated", "High"],
    include_lowest=True
)

print("\nDataset after feature engineering:")
print(data.head())

# 4. OBJECTIVE 1 - STROKE DISTRIBUTION
# =========================================

print("\nStroke count:")
print(data["stroke"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="stroke")
plt.title("Stroke Distribution")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.show()

# 5. OBJECTIVE 2 - AGE ANALYSIS
# =========================================

plt.figure(figsize=(7, 4))
sns.histplot(data=data, x="age", hue="stroke", bins=25, kde=True)
plt.title("Age Distribution by Stroke Status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(data=data, x="Age_Group", hue="stroke")
plt.title("Stroke by Age Group")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 6. OBJECTIVE 3 - HEALTH CONDITIONS
# =========================================

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="hypertension", hue="stroke")
plt.title("Stroke by Hypertension")
plt.xlabel("Hypertension (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="heart_disease", hue="stroke")
plt.title("Stroke by Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

# 7. OBJECTIVE 4 - GLUCOSE AND BMI
# =========================================

plt.figure(figsize=(7, 4))
sns.boxplot(data=data, x="stroke", y="avg_glucose_level")
plt.title("Average Glucose Level by Stroke Status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(data=data, x="stroke", y="bmi")
plt.title("BMI by Stroke Status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(data=data, x="Glucose_Level", hue="stroke")
plt.title("Stroke by Glucose Category")
plt.tight_layout()
plt.show()

# 8. OBJECTIVE 5 - LIFESTYLE AND WORK
# =========================================

plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="smoking_status", hue="stroke")
plt.title("Stroke by Smoking Status")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="work_type", hue="stroke")
plt.title("Stroke by Work Type")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="Residence_type", hue="stroke")
plt.title("Stroke by Residence Type")
plt.tight_layout()
plt.show()

# 9. CORRELATION HEATMAP
# =======================

numeric_data = data.select_dtypes(include=np.number)

plt.figure(figsize=(9, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 10. PREPARE DATA FOR MACHINE LEARNING
# =========================================

# Convert categorical columns into numeric columns.
data_ml = pd.get_dummies(data, drop_first=True, dtype=int)

X = data_ml.drop("stroke", axis=1)
y = data_ml["stroke"]

# Keep the same class ratio in training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale numerical features for models that use distance/linear calculations.
scaler = MinMaxScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 11. TRAIN FIVE MACHINE LEARNING MODELS
# =========================================

models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
}

results = []

for name, model in models.items():
    print("\n=========================================")
    print("Model:", name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4) if not np.isnan(auc) else "Not available")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

# 12. COMPARE MODEL RESULTS
# ==========================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("stroke_model_results.csv", index=False)
print("\nModel results saved as stroke_model_results.csv")

plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x="Model", y="F1 Score")
plt.title("F1 Score Comparison of Models")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

best_model = results_df.iloc[0]
print("\nBest model based on F1 Score:", best_model["Model"])
print("Best F1 Score:", round(best_model["F1 Score"], 4))

#END
