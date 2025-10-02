#!/usr/bin/env python3
"""
Flexible XGBoost Multi-class Classification
Works with: Label (binary), Category (multi), Sub-category (multi)

Preprocessing:
1. Drop ID-like columns
2. Drop duplicates
3. Median imputation
4. Min-Max normalization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
import joblib 

# ---------------- Config ----------------
INPUT_CSV = r"iotid20_dataset.csv"
PROC_OUT = "processed_xgb_dataset.csv"

# Choose target: "Label", "Category", "Sub-category"
TARGET = "Sub-category"    

RANDOM_STATE = 42
TEST_SIZE = 0.3

# ---------------- Load ----------------
df = pd.read_csv(INPUT_CSV)
print("Loaded dataset:", df.shape)

# Drop ID-like columns
drop_candidates = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "source_file"]
df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

# Drop duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print("Removed duplicates:", before - after)

# ---------------- Separate features/target ----------------
if TARGET == "Category":
    X = df.drop(columns=[TARGET, "Label", "Sub-category"], errors="ignore")
elif TARGET == "Sub-category":
    X = df.drop(columns=[TARGET, "Label", "Category"], errors="ignore")
elif TARGET == "Label":
    X = df.drop(columns=[TARGET, "Category", "Sub-category"], errors="ignore")
else:
    raise SystemExit(f"Unknown target: {TARGET}")

y = df[TARGET]

# ---------------- Encode target labels ----------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Target = {TARGET}, Classes = {class_names}")

# ---------------- Convert features to numeric ----------------
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = pd.to_numeric(X[c], errors="coerce")

# ---------------- Median imputation ----------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------------- Min-Max normalization ----------------
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Save processed dataset
proc_df = X_scaled.copy()
proc_df[TARGET] = y_encoded
proc_df.to_csv(PROC_OUT, index=False)
print("Saved processed dataset:", PROC_OUT)

# ---------------- Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------- Train XGBoost ----------------
xgb_model = XGBClassifier(
    objective="multi:softmax" if len(class_names) > 2 else "binary:logistic",
    num_class=len(class_names) if len(class_names) > 2 else None,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model,"xgb_Sub-category.pkl")
# ---------------- Evaluate ----------------
y_pred = xgb_model.predict(X_test)

print(f"\n=== Classification Report (XGBoost, Target={TARGET}) ===\n")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Confusion matrix plot
# plt.figure(figsize=(8,6))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(cmap="Blues", xticks_rotation=45, ax=plt.gca())
# plt.title(f"XGBoost Confusion Matrix ({TARGET} classification)")
# plt.show()

fig, ax = plt.subplots(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False)
ax.set_title(f"XGBoost Confusion Matrix ({TARGET} classification)")
plt.show()


# Feature importance
# plt.figure(figsize=(10,6))
# plot_importance(xgb_model, max_num_features=15, importance_type="weight")
# plt.title(f"Top 15 Important Features - XGBoost ({TARGET} classification)")
# plt.show()

fig, ax = plt.subplots(figsize=(10,6))
plot_importance(xgb_model, max_num_features=15, importance_type="weight", ax=ax)
ax.set_title(f"Top 15 Important Features - XGBoost ({TARGET} classification)")
plt.show()

