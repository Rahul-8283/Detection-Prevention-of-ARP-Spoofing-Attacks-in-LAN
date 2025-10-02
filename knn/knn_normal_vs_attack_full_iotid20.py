#!/usr/bin/env python3
"""
full_iotid20_knn_pipeline.py

1) Load final_iotid20_like_dataset.csv (must include 'Label','Category','Sub-category')
2) Drop identifier columns (Flow ID, Src IP, Dst IP, Timestamp)
3) Drop duplicates
4) Impute missing values using MEDIAN
5) Normalize features using MIN-MAX
6) Save processed dataset -> processed_full_dataset.csv
7) Train baseline KNN (n_neighbors=5, weights=uniform, leaf_size=30) using ALL features
8) Evaluate and plot:
    - Class distribution
    - Confusion matrix
    - PCA 2D scatter of test set (colors by label)
    - Boxplots of first N features (normalized)
9) Save test predictions for later analysis

Requirements:
  pip install pandas numpy scikit-learn matplotlib seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from collections import Counter

# ---------------- Config ----------------
INPUT_CSV = r"..\iotid20_dataset.csv"
PROCESSED_OUT = "processed_full_dataset.csv"
TEST_PRED_OUT = "knn_full_test_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.30

KNN_PARAMS = dict(n_neighbors=5, weights="uniform", leaf_size=30)

# ---------------- Load ----------------
if not os.path.exists(INPUT_CSV):
    raise SystemExit(f"Input file not found: {INPUT_CSV}. Put final_iotid20_like_dataset.csv in this folder.")

df = pd.read_csv(INPUT_CSV)
print("Loaded:", INPUT_CSV, "shape:", df.shape)
print("Columns:", list(df.columns)[:20], "...")

# ---------------- 1. Drop identifier columns ----------------
# handle common variants / case differences
drop_candidates = ["Flow ID", "Flow_ID", "flow id", "flow_id", "Src IP", "Src_IP", "src ip", "src_ip",
                   "Dst IP", "Dst_IP", "dst ip", "dst_ip", "Timestamp", "timestamp", "Time", "time"]
to_drop = [c for c in df.columns if c in drop_candidates]
print("Dropping identifier columns (if present):", to_drop)
df = df.drop(columns=to_drop, errors='ignore')

# ---------------- 2. Remove exact duplicates ----------------
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print("Dropped duplicates:", before - after, "rows removed. New shape:", df.shape)

# ---------------- 3. Fill missing values with MEDIAN ----------------
# Identify label columns and feature columns
label_cols = []
for name in ['label', 'Label', 'CATEGORY', 'Category', 'Sub-category', 'Sub_category', 'SubCategory', 'sub-category']:
    if name in df.columns:
        label_cols.append(name)
# Standardize to 'label', 'Category', 'Sub-category' names if present
# Expecting your CSV to already have 'label','Category','Sub-category' as per earlier steps.
# If not, we try to map them:
if 'label' not in df.columns and ('Label' in df.columns):
    df = df.rename(columns={'Label':'label'})
if 'Category' not in df.columns and ('category' in df.columns):
    df = df.rename(columns={'category':'Category'})
if 'Sub-category' not in df.columns and ('Sub_category' in df.columns):
    df = df.rename(columns={'Sub_category':'Sub-category'})

if 'label' not in df.columns:
    raise SystemExit("No 'label' column found in dataset. Make sure final_iotid20_like_dataset.csv contains a 'label' column (0=normal,1=attack).")

# Extract features (all columns except label + Category/Sub-category if present)
non_feature_cols = ['label', 'Category', 'Sub-category']
feature_cols = [c for c in df.columns if c not in non_feature_cols]
print("Feature count (before cleaning):", len(feature_cols))

X = df[feature_cols].copy()
y = df['label'].copy()

# Convert non-numeric columns in X to numeric if possible (coerce)
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = pd.to_numeric(X[c], errors='coerce')

# Impute numeric columns with median; keep any categorical (shouldn't be many)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric feature count:", len(num_cols))
imputer = SimpleImputer(strategy='median')
X[num_cols] = imputer.fit_transform(X[num_cols])

# If any non-numeric columns remain, drop them (or encode) — we prefer full numeric set for KNN
remaining_non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
if remaining_non_numeric:
    print("Warning: dropping non-numeric columns that could not be converted:", remaining_non_numeric)
    X = X.drop(columns=remaining_non_numeric)
    feature_cols = [c for c in feature_cols if c not in remaining_non_numeric]

# ---------------- 4. Min-Max Normalization ----------------
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Completed Min-Max scaling. Feature shape:", X_scaled.shape)

# Save processed dataset (features + labels + Category/Sub-category if present)
proc_df = X_scaled.copy()
proc_df['label'] = y.values
# keep Category/Sub-category if present in original df
if 'Category' in df.columns:
    proc_df['Category'] = df['Category'].values
if 'Sub-category' in df.columns:
    proc_df['Sub-category'] = df['Sub-category'].values

proc_df.to_csv(PROCESSED_OUT, index=False)
print("Saved processed dataset to:", PROCESSED_OUT)

# ---------------- Train/test split (stratified by label) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Label distribution (full):", Counter(y))
print("Label distribution (train):", Counter(y_train), " (test):", Counter(y_test))

# ---------------- Train KNN ----------------
knn = KNeighborsClassifier(**KNN_PARAMS)
knn.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = knn.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Classification report (KNN, all features) ===\n")
print(report)
print("Confusion matrix:\n", cm)

# ---------------- Visualizations ----------------
sns.set(style="whitegrid")

# 1) Class distribution (full dataset)
plt.figure(figsize=(6,4))
sns.countplot(x=proc_df['label'])
plt.title("Label distribution (processed full dataset)")
plt.xlabel("Label (0=Normal, 1=Attack)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution_full.png")
plt.show()

# 2) Confusion matrix heatmap
plt.figure(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title("Confusion Matrix - KNN (all features)")
plt.tight_layout()
plt.savefig("confusion_matrix_knn_all.png")
plt.show()


# 4) Boxplots of first 10 features (normalized)
n_box = min(10, X_scaled.shape[1])
plt.figure(figsize=(12,6))
X_scaled.iloc[:, :n_box].boxplot()
plt.xticks(rotation=45, ha='right')
plt.title("Boxplots (first %d normalized features)" % n_box)
plt.tight_layout()
plt.savefig("feature_boxplots_first10.png")
plt.show()

# Save test predictions
out_df = X_test.copy()
out_df['true_label'] = y_test.values
out_df['pred_label'] = y_pred
out_df.to_csv(TEST_PRED_OUT, index=False)
print("Saved test predictions to:", TEST_PRED_OUT)

print("\nAll done. Files saved: ", PROCESSED_OUT, TEST_PRED_OUT,
      "and plots: class_distribution_full.png, confusion_matrix_knn_all.png, pca_test_labels.png, feature_boxplots_first10.png")
