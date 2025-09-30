#!/usr/bin/env python3
"""
arp_vs_normal_knn.py

Pipeline for ARP Spoofing detection only (Option A):
1) Load final_iotid20_like_dataset.csv
2) Filter to keep only:
     - Normal (label==0)
     - ARP Spoofing (Sub-category == "ARP Spoofing")
3) Drop identifier columns (Flow ID, Src IP, Dst IP, Timestamp)
4) Drop duplicates
5) Impute missing values with median
6) Normalize features with Min-Max
7) Save processed dataset
8) Train KNN baseline (n_neighbors=5, weights=uniform, leaf_size=30)
9) Evaluate and visualize results
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
from collections import Counter

# ---------------- Config ----------------
INPUT_CSV = r"D:\Desktop\Sem_3\CEN\Detection-Prevention-of-ARP-Spoofing-Attacks-in-LAN\final_iotid20_like_dataset.csv"
PROC_OUT = "processed_arp_vs_normal.csv"
TEST_PRED_OUT = "knn_arp_vs_normal_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.30
KNN_PARAMS = dict(n_neighbors=5, weights="uniform", leaf_size=30)

# ---------------- Load ----------------
df = pd.read_csv(INPUT_CSV)
print("Loaded dataset:", df.shape)

# ---------------- Filter Normal + ARP Spoofing ----------------
if "Sub-category" not in df.columns:
    raise SystemExit("Dataset must contain 'Sub-category' column")

df_filtered = df[(df['Label'] == 0) | (df['Sub-category'] == "ARP Spoofing")].copy()
print("Filtered dataset (Normal + ARP Spoofing):", df_filtered.shape)
print("Class counts:\n", df_filtered['Sub-category'].value_counts())

# Map target labels: Normal=0, ARP Spoofing=1
df_filtered['arp_label'] = df_filtered['Sub-category'].apply(lambda x: 0 if x=="Normal" else 1)

# ---------------- Drop identifier columns ----------------
drop_candidates = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
df_filtered = df_filtered.drop(columns=[c for c in drop_candidates if c in df_filtered.columns], errors="ignore")

# ---------------- Drop duplicates ----------------
before = df_filtered.shape[0]
df_filtered = df_filtered.drop_duplicates()
after = df_filtered.shape[0]
print("Removed duplicates:", before - after)

# ---------------- Separate features & target ----------------
feature_cols = [c for c in df_filtered.columns if c not in ['Label','Category','Sub-category','arp_label']]
X = df_filtered[feature_cols]
y = df_filtered['arp_label']

# Convert object columns to numeric if possible
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = pd.to_numeric(X[c], errors='coerce')

# ---------------- Handle missing values ----------------
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# ---------------- Min-Max normalization ----------------
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Save processed dataset
proc = X_scaled.copy()
proc['arp_label'] = y.values
proc.to_csv(PROC_OUT, index=False)
print("Saved processed dataset:", PROC_OUT)

# ---------------- Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train distribution:", Counter(y_train), " Test distribution:", Counter(y_test))

# ---------------- Train KNN ----------------
knn = KNeighborsClassifier(**KNN_PARAMS)
knn.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = knn.predict(X_test)
print("\n=== Classification Report (Normal vs ARP Spoofing) ===\n")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# ---------------- Visualizations ----------------
# 1) Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.title("Class Distribution (0=Normal, 1=ARP Spoofing)")
plt.show()

# 2) Confusion matrix
plt.figure(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Confusion Matrix - KNN (ARP vs Normal)")
plt.show()
