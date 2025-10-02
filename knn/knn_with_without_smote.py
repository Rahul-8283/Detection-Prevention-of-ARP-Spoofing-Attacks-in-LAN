# #!/usr/bin/env python3
# """
# process_and_knn.py

# Steps:
# 1. Load combined_arp_dataset.csv
# 2. Drop identifier columns
# 3. Remove duplicates
# 4. Fill missing values (median)
# 5. Normalize (min-max)
# 6. Save processed dataset
# 7. Train + evaluate baseline KNN classifier
# 8. Show useful visualizations
# """

#import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ---------------- Load dataset ----------------
# input_file = r"D:\Desktop\Sem_3\CEN\combined_arp_dataset.csv"
# df = pd.read_csv(input_file)

# print("Initial shape:", df.shape)

# # ---------------- Step 1: Drop inappropriate features ----------------
# drop_cols = [c for c in df.columns if c.lower() in ["flow id", "src ip", "dst ip", "timestamp"]]
# print("Dropping:", drop_cols)
# df = df.drop(columns=drop_cols, errors="ignore")

# # ---------------- Step 2: Remove duplicates ----------------
# before = df.shape[0]
# df = df.drop_duplicates()
# after = df.shape[0]
# print(f"Removed {before - after} duplicate rows")

# # ---------------- Step 3: Handle missing values (median) ----------------
# # Identify feature columns (exclude label if present)
# feature_cols = [c for c in df.columns if c not in ["Label"]]
# X = df[feature_cols]
# y = df["Label"]

# imputer = SimpleImputer(strategy="median")
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# # ---------------- Step 4: Min-Max normalization ----------------
# scaler = MinMaxScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols)

# # Save processed dataset
# processed = X_scaled.copy()
# processed["Label"] = y.values
# processed.to_csv("processed_dataset.csv", index=False)
# print("Processed dataset saved to processed_dataset.csv")
# print("Shape after processing:", processed.shape)

# # ---------------- Step 5: Train/test split ----------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )

# # ---------------- Step 6: Train KNN ----------------
# knn = KNeighborsClassifier(
#     n_neighbors=5,
#     weights="uniform",
#     leaf_size=30
# )
# knn.fit(X_train, y_train)

# # ---------------- Step 7: Evaluation ----------------
# y_pred = knn.predict(X_test)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)

# # ---------------- Step 8: Visualizations ----------------

# # 1) Class distribution
# plt.figure(figsize=(5,4))
# sns.countplot(x=y, palette="viridis")
# plt.title("Class Distribution")
# plt.xlabel("Class (0=Benign, 1=ARP Spoofing)")
# plt.ylabel("Count")
# plt.show()

# # 2) Confusion matrix
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
# plt.title("Confusion Matrix - KNN")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()

# # 3) Example feature distribution (first 10 features)
# plt.figure(figsize=(12,6))
# X_scaled.iloc[:, :10].boxplot()
# plt.xticks(rotation=45, ha="right")
# plt.title("Boxplot of First 10 Features (Normalized)")
# plt.show()
#!/usr/bin/env python3
"""
knn_with_smote.py

Load processed_dataset.csv (assumed already imputed + min-max scaled),
apply SMOTE on training set only, train KNN, and evaluate.

Requirements:
  pip install scikit-learn imbalanced-learn matplotlib seaborn pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# SMOTE from imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except Exception as e:
    raise RuntimeError("imblearn is required. Install with: pip install imbalanced-learn") from e

# ---------------- Config ----------------
INPUT = r"..\processed_dataset.csv"   # file you saved earlier
TEST_SIZE = 0.30
RANDOM_STATE = 42
KNN_PARAMS = dict(n_neighbors=5, weights="uniform", leaf_size=30)

# ---------------- Load data ----------------
df = pd.read_csv(INPUT)
if 'Label' not in df.columns:
    raise SystemExit("processed_dataset.csv must contain a 'Label' column.")

# Separate features and label
X = df.drop(columns=['Label'])
y = df['Label']

print("Dataset shape:", X.shape, "Labels distribution:", Counter(y))

# ---------------- Train/test split (stratified) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("\nBefore SMOTE -> train distribution:", Counter(y_train), "test distribution:", Counter(y_test))

# Plot class distribution before SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Train class distribution (before SMOTE)")
plt.xlabel("Class (0=Benign, 1=ARP)")
plt.ylabel("Count")
plt.show()

# ---------------- Apply SMOTE on training set only ----------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE -> train distribution:", Counter(y_train_res))

# Plot class distribution after SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res)
plt.title("Train class distribution (after SMOTE)")
plt.xlabel("Class (0=Benign, 1=ARP)")
plt.ylabel("Count")
plt.show()

# ---------------- Train KNN ----------------
knn = KNeighborsClassifier(**KNN_PARAMS)
knn.fit(X_train_res, y_train_res)

# ---------------- Evaluate ----------------
y_pred = knn.predict(X_test)
print("\nClassification Report (test set):")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.title("Confusion Matrix - KNN (with SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Also print raw counts
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ---------------- Optional: save model predictions ----------------
out_df = X_test.copy()
out_df['true_label'] = y_test.values
out_df['pred_label'] = y_pred
out_df.to_csv("knn_smote_test_predictions.csv", index=False)
print("Saved test predictions to knn_smote_test_predictions.csv")
