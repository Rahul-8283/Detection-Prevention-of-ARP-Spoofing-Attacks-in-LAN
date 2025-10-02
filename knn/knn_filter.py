# #!/usr/bin/env python3
# """
# arp_feature_selection_knn.py

# Pipeline:
# 1) Load combined ARP dataset (Normal + ARP Spoofing)
# 2) Drop identifier columns (Flow ID, Src IP, Dst IP, Timestamp) if present
# 3) Drop duplicates
# 4) Impute missing values (median)
# 5) Min-Max normalization
# 6) Feature Selection: Filter correlation method (threshold = 0.8)
# 7) Train KNN (n_neighbors=5, weights=uniform, leaf_size=30)
# 8) Evaluate & visualize
# """

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
INPUT_CSV = r"..\combined_arp_dataset.csv"   # your binary file: Normal + ARP
PROC_OUT = "processed_arp_corr_selected.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CORR_THRESHOLD = 0.8
KNN_PARAMS = dict(n_neighbors=5, weights="uniform", leaf_size=30)

# ---------------- Load ----------------
df = pd.read_csv(INPUT_CSV)
print("Loaded dataset:", df.shape)

# ---------------- Drop ID-like columns ----------------
drop_candidates = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "source_file"]
df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

# ---------------- Ensure Label column exists ----------------
if "Label" not in df.columns:
    raise SystemExit("Dataset must have a column named 'Label' (0=Normal, 1=ARP Spoofing).")

# ---------------- Drop duplicates ----------------
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print("Removed duplicates:", before - after)

# ---------------- Split features/target ----------------
X = df.drop(columns=["Label"])
y = df["Label"]

# Convert object columns to numeric if possible
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = pd.to_numeric(X[c], errors="coerce")

# ---------------- Handle missing values ----------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------------- Normalize ----------------
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Feature shape before selection:", X_scaled.shape)

# ---------------- Feature Selection (Correlation filter) ----------------
# Compute correlation matrix (features only)
corr_matrix = X_scaled.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Drop features with correlation > threshold
to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]
print("Features to drop due to correlation >", CORR_THRESHOLD, ":", len(to_drop))

X_selected = X_scaled.drop(columns=to_drop)
print("Feature shape after selection:", X_selected.shape)

# ---------------- Save processed dataset ----------------
proc_df = X_selected.copy()
proc_df["Label"] = y.values
proc_df.to_csv(PROC_OUT, index=False)
print("Saved processed dataset:", PROC_OUT)

# ---------------- Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train distribution:", Counter(y_train), " Test distribution:", Counter(y_test))

# ---------------- Train KNN ----------------
knn = KNeighborsClassifier(**KNN_PARAMS)
knn.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = knn.predict(X_test)
print("\n=== Classification Report (KNN after Correlation-based FS) ===\n")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# ---------------- Visualizations ----------------
# 1) Feature count before vs after
plt.figure(figsize=(6,4))
plt.bar(["Before FS","After FS"], [X_scaled.shape[1], X_selected.shape[1]], color=["steelblue","seagreen"])
plt.title("Feature Count Before vs After Correlation Filter")
plt.ylabel("Number of features")
plt.show()

# 2) Confusion matrix heatmap
plt.figure(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Confusion Matrix - KNN (ARP vs Normal, Corr Filter)")
plt.show()

#!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     roc_curve,
#     auc,
#     classification_report,
# )

# # --- 1. Define Features and Load Data ---

# # CORRECTED 15 features selected by the Correlation Filter method (Table 2)
# SELECTED_FEATURES = [
#     "Dst Port",
#     "Fwd PSH Flags",
#     "Fwd URG Flags",
#     # Corrected names for Bulk features:
#     "Fwd Bytes/Bulk Avg",  # Corresponds to Fwd Byts/b Avg in paper
#     "Fwd Packet/Bulk Avg", # Corresponds to Fwd Pkts/b Avg in paper
#     "Fwd Bulk Rate Avg",   # Corresponds to Fwd Blk Rate Avg in paper
#     "Bwd Bytes/Bulk Avg",  # Corresponds to Bwd Byts/b Avg in paper
#     "Bwd Packet/Bulk Avg", # Corresponds to Bwd Pkts/b Avg in paper
#     "Bwd Bulk Rate Avg",   # Corresponds to Bwd Blk Rate Avg in paper
#     # Remaining names (verified in previous steps):
#     "FWD Init Win Bytes", 
#     "Fwd Seg Size Min",
#     "Active Mean",
#     "Active Std",
#     "Active Max",
#     "Active Min",
# ]
# TARGET_LABEL = "Label"

# file_path = r"D:\Desktop\Sem_3\CEN\combined_arp_dataset.csv"
# df = pd.read_csv(file_path)

# # --- 2. Preprocessing and Feature Engineering ---

# # a. Drop known problematic/irrelevant columns as per the paper's preprocessing (Page 5, Section 3.2)
# IRRELEVANT_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
# df_processed = df.drop(columns=IRRELEVANT_COLS, errors='ignore')

# # Separate initial features and label
# X_all = df_processed.drop(columns=[TARGET_LABEL])
# y = df_processed[TARGET_LABEL]

# # b. Duplicate Removal
# initial_rows = len(X_all)
# X_all = X_all.drop_duplicates()
# y = y[X_all.index] 
# print(f"Removed {initial_rows - len(X_all)} duplicate rows.")

# # c. Missing Value Imputation (Replace Inf and NaNs with Median)
# X_all = X_all.replace([np.inf, -np.inf], np.nan)
# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X_imputed_array = imputer.fit_transform(X_all)
# X_imputed = pd.DataFrame(X_imputed_array, columns=X_all.columns)
# print("Missing values replaced with median (after converting Inf to NaN).")

# # --- 3. Feature Selection (Using the 15 Filtered Features) ---

# # Select only the 15 features defined at the start
# X_selected = X_imputed[SELECTED_FEATURES]
# print(f"Features reduced to: {len(X_selected.columns)} features.")

# # d. Normalization (Min-Max Normalization)
# # Note: Normalization is done AFTER feature selection but BEFORE splitting
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(X_selected)
# X_normalized = pd.DataFrame(X_normalized, columns=X_selected.columns)
# print("Selected features normalized using Min-Max Scaling.")


# # --- 4. Data Splitting (Stratified Split) ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X_normalized, y,
#     test_size=0.30, 
#     random_state=42, 
#     stratify=y 
# )
# print(f"\nData split into Train ({len(X_train)} samples) and Test ({len(X_test)} samples) with stratification.")

# # --- 5. Model Training (KNN Classifier) ---
# # Paper's KNN settings: n_neighbors=5, weights='uniform' (Table 1)
# knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
# knn.fit(X_train, y_train)

# # --- 6. Evaluation and Visualization ---

# # Predict on the test set
# y_pred = knn.predict(X_test)
# y_pred_proba = knn.predict_proba(X_test)[:, 1]

# # --- a. Confusion Matrix ---
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Normal (0)', 'Attack (1)'],
#             yticklabels=['Normal (0)', 'Attack (1)'])
# plt.title('Confusion Matrix for KNN (15 Filtered Features)')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.savefig('knn_filter_confusion_matrix.png')
# plt.close()
# print("Generated 'knn_filter_confusion_matrix.png'")


# # --- b. ROC Curve and AUC Score ---
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.title('ROC Curve for KNN (15 Filtered Features)')
# plt.legend(loc="lower right")
# plt.savefig('knn_filter_roc_auc_curve.png')
# plt.close()
# print("Generated 'knn_filter_roc_auc_curve.png'")


# # --- Final Metrics (Text Output) ---
# print("\n--- Model Evaluation Metrics (KNN with 15 Filter Features) ---")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(f"AUC Score: {roc_auc:.4f}")
# print("\nClassification Report (Includes Precision, Recall/Sensitivity, F-measure):")
# print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
