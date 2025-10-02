import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# --- CONFIGURATION ---
TARGET_LABEL = "Label"
file_path = r"..\combined_arp_dataset.csv"
# CORRECTED 15 features selected by the Correlation Filter method (Table 2)
SELECTED_FEATURES = [
    "Dst Port", "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Bytes/Bulk Avg",  # Corresponds to Fwd Byts/b Avg in paper
    "Fwd Packet/Bulk Avg", # Corresponds to Fwd Pkts/b Avg in paper
    "Fwd Bulk Rate Avg",   # Corresponds to Fwd Blk Rate Avg in paper
    "Bwd Bytes/Bulk Avg",  # Corresponds to Bwd Byts/b Avg in paper
    "Bwd Packet/Bulk Avg", # Corresponds to Bwd Pkts/b Avg in paper
    "Bwd Bulk Rate Avg",   # Corresponds to Bwd Blk Rate Avg in paper
    "FWD Init Win Bytes", 
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
]

# --- 1. Data Loading and Preprocessing (Pipeline) ---
df = pd.read_csv(file_path)

# a. Drop irrelevant columns
IRRELEVANT_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
df_processed = df.drop(columns=IRRELEVANT_COLS, errors='ignore')

# Separate initial features and label
X_all = df_processed.drop(columns=[TARGET_LABEL])
y = df_processed[TARGET_LABEL]

# b. Duplicate Removal
X_all = X_all.drop_duplicates()
y = y[X_all.index] 

# c. Missing Value Imputation (Replace Inf and NaNs with Median)
X_all = X_all.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_imputed_array = imputer.fit_transform(X_all)
X_imputed = pd.DataFrame(X_imputed_array, columns=X_all.columns)
print("Preprocessing complete.")

# --- 3. Feature Selection (Reduce to 15 Features) ---
X_selected = X_imputed[SELECTED_FEATURES]
print(f"Features reduced to: {len(X_selected.columns)} features.")

# d. Normalization (Min-Max Normalization)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_selected)
X_normalized = pd.DataFrame(X_normalized, columns=X_selected.columns)

# --- 4. Data Splitting (Stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.30, 
    random_state=42, 
    stratify=y 
)

# --- 5. SMOTE Implementation (On Training Data Only) ---

# Visualization BEFORE SMOTE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Class Balance BEFORE SMOTE')
plt.bar(Counter(y_train).keys(), Counter(y_train).values(), color=['skyblue', 'red'])
plt.xticks([0, 1], ['Normal (0)', 'Attack (1)'])
plt.ylabel('Number of Samples')

# Apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"\nOriginal Training Samples: {len(X_train)}")
print(f"Resampled Training Samples (After SMOTE): {len(X_res)}")
print(f"New Class Distribution: {Counter(y_res)}")

# Visualization AFTER SMOTE
plt.subplot(1, 2, 2)
plt.title('Class Balance AFTER SMOTE')
plt.bar(Counter(y_res).keys(), Counter(y_res).values(), color=['skyblue', 'red'])
plt.xticks([0, 1], ['Normal (0)', 'Attack (1)'])
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('dt_15_smote_balance.png')
plt.show()

# --- 6. Model Training and Evaluation ---
dt = DecisionTreeClassifier(
    criterion='entropy', 
    splitter='best', 
    min_samples_split=2, 
    random_state=42
)
# Train on RESAMPLED data
dt.fit(X_res, y_res)

# Evaluate on ORIGINAL test data
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)[:, 1]

# Metrics and ROC/AUC
roc_auc = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])
print("\n--- Model Evaluation (DT with 15 Filtered Features + SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))

# ROC Curve Visualization
plt.figure(figsize=(6, 5))
plt.plot(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1], 
         color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve: DT (15 Filtered Features + SMOTE)')
plt.legend(loc="lower right")
plt.savefig('dt_15_smote_roc.png')
plt.show()
