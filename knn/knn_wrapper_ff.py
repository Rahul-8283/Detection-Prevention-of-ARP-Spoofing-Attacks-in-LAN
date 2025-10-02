import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
import time

# --- A. WRAPPER CONCEPTS FROM PAPER (Conceptual Definition) ---

def fitness_function(X_subset, y_train, y_val):
    """
    Implements the Fitness Function (Equation 1) from the paper.
    Fitness = alpha * gamma + (1 - alpha) * (R / N)
    Where gamma is the KNN classification error rate.
    """
    alpha = 0.99
    R = X_subset.shape[1]  # Number of selected features
    N = X_train_full.shape[1] # Total initial features

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    X_f_train, X_f_val, y_f_train, y_f_val = train_test_split(
        X_subset, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    knn.fit(X_f_train, y_f_train)
    error_rate = 1.0 - accuracy_score(y_f_val, knn.predict(X_f_val))
    gamma = error_rate
    fitness = (alpha * gamma) + ((1 - alpha) * (R / N))
    return fitness

# --- 1. Define Features and Load Data ---

WRAPPER_FEATURES = [
    "Src Port", "Dst Port", "Protocol",
    "Total Length of Fwd Packet",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Fwd IAT Total",
    "Fwd IAT Std", "Fwd IAT Max", "Flow IAT Min",
    "Fwd PSH Flags", "Bwd URG Flags", "Fwd URG Flags", "Fwd Packets/s",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWR Flag Count",
    "ECE Flag Count",
    "Fwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Fwd Packet/Bulk Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "Subflow Fwd Bytes",
    "FWD Init Win Bytes",
    "Fwd Act Data Pkts",
    "Active Mean",
    "Fwd Seg Size Min",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Std"
]
TARGET_LABEL = "Label"
file_path = r"..\combined_arp_dataset.csv"
df = pd.read_csv(file_path)

# --- 2. Preprocessing and Feature Engineering ---
IRRELEVANT_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
df_processed = df.drop(columns=IRRELEVANT_COLS, errors='ignore')
X_all = df_processed.drop(columns=[TARGET_LABEL])
y = df_processed[TARGET_LABEL]
X_all = X_all.drop_duplicates()
y = y[X_all.index]
X_all = X_all.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_imputed_array = imputer.fit_transform(X_all)
X_imputed = pd.DataFrame(X_imputed_array, columns=X_all.columns)
X_train_full = X_imputed # Used as N in the fitness function concept

# --- 3. Feature Selection (Reduce to 39 Features) ---
X_selected = X_imputed[WRAPPER_FEATURES]
print(f"Features reduced to: {len(X_selected.columns)} features (Wrapper Technique Result).")
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

# --- 5. Model Training (KNN Classifier) ---
start_time = time.time()
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski'
)
knn.fit(X_train, y_train)
end_time = time.time()
execution_time_ms = round((end_time - start_time) * 1000)

# --- 6. Evaluation ---
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]
roc_auc = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])

print("\n--- Model Evaluation (KNN with 39 Wrapper Features) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print(f"Execution Time (ms): {execution_time_ms}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
