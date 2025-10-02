import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import time 

# --- A. WRAPPER CONCEPTS FROM PAPER (Conceptual Definition) ---

# The paper's Fitness Function relies on KNN classification error.
def fitness_function(X_subset, y_train, y_val):
    """
    Implements the Fitness Function (Equation 1) from the paper.
    Fitness = alpha * gamma + (1 - alpha) * (R / N)
    Where gamma is the KNN classification error rate.
    """
    # 1. Configuration (Alpha set high to prioritize accuracy over feature count)
    alpha = 0.99 
    R = X_subset.shape[1]  # Number of selected features
    N = X_train_full.shape[1] # Total initial features (79)

    # 2. Performance (KNN used to guide selection)
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    # Use a small validation split (20%) of the training data for selection
    X_f_train, X_f_val, y_f_train, y_f_val = train_test_split(
        X_subset, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    
    knn.fit(X_f_train, y_f_train)
    error_rate = 1.0 - accuracy_score(y_f_val, knn.predict(X_f_val))
    
    gamma = error_rate
    
    # 3. Calculate Fitness
    fitness = (alpha * gamma) + ((1 - alpha) * (R / N))
    
    # NOTE: Wrapper seeks to MINIMIZE this score (lower error, fewer features)
    return fitness

# The full forward selection process (omitted due to complexity) iteratively 
# calls this fitness_function to find the optimal 39 features.
# The result of that simulated search is the WRAPPER_FEATURES list below.


# --- 1. Define Features and Load Data ---

# 39 features selected by the Wrapper Technique (Table 3) - CORRECTED NAMES
WRAPPER_FEATURES = [
    # General Flow & Port
    "Src Port", "Dst Port", "Protocol",
    # Packet Length Stats
    "Total Length of Fwd Packet",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    # IAT Stats
    "Fwd IAT Total",
    "Fwd IAT Std", "Fwd IAT Max", "Flow IAT Min",
    # Flags and Counts
    "Fwd PSH Flags", "Bwd URG Flags", "Fwd URG Flags", "Fwd Packets/s",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWR Flag Count", # Corrected from 'CWE Flag Count'
    "ECE Flag Count",
    # Segment & Bulk Stats
    "Fwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Fwd Packet/Bulk Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    # Subflow and Window
    "Subflow Fwd Bytes",
    "FWD Init Win Bytes",
    "Fwd Act Data Pkts",
    # Active/Idle Stats
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

# Separate initial features and label
X_all = df_processed.drop(columns=[TARGET_LABEL])
y = df_processed[TARGET_LABEL]

# Duplicate Removal
X_all = X_all.drop_duplicates()
y = y[X_all.index] 

# Missing Value Imputation (Median)
X_all = X_all.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_imputed_array = imputer.fit_transform(X_all)
X_imputed = pd.DataFrame(X_imputed_array, columns=X_all.columns)
X_train_full = X_imputed # Used as N in the fitness function concept

# --- 3. Feature Selection (Reduce to 39 Features) ---
X_selected = X_imputed[WRAPPER_FEATURES]
print(f"Features reduced to: {len(X_selected.columns)} features (Wrapper Technique Result).")

# Normalization (Min-Max Scaling)
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

# --- 5. Model Training (Random Forest Classifier) ---
# Paper's RF settings: criterion='entropy', min_samples_split=2, n_estimators=10 (Table 1)
# This is the step that should yield the paper's claimed 99.74% accuracy.
start_time = time.time()
rf = RandomForestClassifier(
    criterion='entropy', 
    min_samples_split=2, 
    n_estimators=10, 
    random_state=42
)
rf.fit(X_train, y_train)
end_time = time.time()
execution_time_ms = round((end_time - start_time) * 1000)

# --- 6. Evaluation ---
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
roc_auc = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])

# --- Final Metrics (Text Output) ---
print("\n--- Model Evaluation (RF with 39 Wrapper Features) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print(f"Execution Time (ms): {execution_time_ms}")
print("\nClassification Report (Compare against Paper's RF Wrapper: Acc 99.74%):")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
