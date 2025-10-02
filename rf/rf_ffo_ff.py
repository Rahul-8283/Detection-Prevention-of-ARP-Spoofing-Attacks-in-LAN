import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import time 

# --- A. WRAPPER/FFO CONCEPTS (Conceptual Definition) ---

# The paper's Fitness Function relies on KNN classification error.
def fitness_function(X_subset, y_train):
    """
    Implements the Fitness Function (Equation 1) from the paper, guided by KNN.
    The function returns a score to be MINIMIZED by the FFO algorithm.
    """
    alpha = 0.99 
    R = X_subset.shape[1]  # Number of selected features
    N = 79                 # Total initial features (after dropping 4 irrelevant columns)

    # Use a small validation split (20%) of the training data for selection 
    # to evaluate the feature subset performance (gamma).
    X_f_train, X_f_val, y_f_train, y_f_val = train_test_split(
        X_subset, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_f_train, y_f_train)
    error_rate = 1.0 - accuracy_score(y_f_val, knn.predict(X_f_val))
    
    gamma = error_rate
    
    # Fitness = (alpha * Error_Rate) + ((1 - alpha) * (Feature_Count / Total_Features))
    fitness = (alpha * gamma) + ((1 - alpha) * (R / N))
    
    return fitness

# --- 1. Define Features and Load Data ---

# 23 features selected by the FFO Technique (Table 4) - CORRECTED NAMES
FFO_FEATURES = [
    "Src Port", "Dst Port", 
    "Fwd Packet Length Max", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Fwd Packet Length Std", 
    "Bwd Packet Length Mean", 
    "Flow IAT Std", "Bwd IAT Std", "Flow IAT Min", 
    "Fwd IAT Mean", "Bwd IAT Max", 
    "Bwd Header Length", # Corrected from 'Bwd_Header_Len'
    "Fwd PSH Flags", "SYN Flag Count", "ECE Flag Count",
    "Fwd Bulk Rate Avg", # Corresponds to Fwd_Blk_Rate_Avg in paper
    "Fwd Bytes/Bulk Avg", # Corresponds to Fwd_Byts/b_Avg in paper
    "Subflow Bwd Bytes",
    "FWD Init Win Bytes",
    "Active Std", "Fwd Act Data Pkts", 
    "Idle Max"
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

# --- 3. Feature Selection (Reduce to 23 Features) ---
# This line should now work with the corrected feature list:
X_selected = X_imputed[FFO_FEATURES]
print(f"Features reduced to: {len(X_selected.columns)} features (FFO Technique Result).")

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
print("\n--- Model Evaluation (RF with 23 FFO Features) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print(f"Execution Time (ms): {execution_time_ms}")
print("\nClassification Report (Compare against Paper's RF FFO: Acc 99.53%):")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
