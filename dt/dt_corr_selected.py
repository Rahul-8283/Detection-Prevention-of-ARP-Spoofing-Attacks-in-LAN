import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# --- 1. Define Features and Load Data ---

# 15 features selected by the Correlation Filter method (Table 2)
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
TARGET_LABEL = "Label"
file_path = r"..\combined_arp_dataset.csv"
df = pd.read_csv(file_path)

# --- 2. Preprocessing and Feature Engineering ---

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

# --- 3. Feature Selection (Reduce to 15 Features) ---

# Select only the 15 features defined at the start
X_selected = X_imputed[SELECTED_FEATURES]
print(f"Features reduced to: {len(X_selected.columns)} features.")

# d. Normalization (Min-Max Normalization)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_selected)
X_normalized = pd.DataFrame(X_normalized, columns=X_selected.columns)
print("Selected features normalized using Min-Max Scaling.")


# --- 4. Data Splitting (Stratified Split) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.30, 
    random_state=42, 
    stratify=y 
)

# --- 5. Model Training (Decision Tree Classifier) ---
# Paper's DT settings: criterion='entropy', splitter='best', min_samples_split=2 (Table 1)
dt = DecisionTreeClassifier(
    criterion='entropy', 
    splitter='best', 
    min_samples_split=2, 
    random_state=42
)
dt.fit(X_train, y_train)

# --- 6. Evaluation ---
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)[:, 1]

# Metrics
roc_auc = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])
print("\n--- Model Evaluation (Decision Tree with 15 Filtered Features) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix: DT (15 Filtered Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show() # Display the plot
