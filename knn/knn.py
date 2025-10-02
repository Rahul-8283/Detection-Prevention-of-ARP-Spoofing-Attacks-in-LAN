import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# --- 1. Data Loading ---
file_path = r"..\combined_arp_dataset.csv"
df = pd.read_csv(file_path)

# --- 2. Preprocessing Steps ---

# a. Feature Removal (Inappropriate Features) 
# The paper specifies removing: Flow_ID, Src_IP, Timestamp, and Dst_IP.
columns_to_drop = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
df_processed = df.drop(columns=columns_to_drop)

# Separate features (X) and label (y)
X = df_processed.drop(columns=["Label"])
y = df_processed["Label"]

# b. Duplicate Removal [cite: 174]
initial_rows = len(X)
X = X.drop_duplicates()
y = y[X.index] # Keep labels aligned with remaining features
print(f"Removed {initial_rows - len(X)} duplicate rows.")

# c. Missing Value Imputation (Replace NaNs with Median) 
# Convert infinite values to NaN first, as Min-Max Scaler handles them poorly.
X = X.replace([np.inf, -np.inf], np.nan)

# Instantiate the imputer to fill NaNs with the median value
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
print("Missing values replaced with median (after converting Inf to NaN).")

# d. Normalization (Min-Max Normalization) 
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_imputed)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
print("Features normalized using Min-Max Scaling.")

# --- 3. Data Splitting (Stratified Split) ---
# Stratification ensures the test set maintains the 80/20 class ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.30,
    random_state=42, # For reproducibility
    stratify=y        # Crucial for imbalanced data
)
print(f"\nData split into 70% Train ({len(X_train)} samples) and 30% Test ({len(X_test)} samples) with stratification.")

# --- 4. Model Training (KNN Classifier) ---
# The paper's KNN setting uses n_neighbors=5, weights='uniform'.
knn = KNeighborsClassifier(n_neighbors=1, weights='uniform')
knn.fit(X_train, y_train)

# --- 5. Evaluation and Visualization ---

# Predict on the test set
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1] # Probability of the positive class (1: Attack)

# --- a. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])
plt.title('Confusion Matrix for KNN Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('knn_confusion_matrix.png')
plt.close()
print("\nGenerated 'knn_confusion_matrix.png'")

# --- b. ROC Curve and AUC Score ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) / Sensitivity')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('knn_roc_auc_curve.png')
plt.close()
print("Generated 'knn_roc_auc_curve.png'")

# --- Final Metrics (Text Output) ---
print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print("\nClassification Report (Includes Precision, Recall/Sensitivity, F-measure):")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))
