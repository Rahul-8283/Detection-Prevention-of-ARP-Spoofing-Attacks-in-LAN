import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models

FEATURES = [
    # # General Flow & Port
    # "Src Port", "Dst Port", "Protocol",
    # # Packet Length Stats
    # "Total Length of Fwd Packet",
    # "Fwd Packet Length Max",
    # "Fwd Packet Length Min",
    # "Fwd Packet Length Mean",
    # "Fwd Packet Length Std",
    # # IAT Stats
    # "Fwd IAT Total",
    # "Fwd IAT Std", "Fwd IAT Max", "Flow IAT Min",
    # # Flags and Counts
    # "Fwd PSH Flags", "Bwd URG Flags", "Fwd URG Flags", "Fwd Packets/s",
    # "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    # "URG Flag Count", "CWR Flag Count", # Corrected from 'CWE Flag Count'
    # "ECE Flag Count",
    # # Segment & Bulk Stats
    # "Fwd Segment Size Avg",
    # "Fwd Bytes/Bulk Avg",
    # "Fwd Bulk Rate Avg",
    # "Fwd Packet/Bulk Avg",
    # "Bwd Bytes/Bulk Avg",
    # "Bwd Packet/Bulk Avg",
    # "Bwd Bulk Rate Avg",
    # # Subflow and Window
    # "Subflow Fwd Bytes",
    # "FWD Init Win Bytes",
    # "Fwd Act Data Pkts",
    # # Active/Idle Stats
    # "Active Mean",
    # "Fwd Seg Size Min",
    # "Active Std",
    # "Active Max",
    # "Active Min",
    # "Idle Std"
    "Src Port", "Dst Port", 
    "Fwd Packet Length Max", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Fwd Packet Length Std", 
    "Bwd Packet Length Mean", 
    "Flow IAT Std", "Bwd IAT Std", "Flow IAT Min", 
    "Fwd IAT Mean", "Bwd IAT Max", 
    "Bwd Header Length",
    "Fwd PSH Flags", "SYN Flag Count", "ECE Flag Count",
    "Fwd Bulk Rate Avg",
    "Fwd Bytes/Bulk Avg",
    "Subflow Bwd Bytes",
    "FWD Init Win Bytes",
    "Active Std", "Fwd Act Data Pkts", 
    "Idle Max"
]

# Load dataset
df = pd.read_csv(r"..\combined_arp_dataset.csv")

# Drop ID-like columns (adjust based on your dataset)
drop_candidates = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "source_file"]
df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

# Ensure Label exists
if "Label" not in df.columns:
    raise SystemExit("Dataset must contain 'Label' column.")

# Reduce to selected features
missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    raise SystemExit(f"Missing features in dataset: {missing_features}")

X = df[FEATURES]
y = df["Label"]

# Convert all columns to numeric
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = pd.to_numeric(X[c], errors="coerce")

# Handle missing values with median
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# NN Model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# Train
history = model.fit(
    X_train, y_train, epochs=30, batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
from sklearn.metrics import classification_report, confusion_matrix
print("\n=== NN Classification Report (Selected Features) ===")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
