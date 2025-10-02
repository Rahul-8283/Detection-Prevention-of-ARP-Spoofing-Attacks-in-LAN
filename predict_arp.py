#!/usr/bin/env python3
"""
Real-time ARP Spoofing Prediction using trained XGBoost model
"""

import pandas as pd
import numpy as np
import joblib
import os

def predict_from_csv(csv_file="live_flow_features.csv"):
    """
    Load the live flow features and make predictions using the trained model
    
    Returns:
        List of dictionaries containing predictions and confidence scores
    """
    
    # File paths for the trained models and preprocessors
    MODEL_PATH = "xgb_mcc_Sub-category-selected.pkl"
    SCALER_PATH = "xgb_scaler_Sub-category.pkl"
    IMPUTER_PATH = "xgb_imputer_Sub-category.pkl"
    LABEL_ENCODER_PATH = "xgb_label_encoder_Sub-category.pkl"
    
    # Check if all required files exist
    required_files = [MODEL_PATH, SCALER_PATH, IMPUTER_PATH, LABEL_ENCODER_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found!")
            return []
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        return []
    
    try:
        # Load the trained model and preprocessors
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        # Load the CSV data
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("Warning: CSV file is empty!")
            return []
        
        # Expected feature names (must match training data)
        expected_features = [
            "Src Port", "Dst Port", "Flow IAT Min", "FWD Init Win Bytes", "Fwd IAT Min",
            "Flow Duration", "Flow Bytes/s", "Fwd IAT Total", "Bwd Init Win Bytes",
            "Fwd IAT Mean", "Bwd Packets/s", "Packet Length Std", "Fwd Packets/s",
            "Total Length of Fwd Packet", "Bwd Bulk Rate Avg"
        ]
        
        # Check if all expected features are present
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing features in CSV: {missing_features}")
            return []
        
        # Select only the required features
        X = df[expected_features].copy()
        
        # Convert to numeric (same preprocessing as training)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce")
        
        # Apply the same preprocessing as training
        X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
        X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X.columns)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        prediction_probabilities = model.predict_proba(X_scaled)
        
        # Convert predictions back to original labels
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        # Prepare results
        results = []
        for i in range(len(predictions)):
            max_prob = np.max(prediction_probabilities[i])
            confidence = f"{max_prob:.4f}"
            
            result = {
                "Row": i + 1,
                "Src_Port": int(X.iloc[i]["Src Port"]),
                "Dst_Port": int(X.iloc[i]["Dst Port"]),
                "Flow_Duration": f"{X.iloc[i]['Flow Duration']:.6f}",
                "Predicted_Category": predicted_labels[i],
                "Confidence": confidence,
                "Risk_Level": "HIGH" if predicted_labels[i] != "Benign" else "LOW"
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []

def main():
    """Test the prediction function"""
    results = predict_from_csv()
    
    if results:
        print(f"Found {len(results)} predictions:")
        for result in results:
            print(f"Row {result['Row']}: {result['Predicted_Category']} (Confidence: {result['Confidence']}, Risk: {result['Risk_Level']})")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()