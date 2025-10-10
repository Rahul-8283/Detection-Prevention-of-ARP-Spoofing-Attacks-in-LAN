from flask import Flask, render_template, request, redirect, url_for, send_file
import subprocess
import sys
import pandas as pd
import time
import os
import joblib
import numpy as np
from predict_arp import predict_from_csv

app = Flask(__name__)

CSV_FILE = "iotid20_dataset.csv"

MODEL_PATH = "xgb_mcc_Sub-category-selected.pkl"
SCALER_PATH = "xgb_scaler_Sub-category.pkl" 
IMPUTER_PATH = "xgb_imputer_Sub-category.pkl"
LABEL_ENCODER_PATH = "xgb_label_encoder_Sub-category.pkl"
EXAMPLE_MAP = {1: 1, 2: 66685, 3: 75645, 4: 64640}


_model = None
_scaler = None
_imputer = None
_label_encoder = None


def load_model_artifacts():
    global _model, _scaler, _imputer, _label_encoder
    if _model is not None:
        return True

    required = [MODEL_PATH, SCALER_PATH, IMPUTER_PATH, LABEL_ENCODER_PATH]
    for p in required:
        if not os.path.exists(p):
            print(f"Model artifact not found: {p}")
            return False

    try:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _imputer = joblib.load(IMPUTER_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False


def predict_single_row_from_series(row_series):
    if not load_model_artifacts():
        return {"error": "Model artifacts not available"}

    expected_features = [
        "Src Port", "Dst Port", "Flow IAT Min", "FWD Init Win Bytes", "Fwd IAT Min",
        "Flow Duration", "Flow Bytes/s", "Fwd IAT Total", "Bwd Init Win Bytes",
        "Fwd IAT Mean", "Bwd Packets/s", "Packet Length Std", "Fwd Packets/s",
        "Total Length of Fwd Packet", "Bwd Bulk Rate Avg"
    ]

    row_df = pd.DataFrame([row_series])
    missing = [f for f in expected_features if f not in row_df.columns]
    if missing:
        return {"error": f"Missing features in input row: {missing}"}

    X = row_df[expected_features].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    try:
        X_imputed = pd.DataFrame(_imputer.transform(X), columns=X.columns)
        X_scaled = pd.DataFrame(_scaler.transform(X_imputed), columns=X.columns)

        preds = _model.predict(X_scaled)
        probs = _model.predict_proba(X_scaled)
        label = _label_encoder.inverse_transform(preds)[0]
        confidence = float(np.max(probs[0]))

        features_dict = {col: (float(X.iloc[0][col]) if pd.notnull(X.iloc[0][col]) else None) for col in X.columns}

        result = {
            "Row": int(row_series.name) + 1 if hasattr(row_series, 'name') else 1,
            "Src_Port": int(X.iloc[0]["Src Port"]),
            "Dst_Port": int(X.iloc[0]["Dst Port"]),
            "Flow_Duration": f"{X.iloc[0]['Flow Duration']:.6f}",
            "Predicted_Category": label,
            "Confidence": f"{confidence:.4f}",
            "features": features_dict
        }
        return result
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sniff', methods=['POST'])
def sniff():
    try:
        duration = int(request.form.get('duration', 60))
    except Exception:
        duration = 60

    real_time_script = os.path.join(os.path.dirname(__file__), "Real_time.py")
    if not os.path.exists(real_time_script):
        return render_template('single_result.html', error=f"Real_time.py not found: {real_time_script}", result=None)

    try:
        completed = subprocess.run([sys.executable, real_time_script, str(duration)], check=False)
    except Exception as e:
        return render_template('single_result.html', error=f"Failed to launch Real_time.py: {e}", result=None)

    if completed.returncode != 0:
        print(f"Real_time.py exited with code {completed.returncode} (see terminal output for details)")

    if not os.path.exists("live_flow_features.csv"):
        return render_template('single_result.html', error="live_flow_features.csv not created by Real_time.py", result=None)

    try:
        predictions = predict_from_csv("live_flow_features.csv")
        if predictions:
            pd.DataFrame(predictions).to_csv("live_flow_predictions.csv", index=False)
        return render_template('results.html', rows=predictions)
    except Exception as e:
        return render_template('single_result.html', error=f"Prediction failed: {e}", result=None)
    return render_template('index.html')


@app.route('/predict_row', methods=['POST'])
def predict_row():
    row_num = int(request.form.get('row_num', 1))

    if not os.path.exists(CSV_FILE):
        return render_template('single_result.html', error="CSV file not found", result=None)

    try:
        df = pd.read_csv(CSV_FILE)
        if row_num < 1 or row_num > len(df):
            return render_template('single_result.html', error=f"Row number out of range (1-{len(df)})", result=None)

        row_series = df.iloc[row_num - 1]
        row_series.name = row_num - 1
        result = predict_single_row_from_series(row_series)

        if result.get('error'):
            return render_template('single_result.html', error=result.get('error'), result=None)

        return render_template('single_result.html', result=result, error=None)

    except Exception as e:
        return render_template('single_result.html', error=str(e), result=None)


@app.route('/predict_example', methods=['POST'])
def predict_example():
    example_id = int(request.form.get('example_id', 1))

    chosen = EXAMPLE_MAP.get(example_id, 1)
    try:
        df = pd.read_csv(CSV_FILE)
        if chosen < 1 or chosen > len(df):
            return render_template('single_result.html', error=f"Example row {chosen} out of range", result=None)

        row_series = df.iloc[chosen - 1]
        row_series.name = chosen - 1
        result = predict_single_row_from_series(row_series)

        if result.get('error'):
            return render_template('single_result.html', error=result.get('error'), result=None)

        return render_template('single_result.html', result=result, error=None)
    except Exception as e:
        return render_template('single_result.html', error=str(e), result=None)

@app.route('/results')
def results():
    if os.path.exists(CSV_FILE):
        predictions = predict_from_csv(CSV_FILE)
        
        df = pd.read_csv(CSV_FILE)
        raw_features = df.to_dict(orient="records")
        
        display_data = predictions.copy() if predictions else raw_features
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv("live_flow_predictions.csv", index=False)
        elif raw_features:
            pd.DataFrame(raw_features).to_csv("live_flow_features.csv", index=False)
        
        data = display_data
    else:
        data = []
    
    return render_template('results.html', rows=data)

@app.route('/download')
def download():
    predictions_file = "live_flow_predictions.csv"
    if os.path.exists(predictions_file):
        return send_file(predictions_file, as_attachment=True, download_name="arp_spoofing_predictions.csv")
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
