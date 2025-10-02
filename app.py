from flask import Flask, render_template, request, redirect, url_for, send_file
import subprocess
import pandas as pd
import time
import os
from predict_arp import predict_from_csv

app = Flask(__name__)

CSV_FILE = "live_flow_features.csv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sniff', methods=['POST'])
def sniff():
    duration = int(request.form['duration'])  # time in seconds from form input
    
    # Run your packet sniffing script for given duration
    # Assuming your sniff script is called "sniffer.py"
    subprocess.Popen(["python", "Real_time.py", str(duration)])
    
    # Wait for sniffing to complete (optional if subprocess handles it)
    time.sleep(duration + 2)
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    if os.path.exists(CSV_FILE):
        # Get predictions using the trained model
        predictions = predict_from_csv(CSV_FILE)
        
        # Also get raw features for display
        df = pd.read_csv(CSV_FILE)
        raw_features = df.to_dict(orient="records")
        
        # Create display data (for web interface) - keep predictions clean
        display_data = predictions.copy() if predictions else raw_features
        
        # Save CSV for download - keep it clean and simple
        if predictions:
            # For CSV download, save only the clean prediction results
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv("live_flow_predictions.csv", index=False)
        elif raw_features:
            # If no predictions available, save raw features as fallback
            pd.DataFrame(raw_features).to_csv("live_flow_features", index=False)
        
        data = display_data
    else:
        data = []
    
    return render_template('results.html', rows=data)

@app.route('/download')
def download():
    """Download the predictions CSV file"""
    predictions_file = "live_flow_predictions.csv"
    if os.path.exists(predictions_file):
        return send_file(predictions_file, as_attachment=True, download_name="arp_spoofing_predictions.csv")
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
