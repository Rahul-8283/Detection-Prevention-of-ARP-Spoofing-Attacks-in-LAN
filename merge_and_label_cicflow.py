# merge_and_label_cicflow.py
import pandas as pd
import glob
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# -------------- configuration ----------------
input_pattern = "output/*_Flow.csv"   # glob for CICFlowMeter CSV output files
# If CICFlowMeter gave .xlsx, convert to csv first or change pattern to .xlsx and use pd.read_excel
output_file = "final_iotid20_like_dataset.csv"

# mapping rules based on filename substrings:
def map_labels(filename):
    fn = filename.lower()
    if "benign" in fn or "normal" in fn:
        return ("Normal", "Benign", "Normal")
    if "mitm-arpspoof" in fn or "arpspoof" in fn or "arp" in fn:
        return ("Attack", "Man in the Middle (MITM)", "ARP Spoofing")
    if "dos-syn" in fn or "synflood" in fn:
        return ("Attack", "Denial of Service (DoS)", "SYN Flooding")
    if "mirai-http" in fn or "httpflood" in fn:
        return ("Attack", "Mirai Botnet", "HTTP Flooding")
    if "mirai-udp" in fn or "udpflood" in fn:
        return ("Attack", "Mirai Botnet", "UDP Flooding")
    if "mirai-ack" in fn or "ackflood" in fn:
        return ("Attack", "Mirai Botnet", "ACK Flooding")
    if "mirai-hostbruteforce" in fn or "hostbrute" in fn:
        return ("Attack", "Mirai Botnet", "Host Brute Force")
    if "scan-hostport" in fn:
        return ("Attack", "Port Scanning", "Host Port Scan")
    if "scan-portos" in fn or "scan-portos" in fn:
        return ("Attack", "Port Scanning", "Port OS Scan")
    # fallback
    return ("Attack", "Other", os.path.basename(filename))

# -------------- load and merge ----------------
all_files = glob.glob(input_pattern)
if not all_files:
    print("No files found. Check your input_pattern:", input_pattern)
    raise SystemExit

dfs = []
for f in all_files:
    print("Loading", f)
    # attempt to read CSV; if Excel, read with read_excel
    if f.lower().endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    # Map labels
    label, category, subcat = map_labels(os.path.basename(f))
    df['Label'] = label
    df['Category'] = category
    df['Sub-category'] = subcat
    # Some CICFlowMeter outputs include 'NeedManualLabel' or 'Label' column prefilled - we overwrite deliberately
    dfs.append(df)

merged = pd.concat(dfs, axis=0, ignore_index=True)
print("Merged rows:", merged.shape)

# -------------- drop identifier columns (like paper) ----------------
# The paper drops Flow ID, Src IP, Dst IP, Timestamp — adapt names to what CICFlowMeter uses
cols_to_drop = []
for c in ['Flow ID','Src IP','Dst IP','Timestamp','Fwd Header Length.1','Bwd Header Length.1']:
    if c in merged.columns:
        cols_to_drop.append(c)
print("Dropping columns:", cols_to_drop)
merged = merged.drop(columns=cols_to_drop, errors='ignore')

# -------------- handle missing values ----------------
# Replace textual NaNs with real NaN
merged = merged.replace(['',' '], pd.NA)
# Impute numeric columns with median
num_cols = merged.select_dtypes(include=['int64','float64']).columns.tolist()
print("Numeric columns count:", len(num_cols))
if num_cols:
    imputer = SimpleImputer(strategy='median')
    merged[num_cols] = imputer.fit_transform(merged[num_cols])

# -------------- normalization (min-max) ----------------
feature_cols = [c for c in merged.columns if c not in ['Label','Category','Sub-category']]
print("Feature columns:", len(feature_cols))
scaler = MinMaxScaler()
merged[feature_cols] = scaler.fit_transform(merged[feature_cols])

# -------------- write final CSV ----------------
merged.to_csv(output_file, index=False)
print("Wrote", output_file)
