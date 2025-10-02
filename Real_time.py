# import time
# import statistics
# from scapy.all import sniff, IP, TCP, UDP

# # ==========================
# # Flow Storage
# # ==========================
# flows = {}

# def get_flow_id(pkt):
#     """Return unique flow identifier"""
#     if IP in pkt:
#         proto = pkt[IP].proto
#         src = pkt[IP].src
#         dst = pkt[IP].dst
#         sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
#         dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
#         return (src, dst, sport, dport, proto)
#     return None

# def update_flow(pkt):
#     """Update flow stats for each packet"""
#     fid = get_flow_id(pkt)
#     if fid is None:
#         return

#     now = time.time()
#     length = len(pkt)

#     # Init flow
#     if fid not in flows:
#         flows[fid] = {
#             "src_port": fid[2],
#             "dst_port": fid[3],
#             "start": now,
#             "last": now,
#             "fwd_packets": 0,
#             "bwd_packets": 0,
#             "fwd_bytes": 0,
#             "bwd_bytes": 0,
#             "fwd_times": [],
#             "bwd_times": [],
#             "fwd_pkt_sizes": [],
#             "bwd_pkt_sizes": [],
#             "fwd_window_sizes": [],
#             "bwd_window_sizes": [],
#         }

#     f = flows[fid]

#     # Direction: forward if src matches
#     direction = "fwd" if pkt[IP].src == fid[0] else "bwd"

#     # Packet counts & sizes
#     f[f"{direction}_packets"] += 1
#     f[f"{direction}_bytes"] += length
#     f[f"{direction}_pkt_sizes"].append(length)

#     # Window size
#     if TCP in pkt:
#         f[f"{direction}_window_sizes"].append(pkt[TCP].window)

#     # IAT
#     iat = now - f["last"]
#     f[f"{direction}_times"].append(iat)

#     # Update last timestamp
#     f["last"] = now

# def extract_features(fid, f):
#     """Compute 15 selected features for one flow"""
#     duration = f["last"] - f["start"]
#     if duration == 0:
#         duration = 1e-6  # avoid div/0

#     features = {
#         "Src Port": f["src_port"],
#         "Dst Port": f["dst_port"],
#         "Flow IAT Min": min(f["fwd_times"] + f["bwd_times"]) if (f["fwd_times"]+f["bwd_times"]) else 0,
#         "FWD Init Win Bytes": f["fwd_window_sizes"][0] if f["fwd_window_sizes"] else 0,
#         "Fwd IAT Min": min(f["fwd_times"]) if f["fwd_times"] else 0,
#         "Flow Duration": duration,
#         "Flow Bytes/s": (f["fwd_bytes"]+f["bwd_bytes"]) / duration,
#         "Fwd IAT Total": sum(f["fwd_times"]) if f["fwd_times"] else 0,
#         "Bwd Init Win Bytes": f["bwd_window_sizes"][0] if f["bwd_window_sizes"] else 0,
#         "Fwd IAT Mean": statistics.mean(f["fwd_times"]) if f["fwd_times"] else 0,
#         "Bwd Packets/s": f["bwd_packets"] / duration,
#         "Packet Length Std": statistics.pstdev(f["fwd_pkt_sizes"]+f["bwd_pkt_sizes"]) if (f["fwd_pkt_sizes"]+f["bwd_pkt_sizes"]) else 0,
#         "Fwd Packets/s": f["fwd_packets"] / duration,
#         "Length of Fwd Packet": statistics.mean(f["fwd_pkt_sizes"]) if f["fwd_pkt_sizes"] else 0,
#         "Bwd Bulk Rate Avg": f["bwd_bytes"]/duration if f["bwd_packets"] else 0,
#     }
#     return features

# def packet_handler(pkt):
#     update_flow(pkt)

#     # Example: extract after every 10 packets in a flow
#     fid = get_flow_id(pkt)
#     if fid and flows[fid]["fwd_packets"] + flows[fid]["bwd_packets"] >= 10:
#         feats = extract_features(fid, flows[fid])
#         print("[FEATURES]", feats)

# # ==========================
# # Start Sniffing
# # ==========================
# print("[*] Capturing packets... Press CTRL+C to stop.")
# sniff(iface="Wi-Fi", prn=packet_handler, store=0)

import time
import statistics
import pandas as pd
import sys
from scapy.all import sniff, IP, TCP, UDP

# ==========================
# Flow Storage
# ==========================
flows = {}
csv_file = "live_flow_features.csv"
columns = [
    "Src Port","Dst Port","Flow IAT Min","FWD Init Win Bytes","Fwd IAT Min",
    "Flow Duration","Flow Bytes/s","Fwd IAT Total","Bwd Init Win Bytes",
    "Fwd IAT Mean","Bwd Packets/s","Packet Length Std","Fwd Packets/s",
    "Total Length of Fwd Packet","Bwd Bulk Rate Avg"
]

# Initialize CSV with header
pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

def get_flow_id(pkt):
    """Return unique flow identifier"""
    if IP in pkt:
        proto = pkt[IP].proto
        src = pkt[IP].src
        dst = pkt[IP].dst
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        return (src, dst, sport, dport, proto)
    return None

def update_flow(pkt):
    """Update flow stats for each packet"""
    fid = get_flow_id(pkt)
    if fid is None:
        return

    now = time.time()
    length = len(pkt)

    # Init flow
    if fid not in flows:
        flows[fid] = {
            "src_port": fid[2],
            "dst_port": fid[3],
            "start": now,
            "last": now,
            "fwd_packets": 0,
            "bwd_packets": 0,
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "fwd_times": [],
            "bwd_times": [],
            "fwd_pkt_sizes": [],
            "bwd_pkt_sizes": [],
            "fwd_window_sizes": [],
            "bwd_window_sizes": [],
        }

    f = flows[fid]

    # Direction: forward if src matches
    direction = "fwd" if pkt[IP].src == fid[0] else "bwd"

    # Packet counts & sizes
    f[f"{direction}_packets"] += 1
    f[f"{direction}_bytes"] += length
    f[f"{direction}_pkt_sizes"].append(length)

    # Window size
    if TCP in pkt:
        f[f"{direction}_window_sizes"].append(pkt[TCP].window)

    # IAT
    iat = now - f["last"]
    f[f"{direction}_times"].append(iat)

    # Update last timestamp
    f["last"] = now

def extract_features(fid, f):
    """Compute 15 selected features for one flow"""
    duration = f["last"] - f["start"]
    if duration == 0:
        duration = 1e-6  # avoid div/0

    features = {
        "Src Port": f["src_port"],
        "Dst Port": f["dst_port"],
        "Flow IAT Min": min(f["fwd_times"] + f["bwd_times"]) if (f["fwd_times"]+f["bwd_times"]) else 0,
        "FWD Init Win Bytes": f["fwd_window_sizes"][0] if f["fwd_window_sizes"] else 0,
        "Fwd IAT Min": min(f["fwd_times"]) if f["fwd_times"] else 0,
        "Flow Duration": duration,
        "Flow Bytes/s": (f["fwd_bytes"]+f["bwd_bytes"]) / duration,
        "Fwd IAT Total": sum(f["fwd_times"]) if f["fwd_times"] else 0,
        "Bwd Init Win Bytes": f["bwd_window_sizes"][0] if f["bwd_window_sizes"] else 0,
        "Fwd IAT Mean": statistics.mean(f["fwd_times"]) if f["fwd_times"] else 0,
        "Bwd Packets/s": f["bwd_packets"] / duration,
        "Packet Length Std": statistics.pstdev(f["fwd_pkt_sizes"]+f["bwd_pkt_sizes"]) if (f["fwd_pkt_sizes"]+f["bwd_pkt_sizes"]) else 0,
        "Fwd Packets/s": f["fwd_packets"] / duration,
        "Total Length of Fwd Packet": statistics.mean(f["fwd_pkt_sizes"]) if f["fwd_pkt_sizes"] else 0,
        "Bwd Bulk Rate Avg": f["bwd_bytes"]/duration if f["bwd_packets"] else 0,
    }
    return features

def packet_handler(pkt):
    update_flow(pkt)

    # Example: extract after every 10 packets in a flow
    fid = get_flow_id(pkt)
    if fid and flows[fid]["fwd_packets"] + flows[fid]["bwd_packets"] >= 10:
        feats = extract_features(fid, flows[fid])
        print("[FEATURES]", feats)

        # Save to CSV
        df = pd.DataFrame([feats])
        df.to_csv(csv_file, mode="a", header=False, index=False)

        # Reset flow so we don’t save duplicates endlessly
        flows[fid]["fwd_packets"] = 0
        flows[fid]["bwd_packets"] = 0
        flows[fid]["fwd_times"].clear()
        flows[fid]["bwd_times"].clear()
        flows[fid]["fwd_pkt_sizes"].clear()
        flows[fid]["bwd_pkt_sizes"].clear()

# ==========================
# Start Sniffing
# ==========================
# Get duration from command line argument, default to 60 seconds
if len(sys.argv) > 1:
    try:
        duration = int(sys.argv[1])
    except ValueError:
        print("Error: Duration must be a valid integer (seconds)")
        sys.exit(1)
else:
    duration = 60  # default duration

print(f"[*] Capturing packets for {duration} seconds... saving features to {csv_file}")
sniff(iface="Wi-Fi", prn=packet_handler, store=0, timeout=duration)
print("[*] Sniffing finished.")
