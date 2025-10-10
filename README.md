## ARP Spoofing Detection and Prevention System

This project implements a real-time ARP spoofing detection system using machine learning techniques to analyze network traffic and identify potential attacks.

## 🚀 **NEW: Web-based Real-time Detection Interface**

The system now includes a user-friendly web interface for real-time network monitoring and threat detection:

### **Features:**
- **Real-time packet capture** with configurable duration (10-3600 seconds)
- **ML-powered prediction** using trained XGBoost model  
- **Risk assessment** with confidence scores
- **Visual dashboard** showing threat levels and statistics
- **CSV export** functionality for analysis
- **Professional UI** with color-coded risk indicators

### **How to Use:**
1. Run the web application: `python app.py`
2. Open your browser to `http://127.0.0.1:5000`
3. Enter monitoring duration and click "Start Network Monitoring"
4. View results with predicted threat categories and risk levels
5. Download results as CSV for further analysis

### **Model Performance:**
- Uses 15 selected network flow features
- Trained on IoTID20 dataset
- Detects multiple attack types: ARP Spoofing, DDoS, Mirai botnet, etc.
- Provides confidence scores for each prediction

---

## Computer Networks S3

1) what will be the role of wireshark and how will you use it --> wireshark at attacker, using hub
ans : Wireshark is a network protocol analyzer — basically, a tool that lets you capture and examine the data packets moving through a network in real time.
### Monitoring Traffic Between 2 Other Devices (in a LAN)
	Minimum 3 devices:
	Device A – first endpoint (e.g., victim).
	Device B – second endpoint (e.g., router/server).
	Device C – your device with Wireshark (attacker/observer).

	Your device (C) must be placed so it can see A↔B’s traffic — either by:
		1. Being a hub connection, Using port mirroring on a switch, or
		2. Doing ARP spoofing so all traffic passes through C.

2) What is your input -->
ans : 
just for an example we can send a text file, with some simple message, "This is Team - 7 form AIE-C !"

3) What is the approach, was it code or simulation -->
ans : 
simulation via virtual environment [virtual box]

4) What is mean by Ettercap, what is the use of it --> 
ans : 
### Ettercap is an open-source network security tool mainly used for Man-in-the-Middle (MitM) attacks on local area networks.
### What Ettercap Does -> Performs ARP spoofing/poisoning to trick devices into sending traffic through the attacker’s system.
### Captures (sniffs) all network packets passing through the attacker.
### Modifies packets in real time if needed.
### Supports both active attacks (altering traffic) and passive sniffing (just observing traffic).
### Works on many protocols: HTTP, HTTPS (with SSL stripping), FTP, SSH (if unprotected), DNS, etc.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### ARP is used to map receiver's IP address to MAC address.
1. Your PC (192.168.1.5) wants to talk to the router (192.168.1.1).
2. Your PC sends an ARP request:
3. “Who has IP 192.168.1.1? Tell me your MAC address.”
4. The router replies:
5. “192.168.1.1 is at MAC AA:AA:AA:AA:AA:AA.”
6. Your PC stores this in its ARP table for future use.

Weakness --> has no authentication 
--> If your PC receives an ARP reply, it will usually trust it and update the ARP table.

### -> # Both sender and receiver will give both ARP request, and response
Case 1 -->
Victim: 192.168.1.5 (PC)
Router: 192.168.1.1
Attacker: 192.168.1.50

Steps attacker takes:
1. Attacker sends fake ARP replies to the victim:
	“192.168.1.1 (router) is at MAC BB:BB:BB:BB:BB:BB” (attacker’s MAC).
2. Attacker also sends fake ARP replies to the router:
	“192.168.1.5 (victim) is at MAC BB:BB:BB:BB:BB:BB” (attacker’s MAC).

Result --> All traffic between victim and router passes through the attacker first.
Further perform several attacks --> Man-in-the-Middle, modify data, drop data.


## Old with no protection
http, ftp, telnet -> uses plain text -> no encryption while sending data
DIR - STOPS
