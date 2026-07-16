import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SIMULATING TELEMETRY & TRANSACTION DATA
# ==========================================
def generate_mock_data():
    np.random.seed(42)
    base_time = datetime.now()
    
    # User base
    users = [f"USER_{i:04d}" for i in range(1, 101)]
    
    # Generate Cyber Telemetry (Logins)
    telemetry_records = []
    for user in users:
        # Normal logins
        for _ in range(3):
            telemetry_records.append({
                "user_id": user,
                "timestamp": base_time - timedelta(minutes=np.random.randint(10, 300)),
                "ip_risk_score": np.random.uniform(0.0, 0.2), # Low risk IP
                "is_new_device": 0
            })
    
    # Add a couple of malicious security anomalies (e.g., suspicious login)
    telemetry_records.append({
        "user_id": "USER_0042",
        "timestamp": base_time - timedelta(minutes=5),
        "ip_risk_score": 0.95, # High risk IP / VPN
        "is_new_device": 1     # Brand new device
    })

    # Generate Transaction Behaviour
    transaction_records = []
    for user in users:
        # Normal transactions
        for _ in range(2):
            transaction_records.append({
                "user_id": user,
                "tx_id": f"TX_{np.random.randint(100000, 999999)}",
                "timestamp": base_time - timedelta(minutes=np.random.randint(5, 290)),
                "amount": np.random.uniform(100, 5000),
                "is_new_beneficiary": np.random.choice([0, 1], p=[0.9, 0.1])
            })
            
    # Add a fraudulent transaction corresponding to the suspicious login
    transaction_records.append({
        "user_id": "USER_0042",
        "tx_id": "TX_FRAUD_99",
        "timestamp": base_time - timedelta(minutes=2), # Occurred 3 mins after the bad login
        "amount": 95000,                                # Abnormally high amount
        "is_new_beneficiary": 1
    })
    
    return pd.DataFrame(telemetry_records), pd.DataFrame(transaction_records)

# ==========================================
# 2. THE CORRELATION ENGINE
# ==========================================
def correlate_streams(df_telemetry, df_transactions):
    print("[*] Correlating streams using temporal and user joins...")
    
    # Sort data for window merging
    df_telemetry = df_telemetry.sort_values('timestamp')
    df_transactions = df_transactions.sort_values('timestamp')
    
    # Perform a backward merge: Link each transaction to the closest preceding login from that user
    correlated = pd.merge_asof(
        df_transactions,
        df_telemetry,
        on='timestamp',
        by='user_id',
        direction='backward',
        suffixes=('_tx', '_telemetry')
    )
    
    # Calculate time delta between login and transaction (in minutes)
    correlated['time_delta_minutes'] = (correlated['timestamp'] - correlated['timestamp_telemetry']).dt.total_seconds() / 60.0
    
    # Handle transactions with no preceding telemetry in the dataset window
    correlated.fillna({'ip_risk_score': 0.1, 'is_new_device': 0, 'time_delta_minutes': 999}, inplace=True)
    
    return correlated

# ==========================================
# 3. AI DETECTION MODEL
# ==========================================
def train_and_detect_anomalies(df_features):
    print("[*] Extracting feature vectors and training AI Core...")
    
    # Define features for the AI model (combining cybersecurity + transactional behavior metrics)
    feature_cols = ['amount', 'is_new_beneficiary', 'ip_risk_score', 'is_new_device', 'time_delta_minutes']
    X = df_features[feature_cols]
    
    # Initialize Isolation Forest (Unsupervised Anomaly Detection)
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    
    # Fit model and predict (-1 for anomaly, 1 for normal)
    df_features['anomaly_score'] = model.fit_predict(X)
    
    # Convert prediction to a cleaner alert classification
    df_features['risk_status'] = df_features['anomaly_score'].apply(lambda x: 'CRITICAL ALERT' if x == -1 else 'CLEAR')
    
    return df_features

# ==========================================
# 4. EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    print("=== Finspark '26 Cyber-Transaction AI Prototype ===")
    
    # 1. Get raw data streams
    df_telemetry, df_transactions = generate_mock_data()
    print(f"[+] Ingested {len(df_telemetry)} security logs and {len(df_transactions)} transaction events.")
    
    # 2. Correlate data streams
    df_correlated = correlate_streams(df_telemetry, df_transactions)
    
    # 3. Run AI Engine
    df_results = train_and_detect_anomalies(df_correlated)
    
    # 4. Display Results
    alerts = df_results[df_results['risk_status'] == 'CRITICAL ALERT']
    
    print("\n=== SYSTEM ALERTS GENERATED ===")
    if len(alerts) > 0:
        for idx, row in alerts.iterrows():
            print(f"\n[!] ALERT: Suspicious Transaction Detected!")
            print(f"    User ID         : {row['user_id']}")
            print(f"    Transaction ID  : {row['tx_id']}")
            print(f"    Amount          : ₹{row['amount']:,}")
            print(f"    New Beneficiary : {'Yes' if row['is_new_beneficiary']==1 else 'No'}")
            print(f"    Login IP Risk   : {row['ip_risk_score']} (New Device: {'Yes' if row['is_new_device']==1 else 'No'})")
            print(f"    Time since login: {row['time_delta_minutes']:.1f} minutes")
    else:
        print("[+] No threats detected. System secure.")
