import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import math
import json

st.set_page_config(page_title="Fraud Data Generator", layout="wide")
st.title("Fraud Data Generator — Tạo dữ liệu mẫu cho các mô hình Fraud AI")

OUT_DIR = Path("./fraud_data_generator_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def generate_base_users(num_users, seed=42):
    rng = np.random.default_rng(seed)
    users = []
    for i in range(1, num_users+1):
        uid = f"U{i:06d}"
        avg_amount = float(10**rng.uniform(4,6))
        std_amount = max(1.0, avg_amount * rng.uniform(0.1, 0.6))
        account_age = int(max(1, rng.exponential(365)))
        home_lat = rng.uniform(8.0, 21.0)
        home_lon = rng.uniform(102.0, 110.0)
        users.append({
            "user_id": uid,
            "avg_amount": avg_amount,
            "std_amount": std_amount,
            "account_age_days": account_age,
            "home_lat": home_lat,
            "home_lon": home_lon
        })
    return pd.DataFrame(users)

def sample_timestamp(start_date, idx, rng):
    base = start_date + timedelta(seconds=int(idx*13 + rng.integers(0,10000)))
    return base

def generate_transactions(num_rows=20000, num_users=5000, anomaly_rate=0.05, seed=42):
    rng = np.random.default_rng(seed)
    users = generate_base_users(num_users, seed=seed)
    rows = []
    start_date = datetime.now() - timedelta(days=365)
    merchant_pool = ["ShopA","ShopB","NguyenStore","HospitalX","Utilities","VNPAY","MBBank","Agribank","Vietcombank"]
    channels = [1,2,3,4,5]

    for i in range(num_rows):
        uidx = rng.integers(0, num_users)
        urow = users.iloc[uidx]
        uid = urow["user_id"]
        avg = urow["avg_amount"]
        std = urow["std_amount"]
        is_anom = rng.random() < anomaly_rate

        if not is_anom:
            amount = int(max(1000, rng.normal(avg, std)))
        else:
            if rng.random() < 0.6:
                amount = int(max(1000, avg * rng.uniform(5,30)))
            else:
                amount = int(max(1000, avg * rng.uniform(0.01,0.1)))

        ts = sample_timestamp(start_date, i, rng)
        hour = ts.hour
        day_of_week = ts.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        velocity_1h = int(rng.poisson(0.5) if not is_anom else rng.poisson(3))
        velocity_24h = int(rng.poisson(2) if not is_anom else rng.poisson(8))
        avg_velocity = max(0.1, rng.normal(2,1))
        freq_norm = float(velocity_24h / avg_velocity)

        is_new_recipient = 0 if rng.random() > 0.15 else 1
        if is_anom and rng.random() < 0.7:
            is_new_recipient = 1

        recipient_count_30d = max(1, int(rng.poisson(3)))
        is_new_device = 0 if rng.random() > 0.12 else 1
        if is_anom and rng.random() < 0.6:
            is_new_device = 1

        device_count_30d = max(1, int(rng.poisson(1.5)))

        lat = urow["home_lat"] + rng.normal(0,0.02)
        lon = urow["home_lon"] + rng.normal(0,0.03)
        location_diff_km = haversine_km(urow["home_lat"], urow["home_lon"], lat, lon)
        if is_anom and rng.random() < 0.5:
            lat = urow["home_lat"] + rng.uniform(1,3)
            lon = urow["home_lon"] + rng.uniform(1,3)
            location_diff_km = haversine_km(urow["home_lat"], urow["home_lon"], lat, lon)

        channel = int(rng.choice(channels))
        account_age_days = int(max(1, urow["account_age_days"] + rng.integers(-10,50)))
        amount_norm = float((amount - avg) / (std if std>0 else 1.0))
        amount_log = float(np.log(amount + 1))
        amount_percentile_system = float(rng.random())
        global_anomaly_score_prev = float(max(0, rng.normal(0.2,0.1)))
        time_gap_prev_min = int(max(0, rng.exponential(60)))
        tx_id = f"TX{100000+i}"
        merchant = str(rng.choice(merchant_pool))
        receiving_bank = str(rng.choice(["Agribank","Vietcombank","BIDV","MBBank","VPBank","Sacombank"]))

        rows.append({
            "tx_id": tx_id,
            "user_id": uid,
            "amount": int(amount),
            "amount_log": amount_log,
            "amount_norm": amount_norm,
            "hour_of_day": int(hour),
            "day_of_week": int(day_of_week),
            "is_weekend": int(is_weekend),
            "time_gap_prev_min": int(time_gap_prev_min),
            "velocity_1h": int(velocity_1h),
            "velocity_24h": int(velocity_24h),
            "freq_norm": float(freq_norm),
            "is_new_recipient": int(is_new_recipient),
            "recipient_count_30d": int(recipient_count_30d),
            "is_new_device": int(is_new_device),
            "device_count_30d": int(device_count_30d),
            "location_diff_km": float(round(location_diff_km,3)),
            "channel": int(channel),
            "account_age_days": int(account_age_days),
            "amount_percentile_system": float(round(amount_percentile_system,3)),
            "global_anomaly_score_prev": float(round(global_anomaly_score_prev,3)),
            "merchant": merchant,
            "receiving_bank": receiving_bank,
            "is_anomaly": int(is_anom),
            "timestamp": ts.isoformat()
        })
    df = pd.DataFrame(rows)
    return df

def create_lightgbm(df):
    df2 = df.copy()
    df2["is_fraud"] = df2["is_anomaly"].apply(lambda x: x if np.random.rand() < 0.95 else int(1-x))
    return df2

def create_autoencoder(df):
    cols = ["amount_log","amount_norm","hour_of_day","is_weekend","time_gap_prev_min",
            "velocity_1h","velocity_24h","freq_norm","is_new_recipient","is_new_device",
            "location_diff_km","account_age_days","amount_percentile_system","global_anomaly_score_prev"]
    return df[cols + ["is_anomaly"]].copy()

def create_lstm_sequences(df, max_seq_len=8, max_sequences=3000):
    seq_rows = []
    grouped = df.sort_values(["user_id","timestamp"]).groupby("user_id")
    seq_id = 0
    for uid, g in grouped:
        g_sorted = g.sort_values("timestamp")
        vals = g_sorted.to_dict("records")
        for start in range(0, max(1, len(vals))):
            seq = vals[start:start+max_seq_len]
            if len(seq) < 2:
                continue
            seq_id += 1
            for idx, row in enumerate(seq):
                seq_rows.append({
                    "seq_id": f"S{seq_id:06d}",
                    "seq_index": int(idx),
                    "user_id": row["user_id"],
                    "tx_id": row["tx_id"],
                    "amount_log": row["amount_log"],
                    "amount_norm": row["amount_norm"],
                    "velocity_1h": row["velocity_1h"],
                    "velocity_24h": row["velocity_24h"],
                    "is_new_recipient": row["is_new_recipient"],
                    "is_new_device": row["is_new_device"],
                    "location_diff_km": row["location_diff_km"],
                    "hour_of_day": row["hour_of_day"],
                    "is_anomaly": row["is_anomaly"]
                })
            if seq_id >= max_sequences:
                break
        if seq_id >= max_sequences:
            break
    return pd.DataFrame(seq_rows)

def create_gnn_nodes_edges(df):
    user_groups = df.groupby("user_id").agg({
        "amount": ["mean","std","count"],
        "amount_norm": "mean",
        "velocity_24h": "mean",
        "account_age_days": "mean"
    }).fillna(0)
    user_groups.columns = ["_".join(c).strip() for c in user_groups.columns.values]
    user_nodes = user_groups.reset_index().rename(columns={
        "amount_mean":"avg_amount",
        "amount_std":"std_amount",
        "amount_count":"tx_count",
        "amount_norm_mean":"avg_amount_norm",
        "velocity_24h_mean":"avg_velocity_24h",
        "account_age_days_mean":"account_age_days"
    })
    receiving_map = {name:i for i,name in enumerate(df["receiving_bank"].unique(), start=1)}
    edges = df[["tx_id","user_id","receiving_bank","amount","is_anomaly"]].copy()
    edges["dst"] = edges["receiving_bank"].map(receiving_map)
    edges = edges.rename(columns={"user_id":"src","amount":"amt","is_anomaly":"label"})
    edges = edges[["tx_id","src","dst","amt","label"]]
    return user_nodes, edges

# ---------------------------
# UI
# ---------------------------
st.sidebar.header("Thông số sinh dữ liệu")
num_rows = st.sidebar.number_input("Số dòng giao dịch", min_value=2000, max_value=200000, value=20000, step=1000)
num_users = st.sidebar.number_input("Số user giả lập", min_value=100, max_value=20000, value=5000, step=100)
anomaly_rate = st.sidebar.slider("Tỷ lệ bất thường", 0.0, 0.2, 0.05, 0.01)
seed = st.sidebar.number_input("Seed ngẫu nhiên", value=42)

if st.button("Tạo dữ liệu mẫu"):
    st.info("Đang tạo dữ liệu, vui lòng chờ...")

    df = generate_transactions(int(num_rows), int(num_users), float(anomaly_rate), int(seed))

    # Isolation CSV
    isolation_cols = [
        "tx_id","user_id","amount","amount_log","amount_norm","hour_of_day","day_of_week","is_weekend",
        "time_gap_prev_min","velocity_1h","velocity_24h","freq_norm","is_new_recipient","recipient_count_30d",
        "is_new_device","device_count_30d","location_diff_km","channel","account_age_days",
        "amount_percentile_system","global_anomaly_score_prev"
    ]
    iso_df = df[isolation_cols + ["is_anomaly","timestamp","merchant","receiving_bank"]].copy()
    iso_path = OUT_DIR / f"isolation_21cols_{num_rows}rows.csv"
    iso_df.to_csv(iso_path, index=False)

    # LightGBM CSV
    lgb = create_lightgbm(iso_df)
    lgb_path = OUT_DIR / f"lightgbm_train_{num_rows}rows.csv"
    lgb.to_csv(lgb_path, index=False)

    # Autoencoder CSV
    ae = create_autoencoder(iso_df)
    ae_path = OUT_DIR / f"autoencoder_numeric_{num_rows}rows.csv"
    ae.to_csv(ae_path, index=False)

    # LSTM
    lstm = create_lstm_sequences(iso_df)
    lstm_path = OUT_DIR / f"lstm_sequences_{len(lstm)}rows.csv"
    lstm.to_csv(lstm_path, index=False)

    # GNN files
    nodes, edges = create_gnn_nodes_edges(iso_df)
    nodes_path = OUT_DIR / "gnn_nodes_users.csv"
    edges_path = OUT_DIR / "gnn_edges_transactions.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    st.success("Đã tạo xong dữ liệu mẫu!")

    st.subheader("📥 Tải xuống dữ liệu đã tạo")

    files = {
        "Isolation Forest CSV": iso_path,
        "LightGBM CSV": lgb_path,
        "Autoencoder CSV": ae_path,
        "LSTM Sequences CSV": lstm_path,
        "GNN Nodes CSV": nodes_path,
        "GNN Edges CSV": edges_path
    }

    for label, path in files.items():
        with open(path, "rb") as f:
            st.download_button(
                label=f"Tải {label}",
                data=f,
                file_name=path.name,
                mime="text/csv"
            )

st.markdown("""
## 📘 Ghi chú
- Isolation dùng 21 cột chuẩn + tỷ lệ anomaly bạn chọn.
- LightGBM là mô hình có nhãn.
- Autoencoder dùng cột numeric thuần.
- LSTM sử dụng chuỗi giao dịch theo user.
- GNN gồm nodes (user features) và edges (giao dịch).
""")
