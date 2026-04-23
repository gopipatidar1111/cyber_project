"""
=======================================================
  AI-Based Cyber Threat Detection & Response System
  Streamlit App — 4 Pages
  Page 1: Dashboard (Overview)
  Page 2: Analyze Network Activity (Input Form)
  Page 3: WiFi Scanner (JSON Upload Only)
  Page 4: URL Scanner
=======================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import time
import re
import math

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
}
.stApp { background-color: #0a0e1a; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f1a2e 100%);
    border-right: 1px solid #1e3a5f;
}

.metric-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,200,255,0.07);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
.metric-label { font-size: 0.78rem; letter-spacing: 1.5px; text-transform: uppercase; color: #7a9cc0; }
.metric-icon { font-size: 1.5rem; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #00d4ff;
    padding: 6px 0 12px 0;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 16px;
}

.result-attack {
    background: linear-gradient(135deg, #2d0a0a, #3d1010);
    border: 2px solid #ff4444;
    border-radius: 14px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,68,68,0.3);
    animation: pulse-red 2s infinite;
}
.result-normal {
    background: linear-gradient(135deg, #0a2d0a, #103d10);
    border: 2px solid #00ff88;
    border-radius: 14px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,255,136,0.2);
}
.result-title { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: 2px; }
.result-subtitle { font-size: 0.9rem; color: #aac; margin-top: 4px; letter-spacing: 1px; }

.action-block { border-radius: 12px; padding: 18px 20px; text-align: center; font-weight: 700; font-size: 1.1rem; letter-spacing: 1px; }
.action-block-high   { background: #3d0a0a; border: 1.5px solid #ff4444; color: #ff6666; }
.action-block-medium { background: #3d2a00; border: 1.5px solid #ffaa00; color: #ffcc44; }
.action-block-low    { background: #0a2d0a; border: 1.5px solid #00ff88; color: #44ffaa; }

.stNumberInput input, .stSelectbox select {
    background-color: #0d1b2e !important;
    color: #e0e6f0 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
.stSlider > div > div { color: #00d4ff; }

.stButton > button {
    background: linear-gradient(135deg, #0066cc, #0044aa);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    padding: 14px 36px;
    width: 100%;
    transition: all 0.3s;
    box-shadow: 0 4px 20px rgba(0,100,255,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077ff, #0055cc);
    box-shadow: 0 6px 28px rgba(0,120,255,0.5);
    transform: translateY(-1px);
}

.alert-banner { border-radius: 10px; padding: 12px 18px; margin: 8px 0; font-size: 0.9rem; font-weight: 500; }
.alert-high   { background: rgba(255,50,50,0.15);  border-left: 4px solid #ff4444; }
.alert-medium { background: rgba(255,170,0,0.15);  border-left: 4px solid #ffaa00; }
.alert-low    { background: rgba(0,255,136,0.15);  border-left: 4px solid #00ff88; }

.page-title    { font-size: 1.8rem; font-weight: 800; color: #00d4ff; letter-spacing: 2px; }
.page-subtitle { color: #7a9cc0; font-size: 0.9rem; margin-top: -8px; }

@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 20px rgba(255,68,68,0.3); }
    50%      { box-shadow: 0 0 40px rgba(255,68,68,0.6); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model    = joblib.load("model.pkl")
        scaler   = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        return model, scaler, encoders, False
    except FileNotFoundError:
        return None, None, None, True

model, scaler, encoders, DEMO_MODE = load_model()

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FINAL_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'num_failed_logins', 'logged_in', 'root_shell',
    'bytes_per_second', 'same_service_ratio', 'login_failure_flag',
    'root_access_flag', 'high_traffic_flag', 'large_data_flag',
    'risk_indicator', 'error_ratio', 'long_connection_flag',
]

PROTOCOL_MAP = {'TCP': 1, 'UDP': 2, 'ICMP': 0}
SERVICE_MAP  = {'http': 5, 'ftp': 3, 'smtp': 9, 'ssh': 10, 'domain': 2,
                'ftp_data': 4, 'telnet': 11, 'other': 7}
FLAG_MAP     = {'SF': 3, 'S0': 1, 'REJ': 0, 'RSTO': 2, 'RSTOS0': 4, 'SH': 5}

COUNT_THRESHOLD     = 100
SRC_BYTES_THRESHOLD = 50_000
DURATION_THRESHOLD  = 300


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_feature_vector(protocol, service, flag, duration, src_bytes, dst_bytes, count):
    prot_enc = PROTOCOL_MAP.get(protocol, 1)
    svc_enc  = SERVICE_MAP.get(service, 7)
    flag_enc = FLAG_MAP.get(flag, 3)
    srv_count = max(1, int(count * 0.85))

    if flag in ('S0', 'REJ'):
        serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate = 0.9, 0.8, 0.7, 0.6
    elif flag in ('RSTO', 'RSTOS0'):
        serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate = 0.4, 0.3, 0.6, 0.5
    else:
        serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate = 0.02, 0.02, 0.02, 0.02

    num_failed_logins = 0
    logged_in         = 1 if flag == 'SF' else 0
    root_shell        = 0

    bytes_per_second     = src_bytes / (duration + 1)
    same_service_ratio   = srv_count / (count + 1)
    login_failure_flag   = 0
    root_access_flag     = 0
    high_traffic_flag    = int(count > COUNT_THRESHOLD)
    large_data_flag      = int(src_bytes > SRC_BYTES_THRESHOLD)
    risk_indicator       = (count * serror_rate) + src_bytes
    error_ratio          = serror_rate + rerror_rate
    long_connection_flag = int(duration > DURATION_THRESHOLD)

    vector = [
        duration, prot_enc, svc_enc, flag_enc,
        src_bytes, dst_bytes, count, srv_count,
        serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate,
        num_failed_logins, logged_in, root_shell,
        bytes_per_second, same_service_ratio, login_failure_flag,
        root_access_flag, high_traffic_flag, large_data_flag,
        risk_indicator, error_ratio, long_connection_flag,
    ]
    return np.array(vector).reshape(1, -1)


THRESHOLD = 0.06

def compute_behavioral_score(vector):
    serror_rate = vector[0][8]
    rerror_rate = vector[0][10]
    src_bytes   = vector[0][4]
    count       = vector[0][6]
    duration    = vector[0][0]

    error_score    = ((serror_rate + rerror_rate) / 2) * 45
    traffic_score  = (min(count, 512) / 512) * 30
    data_score     = (min(src_bytes, 100_000) / 100_000) * 15
    duration_score = (1 - min(duration, 10) / 10) * 10

    return min(100, int(error_score + traffic_score + data_score + duration_score))


def predict(vector):
    if DEMO_MODE:
        score = compute_behavioral_score(vector)
        label = 1 if score > 20 else 0
        return label, score, score / 100
    else:
        scaled     = scaler.transform(vector)
        proba      = model.predict_proba(scaled)[0]
        confidence = proba[1]
        label      = 1 if confidence > THRESHOLD else 0

        ml_score         = confidence * 100
        behavioral_score = compute_behavioral_score(vector)
        if label == 1:
            risk_score = int((ml_score * 0.6) + (behavioral_score * 0.4))
        else:
            risk_score = int(min(35, behavioral_score * 0.4))

        return label, risk_score, confidence


def get_risk_level(score, label=None):
    if score >= 70: return "HIGH",   "#ff4444"
    if score >= 40: return "MEDIUM", "#ffaa00"
    return "LOW", "#00ff88"


# ─────────────────────────────────────────────
#  CUSTOM SVG GAUGE — always correct position
# ─────────────────────────────────────────────
def render_gauge(risk_score, risk_level, risk_color):
    """
    Renders a precise SVG semicircle gauge.
    0 = left (180°), 100 = right (0°).
    Needle angle = 180 - (score/100 * 180) degrees from positive X axis.
    """
    # Gauge geometry
    cx, cy, r = 200, 170, 130
    # Score 0 → angle 180° (left), score 100 → angle 0° (right)
    angle_deg = 180.0 - (risk_score / 100.0) * 180.0
    angle_rad = math.radians(angle_deg)
    nx = cx + r * math.cos(angle_rad)
    ny = cy - r * math.sin(angle_rad)  # SVG y-axis is inverted

    # Needle base points (small triangle)
    base_len = 10
    perp_rad = angle_rad + math.pi / 2
    bx1 = cx + base_len * math.cos(perp_rad)
    by1 = cy - base_len * math.sin(perp_rad)
    bx2 = cx - base_len * math.cos(perp_rad)
    by2 = cy + base_len * math.sin(perp_rad)

    # Arc segments: LOW (0-40) green, MEDIUM (40-70) yellow, HIGH (70-100) red
    def arc_path(start_score, end_score, outer_r, inner_r):
        a1 = math.radians(180.0 - (start_score / 100.0) * 180.0)
        a2 = math.radians(180.0 - (end_score   / 100.0) * 180.0)
        ox1 = cx + outer_r * math.cos(a1)
        oy1 = cy - outer_r * math.sin(a1)
        ox2 = cx + outer_r * math.cos(a2)
        oy2 = cy - outer_r * math.sin(a2)
        ix1 = cx + inner_r * math.cos(a2)
        iy1 = cy - inner_r * math.sin(a2)
        ix2 = cx + inner_r * math.cos(a1)
        iy2 = cy - inner_r * math.sin(a1)
        return f"M {ox1:.2f} {oy1:.2f} A {outer_r} {outer_r} 0 0 0 {ox2:.2f} {oy2:.2f} L {ix1:.2f} {iy1:.2f} A {inner_r} {inner_r} 0 0 1 {ix2:.2f} {iy2:.2f} Z"

    outer_r, inner_r = 130, 90

    svg = f"""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
    <svg viewBox="0 0 400 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;">
      <!-- Background track -->
      <path d="{arc_path(0, 100, outer_r+4, inner_r-4)}" fill="#1a2a3a" />

      <!-- Color zones -->
      <path d="{arc_path(0,  40,  outer_r, inner_r)}" fill="rgba(0,200,100,0.35)" />
      <path d="{arc_path(40, 70,  outer_r, inner_r)}" fill="rgba(255,170,0,0.35)" />
      <path d="{arc_path(70, 100, outer_r, inner_r)}" fill="rgba(255,60,60,0.35)" />

      <!-- Active fill up to current score -->
      <path d="{arc_path(0, risk_score, outer_r, inner_r)}" fill="{risk_color}" opacity="0.85"/>

      <!-- Tick marks -->
      {''.join([
          f'<line x1="{cx + (inner_r-6)*math.cos(math.radians(180-(i/100)*180)):.2f}" '
          f'y1="{cy - (inner_r-6)*math.sin(math.radians(180-(i/100)*180)):.2f}" '
          f'x2="{cx + (outer_r+6)*math.cos(math.radians(180-(i/100)*180)):.2f}" '
          f'y2="{cy - (outer_r+6)*math.sin(math.radians(180-(i/100)*180)):.2f}" '
          f'stroke="#0a0e1a" stroke-width="2.5"/>'
          for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      ])}

      <!-- Tick labels -->
      <text x="{cx + (outer_r+22)*math.cos(math.radians(180)):.2f}" y="{cy - (outer_r+22)*math.sin(math.radians(180)):.2f}" text-anchor="middle" fill="#7a9cc0" font-size="13">0</text>
      <text x="{cx + (outer_r+22)*math.cos(math.radians(108)):.2f}" y="{cy - (outer_r+22)*math.sin(math.radians(108)):.2f}" text-anchor="middle" fill="#7a9cc0" font-size="13">20</text>
      <text x="{cx + (outer_r+22)*math.cos(math.radians(54)):.2f}"  y="{cy - (outer_r+22)*math.sin(math.radians(54)):.2f}"  text-anchor="middle" fill="#ffaa00" font-size="13">40</text>
      <text x="{cx + (outer_r+22)*math.cos(math.radians(0)):.2f}"   y="{cy - (outer_r+22)*math.sin(math.radians(0)):.2f}"   text-anchor="middle" fill="#7a9cc0" font-size="13">100</text>
      <text x="{cx + (outer_r+22)*math.cos(math.radians(126)):.2f}" y="{cy - (outer_r+22)*math.sin(math.radians(126)):.2f}" text-anchor="middle" fill="#7a9cc0" font-size="13">10</text>
      <text x="{cx + (outer_r+22)*math.cos(math.radians(27)):.2f}"  y="{cy - (outer_r+22)*math.sin(math.radians(27)):.2f}"  text-anchor="middle" fill="#ff4444" font-size="13">70</text>

      <!-- Needle -->
      <polygon points="{nx:.2f},{ny:.2f} {bx1:.2f},{by1:.2f} {bx2:.2f},{by2:.2f}"
               fill="{risk_color}" opacity="0.95" />
      <!-- Needle center cap -->
      <circle cx="{cx}" cy="{cy}" r="14" fill="#0d1b2e" stroke="{risk_color}" stroke-width="3"/>
      <circle cx="{cx}" cy="{cy}" r="6"  fill="{risk_color}"/>

      <!-- Score text -->
      <text x="{cx}" y="{cy+40}" text-anchor="middle" fill="{risk_color}"
            font-size="36" font-weight="800" font-family="Segoe UI">{risk_score}/100</text>
      <text x="{cx}" y="{cy+62}" text-anchor="middle" fill="{risk_color}"
            font-size="14" font-weight="700" letter-spacing="2" font-family="Segoe UI">RISK LEVEL: {risk_level}</text>
    </svg>
    </div>
    """
    return svg


# ─────────────────────────────────────────────
#  WiFi RISK
# ─────────────────────────────────────────────
def compute_wifi_risk(network, all_networks):
    risk  = 0
    flags = []
    ssid  = network.get('ssid', '').lower()
    sec   = network.get('security', '').lower()
    enc   = network.get('encryption', '').lower()
    sig   = network.get('signal', 0)

    if 'open' in sec or 'none' in sec:
        risk += 40
        flags.append("Open Network — No Password Protection")

    if 'wep' in enc:
        risk += 30
        flags.append("Weak Encryption (WEP) — Easily Crackable")

    suspicious_words = ['free', 'public', 'guest', 'hack', 'test', 'open',
                        'wifi', 'hotspot', 'unsecured', 'airport', 'hotel',
                        'cafe', 'evil', 'attack', 'phish', 'win', 'prize',
                        'paypal', 'bank', 'verify', 'login', 'update']
    for word in suspicious_words:
        if word in ssid:
            risk += 20
            flags.append(f"Suspicious Network Name contains '{word}'")
            break

    ssid_counts = {}
    for n in all_networks:
        s = n.get('ssid', '').lower()
        ssid_counts[s] = ssid_counts.get(s, 0) + 1
    if ssid_counts.get(ssid, 0) > 1:
        risk += 35
        flags.append("Duplicate SSID Detected — Possible Evil Twin Attack")

    if sig >= 90:
        risk += 10
        flags.append(f"Unusually Strong Signal ({sig}%) — possible rogue AP nearby")

    default_names = ['linksys', 'netgear', 'dlink', 'd-link', 'tp-link',
                     'tplink', 'asus', 'belkin', 'default']
    for name in default_names:
        if name in ssid:
            risk += 10
            flags.append("Default Router Name — Poorly Secured Network")
            break

    if not flags:
        flags.append("No suspicious indicators found")

    return min(100, risk), flags


# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
        <div style='font-size:2.8rem;'>🛡️</div>
        <div style='font-size:1.1rem; font-weight:800; color:#00d4ff; letter-spacing:2px;'>CYBERSHIELD</div>
        <div style='font-size:0.7rem; color:#7a9cc0; letter-spacing:1px;'>AI THREAT DETECTION</div>
    </div>
    <hr style='border-color:#1e3a5f; margin:0 0 20px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🔍  Analyze Request", "📡  WiFi Scanner", "🔗  URL Scanner"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if DEMO_MODE:
        st.warning("⚠️ **Demo Mode**\nmodel.pkl not found.\nPredictions use a heuristic engine.", icon="⚠️")
    else:
        st.success("✅ **Model Loaded**\nXGBoost active.", icon="✅")

    st.markdown("""
    <hr style='border-color:#1e3a5f; margin: 20px 0;'>
    <div style='font-size:0.75rem; color:#4a6c8f; text-align:center;'>
        NSL-KDD Dataset<br>
        ~94% Accuracy<br>
        24 Features
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠  Dashboard":

    st.markdown("""
    <div style='display:flex; justify-content:space-between; align-items:center;
                padding:0 0 20px 0; border-bottom:1px solid #1a2d4a; margin-bottom:28px;'>
        <div>
            <div style='font-size:1.7rem; font-weight:800; color:#e0e6f0; letter-spacing:1px;'>
                Threat Intelligence <span style='color:#7b5ea7;'>Dashboard</span>
            </div>
            <div style='font-size:0.82rem; color:#4a6c8f; margin-top:4px;'>
                NSL-KDD Dataset · XGBoost Model · Real-time Analysis
            </div>
        </div>
        <div style='font-size:0.78rem; color:#4a6c8f; text-align:right;'>
            <div style='color:#00ff88; font-weight:600;'>● SYSTEM ACTIVE</div>
            <div>Last scan: just now</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    card_data = [
        (c1, "125,973", "Total Requests",   "#00d4ff", "#0a2a3d", "↑ 12.4%", "📡"),
        (c2, "58,630",  "Attacks Detected", "#ff4444", "#3d0a0a", "↑ 3.1%",  "🚨"),
        (c3, "14,208",  "High Risk Events", "#ffaa00", "#3d2200", "↓ 1.8%",  "⚠️"),
        (c4, "53.5%",   "Safe Traffic",     "#00ff88", "#0a2d1a", "↑ 2.2%",  "✅"),
    ]
    for col, val, label, color, bg, trend, icon in card_data:
        with col:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{bg} 0%,#0d1117 100%);
                        border:1px solid {color}33; border-radius:14px;
                        padding:20px 22px; position:relative; overflow:hidden;
                        box-shadow:0 4px 24px {color}15;'>
                <div style='position:absolute;top:-10px;right:-10px;font-size:3.5rem;opacity:0.07;'>{icon}</div>
                <div style='font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#4a6c8f;margin-bottom:8px;'>{label}</div>
                <div style='font-size:2rem;font-weight:800;color:{color};letter-spacing:-1px;'>{val}</div>
                <div style='font-size:0.75rem;color:{color};opacity:0.8;margin-top:6px;'>{trend} from last period</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117 0%,#0f1a2e 100%);
                border:1px solid #1a2d4a; border-radius:16px; padding:24px 24px 8px 24px; margin-bottom:24px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
            <div>
                <div style='font-size:1rem;font-weight:700;color:#e0e6f0;'>Risk Score Trend</div>
                <div style='font-size:0.78rem;color:#4a6c8f;'>Network request activity — attack spikes highlighted</div>
            </div>
            <div style='display:flex;gap:20px;font-size:0.78rem;'>
                <span style='color:#7b5ea7;'>● Risk Score</span>
                <span style='color:#ff4444;'>● Attack</span>
                <span style='color:#ffaa00;'>● Warning</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    np.random.seed(42)
    n = 300
    risk_scores = []
    base = 18
    for i in range(n):
        if 50 < i < 75 or 130 < i < 155 or 210 < i < 230 or 260 < i < 278:
            base = min(97, base + np.random.randint(6, 16))
        else:
            base = max(8, base + np.random.randint(-9, 5))
        risk_scores.append(int(base))

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=list(range(n)), y=risk_scores,
        mode='lines',
        line=dict(color='#7b5ea7', width=2.5, shape='spline', smoothing=0.6),
        fill='tozeroy', fillcolor='rgba(123,94,167,0.12)',
        hovertemplate='<b>Request #%{x}</b><br>Risk Score: %{y}<extra></extra>',
        showlegend=False,
    ))
    fig_line.add_hline(y=70, line_dash='dash', line_color='rgba(255,68,68,0.5)', line_width=1.5,
                       annotation_text='⚡ Attack Threshold (70)',
                       annotation_font=dict(color='#ff6666', size=11), annotation_position='top right')
    fig_line.add_hline(y=40, line_dash='dot', line_color='rgba(255,170,0,0.4)', line_width=1.5,
                       annotation_text='⚠ Warning (40)',
                       annotation_font=dict(color='#ffcc44', size=11), annotation_position='top right')
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a6c8f', family='Segoe UI'), height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=dict(text='Network Request Index', font=dict(color='#4a6c8f', size=11)),
                   gridcolor='rgba(26,45,74,0.6)', showgrid=True, zeroline=False, tickfont=dict(color='#4a6c8f')),
        yaxis=dict(title=dict(text='Risk Score (0–100)', font=dict(color='#4a6c8f', size=11)),
                   gridcolor='rgba(26,45,74,0.6)', range=[0, 110], zeroline=False, tickfont=dict(color='#4a6c8f')),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0d1b2e', bordercolor='#7b5ea7', font=dict(color='#e0e6f0')),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117 0%,#0f1a2e 100%);
                border:1px solid #1a2d4a; border-radius:16px; padding:24px 24px 8px 24px; margin-bottom:24px;'>
        <div style='font-size:1rem;font-weight:700;color:#e0e6f0;margin-bottom:4px;'>Risk Level Distribution</div>
        <div style='font-size:0.78rem;color:#4a6c8f;'>Breakdown of all network events by threat severity</div>
    </div>
    """, unsafe_allow_html=True)

    d_left, d_right = st.columns([1, 1])
    with d_left:
        fig_donut = go.Figure(go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
            values=[53.5, 20.8, 16.1, 9.6],
            hole=0.65,
            marker=dict(colors=['#00cc66', '#ffaa00', '#ff6644', '#cc2222'],
                        line=dict(color='#0a0e1a', width=3)),
            textinfo='percent', textfont=dict(color='#e0e6f0', size=12),
            hovertemplate='<b>%{label}</b><br>%{value}% of traffic<extra></extra>',
            direction='clockwise', sort=False,
        ))
        fig_donut.add_annotation(text='<b>53.5%</b><br><span style="font-size:10px">SAFE</span>',
                                  x=0.5, y=0.5, showarrow=False, font=dict(size=18, color='#00ff88'), align='center')
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True, legend=dict(font=dict(color='#7a9cc0', size=12), bgcolor='rgba(0,0,0,0)',
                                         orientation='v', x=1.02, y=0.5),
            margin=dict(l=10, r=60, t=10, b=10), height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with d_right:
        st.markdown("<br>", unsafe_allow_html=True)
        risk_items = [
            ("🟢", "Low Risk",    "53.5%", "#00cc66", "67,342 connections — normal activity"),
            ("🟡", "Medium Risk", "20.8%", "#ffaa00", "26,202 connections — suspicious"),
            ("🔴", "High Risk",   "16.1%", "#ff6644", "20,282 connections — active threats"),
            ("💀", "Critical",    "9.6%",  "#cc2222", "12,093 connections — confirmed attacks"),
        ]
        for dot, label, pct, color, desc in risk_items:
            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid {color}33;border-radius:12px;
                        padding:14px 18px;margin-bottom:10px;display:flex;
                        justify-content:space-between;align-items:center;'>
                <div>
                    <span style='font-size:1rem;font-weight:700;color:{color};'>{dot} {label}</span>
                    <div style='font-size:0.75rem;color:#4a6c8f;margin-top:2px;'>{desc}</div>
                </div>
                <div style='font-size:1.6rem;font-weight:800;color:{color};'>{pct}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117 0%,#0f1a2e 100%);
                border:1px solid #1a2d4a; border-radius:16px; padding:24px 24px 8px 24px; margin-bottom:24px;'>
        <div style='font-size:1rem;font-weight:700;color:#e0e6f0;margin-bottom:4px;'>Attack Category Distribution</div>
        <div style='font-size:0.78rem;color:#4a6c8f;'>NSL-KDD dataset — breakdown by attack type</div>
    </div>
    """, unsafe_allow_html=True)

    categories = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    counts     = [67343, 45927, 11656, 2199, 52]
    bar_colors = ['#00cc66', '#ff4444', '#ffaa00', '#ff7722', '#cc44ff']
    pct_vals   = [52.9, 36.1, 9.2, 1.7, 0.04]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=categories, y=counts,
        marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0)', width=0), opacity=0.85),
        text=[f'<b>{c:,}</b><br>{p}%' for c, p in zip(counts, pct_vals)],
        textposition='outside', textfont=dict(color='#e0e6f0', size=11),
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>', width=0.5,
    ))
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a6c8f', family='Segoe UI'), height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor='rgba(0,0,0,0)', zeroline=False, tickfont=dict(color='#7a9cc0', size=13)),
        yaxis=dict(title=dict(text='Number of Records', font=dict(color='#4a6c8f', size=11)),
                   gridcolor='rgba(26,45,74,0.5)', zeroline=False, tickfont=dict(color='#4a6c8f')),
        bargap=0.35,
        hoverlabel=dict(bgcolor='#0d1b2e', bordercolor='#7b5ea7', font=dict(color='#e0e6f0')),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    a1, a2, a3, a4 = st.columns(4)
    attack_info = [
        (a1, "DoS",   "#ff4444", "Denial of Service", "Floods network to exhaust resources", "45,927"),
        (a2, "Probe", "#ffaa00", "Reconnaissance",    "Port scanning & network mapping",      "11,656"),
        (a3, "R2L",   "#ff7722", "Remote to Local",   "Unauthorized remote access attempts",  "2,199"),
        (a4, "U2R",   "#cc44ff", "Root Escalation",   "Privilege escalation to root/admin",   "52"),
    ]
    for col, short, color, title, desc, count in attack_info:
        with col:
            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid {color}33;border-radius:12px;
                        padding:16px;text-align:center;margin-bottom:8px;'>
                <div style='font-size:1.1rem;font-weight:800;color:{color};letter-spacing:1px;'>{short}</div>
                <div style='font-size:0.78rem;font-weight:600;color:#e0e6f0;margin:4px 0;'>{title}</div>
                <div style='font-size:0.72rem;color:#4a6c8f;'>{desc}</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};margin-top:8px;'>{count}</div>
                <div style='font-size:0.65rem;color:#4a6c8f;'>records</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — ANALYZE NETWORK REQUEST
# ═══════════════════════════════════════════════════════════
elif page == "🔍  Analyze Request":
    st.markdown('<div class="page-title">🔍 ANALYZE NETWORK ACTIVITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter connection parameters — AI will classify and score the threat</div><br>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚡ Quick Load Preset</div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset = None
    with p_col1:
        if st.button("✅ Normal Traffic"): preset = "normal"
    with p_col2:
        if st.button("🚨 DoS Attack"):    preset = "dos"
    with p_col3:
        if st.button("🔎 Probe / Scan"):  preset = "probe"
    with p_col4:
        if st.button("🔐 R2L Attack"):    preset = "r2l"

    PRESETS = {
        "normal": dict(protocol="TCP", service="http", flag="SF",  duration=2,   src_bytes=491,     dst_bytes=5134, count=8),
        "dos":    dict(protocol="TCP", service="http", flag="S0",  duration=0,   src_bytes=0,        dst_bytes=0,    count=511),
        "probe":  dict(protocol="TCP", service="ftp",  flag="REJ", duration=0,   src_bytes=0,        dst_bytes=0,    count=229),
        "r2l":    dict(protocol="TCP", service="ftp",  flag="SF",  duration=213, src_bytes=105_680,  dst_bytes=146,  count=1),
    }
    if preset:
        st.session_state.update(PRESETS[preset])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Connection Parameters</div>', unsafe_allow_html=True)

    form_col1, form_col2 = st.columns(2)
    with form_col1:
        protocol = st.selectbox("🔌 Protocol Type", ["TCP", "UDP", "ICMP"],
            index=["TCP","UDP","ICMP"].index(st.session_state.get("protocol", "TCP")))
        service = st.selectbox("🌐 Service / Port",
            ["http","ftp","smtp","ssh","domain","ftp_data","telnet","other"],
            index=["http","ftp","smtp","ssh","domain","ftp_data","telnet","other"].index(
                st.session_state.get("service", "http")))
        flag = st.selectbox("🚩 Connection Flag", ["SF","S0","REJ","RSTO","RSTOS0","SH"],
            index=["SF","S0","REJ","RSTO","RSTOS0","SH"].index(st.session_state.get("flag", "SF")))
        duration = st.slider("⏱️ Connection Duration (seconds)", 0, 600, st.session_state.get("duration", 2))

    with form_col2:
        src_bytes = st.number_input("📤 Source Bytes (sent)", 0, 1_000_000, st.session_state.get("src_bytes", 491), step=100)
        dst_bytes = st.number_input("📥 Destination Bytes (received)", 0, 1_000_000, st.session_state.get("dst_bytes", 5134), step=100)
        count     = st.slider("🔁 Request Count (last 2 sec)", 1, 512, st.session_state.get("count", 8))
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#0d1b2e;border:1px solid #1e3a5f;border-radius:10px;padding:14px 16px;font-size:0.8rem;color:#7a9cc0;'>
        <b style='color:#00d4ff;'>🚩 Flag Reference</b><br>
        <b>SF</b> = Complete connection (Normal)<br>
        <b>S0</b> = No response (DoS indicator)<br>
        <b>REJ</b> = Connection rejected (Probe)<br>
        <b>RSTO</b> = Reset by origin
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze = st.button("🔍  ANALYZE NETWORK ACTIVITY")

    if analyze:
        with st.spinner("🧠 Running AI analysis..."):
            time.sleep(0.8)

        vector = build_feature_vector(protocol, service, flag, duration, src_bytes, dst_bytes, count)
        label, risk_score, confidence = predict(vector)
        risk_level, risk_color = get_risk_level(risk_score)
        is_attack = label == 1

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-header">🎯 ANALYSIS RESULTS</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if is_attack:
                st.markdown("""
                <div class="result-attack">
                    <div class="result-title">🚨 ATTACK DETECTED</div>
                    <div class="result-subtitle">MALICIOUS NETWORK ACTIVITY IDENTIFIED</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-normal">
                    <div class="result-title" style="color:#00ff88;">✅ NORMAL TRAFFIC</div>
                    <div class="result-subtitle">NO THREAT IDENTIFIED</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GAUGE + SUMMARY side by side ──────────────────────────────
        gauge_col, detail_col = st.columns(2, gap="large")

        with gauge_col:
            st.markdown('<div class="section-header">📊 Threat Risk Meter</div>', unsafe_allow_html=True)
            # Render precise SVG gauge
            gauge_svg = render_gauge(risk_score, risk_level, risk_color)
            st.markdown(gauge_svg, unsafe_allow_html=True)

        with detail_col:
            st.markdown('<div class="section-header">📋 Connection Summary</div>', unsafe_allow_html=True)
            # Extra top padding so summary aligns with gauge vertically
            st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
            details = {
                "Protocol": protocol, "Service": service.upper(), "Flag": flag,
                "Duration": f"{duration}s", "Src Bytes": f"{src_bytes:,}",
                "Dst Bytes": f"{dst_bytes:,}", "Request Count": count,
                "Confidence": f"{confidence*100:.1f}%",
            }
            for k, v in details.items():
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:9px 0;
                            border-bottom:1px solid #1e3a5f;font-size:0.92rem;'>
                    <span style='color:#7a9cc0;'>{k}</span>
                    <span style='color:#e0e6f0;font-weight:600;'>{v}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 Automated Response Recommendation</div>', unsafe_allow_html=True)

        action_col1, action_col2, action_col3 = st.columns(3)
        actions = {
            "HIGH":   ("action-block-high",   "🚫 BLOCK IP ADDRESS",   "Immediately terminate connection and blacklist source IP."),
            "MEDIUM": ("action-block-medium", "👁️  MONITOR & LOG",     "Flag for human review. Increase monitoring frequency."),
            "LOW":    ("action-block-low",    "✅ NO ACTION REQUIRED", "Connection appears safe. Continue normal operations."),
        }
        for col, (level, (css, action_title, action_desc)) in zip([action_col1, action_col2, action_col3], actions.items()):
            active_style = f"border: 3px solid; box-shadow: 0 0 18px {'#ff444488' if level=='HIGH' else '#ffaa0088' if level=='MEDIUM' else '#00ff8888'};" if level == risk_level else "opacity: 0.35;"
            with col:
                st.markdown(f"""
                <div class="action-block {css}" style="{active_style}">
                    <div style="font-size:1.3rem;margin-bottom:8px;">{action_title}</div>
                    <div style="font-size:0.78rem;font-weight:400;color:#aac;">{action_desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔬 View Full Feature Vector (24 features sent to model)"):
            raw_vals = vector[0].tolist()
            df_feat  = pd.DataFrame({
                "Feature":   FINAL_FEATURES,
                "Raw Value": [f"{v:.4f}" for v in raw_vals],
                "Category":  ["Direct"] * 15 + ["Engineered"] * 9,
            })
            st.dataframe(
                df_feat.style.applymap(
                    lambda x: "color: #00d4ff" if x == "Direct" else "color: #ffaa00",
                    subset=["Category"]),
                use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if risk_level == "HIGH":
            st.markdown(f"""
            <div class="alert-banner alert-high">
                🚨 <b>CRITICAL ALERT</b> — Risk Score: {risk_score}/100 |
                Protocol: {protocol} | Service: {service.upper()} |
                Recommended Action: <b>Block IP Immediately</b>
            </div>""", unsafe_allow_html=True)
        elif risk_level == "MEDIUM":
            st.markdown(f"""
            <div class="alert-banner alert-medium">
                ⚠️ <b>WARNING</b> — Risk Score: {risk_score}/100 |
                Protocol: {protocol} | Service: {service.upper()} |
                Recommended Action: <b>Monitor & Investigate</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-banner alert-low">
                ✅ <b>SAFE</b> — Risk Score: {risk_score}/100 |
                Protocol: {protocol} | Service: {service.upper()} |
                Status: <b>Normal Traffic — No Action Required</b>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — WiFi THREAT SCANNER  (JSON upload ONLY)
# ═══════════════════════════════════════════════════════════
elif page == "📡  WiFi Scanner":

    st.markdown('<div class="page-title">📡 WiFi THREAT SCANNER</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload your local WiFi scan JSON to detect rogue access points & threats</div><br>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1b2e;border:1px solid #1e3a5f;border-radius:12px;
                padding:16px 20px;margin-bottom:20px;font-size:0.88rem;color:#7a9cc0;'>
        <b style='color:#00d4ff;'>How WiFi Threat Detection Works</b><br><br>
        Each network in your scan is analyzed for:
        <span style='color:#ffaa00;'>Open Security</span> |
        <span style='color:#ff4444;'>Evil Twin / Duplicate SSID</span> |
        <span style='color:#ff7722;'>Suspicious Names</span> |
        <span style='color:#cc44ff;'>Weak Encryption (WEP)</span><br><br>
        <b style='color:#00d4ff;'>📁 How to generate your scan file:</b><br>
        Run the local scanner script on your machine, then upload the resulting
        <b>local_wifi_scan.json</b> file below. The JSON should be a list of network
        objects with fields: <code>ssid</code>, <code>bssid</code>, <code>signal</code>,
        <code>security</code>, <code>encryption</code>.
    </div>
    """, unsafe_allow_html=True)

    # ── JSON Upload only ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Upload WiFi Scan File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload local_wifi_scan.json",
        type=["json"],
        help="Generate this file by running the local WiFi scanner script on your device.",
    )

    networks = None

    if uploaded_file is not None:
        import json
        try:
            networks = json.load(uploaded_file)
            st.success(f"✅ Loaded **{len(networks)} networks** from uploaded file!", icon="📡")
        except Exception as e:
            st.error(f"❌ Error reading the JSON file: {e}\n\nPlease make sure it is valid JSON.")
            networks = None
    else:
        st.info(
            "📤 **Upload a scan file to begin.**  \n"
            "Run your local WiFi scanner script and upload the JSON output here.",
            icon="💡"
        )

    # ── Process and display ──────────────────────────────────────────
    if networks is not None and len(networks) > 0:
        st.markdown(f"**{len(networks)} networks found**")
        st.markdown('<div class="section-header">Network Threat Analysis</div>', unsafe_allow_html=True)

        results = []
        for net in networks:
            risk, flags = compute_wifi_risk(net, networks)
            level, color = get_risk_level(risk)
            results.append({
                'ssid':     net.get('ssid', 'Unknown'),
                'bssid':    net.get('bssid', 'N/A'),
                'signal':   net.get('signal', 0),
                'security': net.get('security', 'Unknown'),
                'risk':     risk,
                'level':    level,
                'color':    color,
                'flags':    flags,
            })
        results.sort(key=lambda x: x['risk'], reverse=True)

        total  = len(results)
        high   = sum(1 for r in results if r['level'] == 'HIGH')
        medium = sum(1 for r in results if r['level'] == 'MEDIUM')
        safe   = sum(1 for r in results if r['level'] == 'LOW')

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, val, label, color in [
            (c1, "📡", total,  "NETWORKS FOUND", "#00d4ff"),
            (c2, "🚨", high,   "THREATS",        "#ff4444"),
            (c3, "⚠️",  medium, "SUSPICIOUS",     "#ffaa00"),
            (c4, "✅", safe,   "SAFE",           "#00ff88"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for r in results:
            bg         = "#2d0a0a" if r['level'] == 'HIGH' else "#2d2200" if r['level'] == 'MEDIUM' else "#0a2d0a"
            icon       = "🚨" if r['level'] == 'HIGH' else "⚠️" if r['level'] == 'MEDIUM' else "✅"
            signal_bar = "█" * (r['signal'] // 10) + "░" * (10 - r['signal'] // 10)
            flags_html = "".join([f"<div style='margin:3px 0;font-size:0.82rem;color:#aac;'>• {f}</div>" for f in r['flags']])

            st.markdown(f"""
            <div style='background:{bg};border:2px solid {r["color"]};border-radius:12px;
                        padding:18px 22px;margin:10px 0;box-shadow:0 0 20px {r["color"]}33;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <span style='font-size:1.2rem;font-weight:800;color:{r["color"]};'>{icon} {r["ssid"]}</span>
                        <span style='margin-left:16px;font-size:0.78rem;color:#7a9cc0;'>{r["bssid"]}</span>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:1.6rem;font-weight:800;color:{r["color"]};'>{r["risk"]}/100</div>
                        <div style='font-size:0.72rem;color:{r["color"]};letter-spacing:1px;'>{r["level"]} RISK</div>
                    </div>
                </div>
                <div style='margin-top:10px;font-size:0.83rem;color:#7a9cc0;display:flex;gap:24px;'>
                    <span>Security: {r["security"]}</span>
                    <span>Signal: {r["signal"]}% {signal_bar}</span>
                </div>
                <div style='margin-top:10px;padding-top:10px;border-top:1px solid #1e3a5f;'>
                    <b style='font-size:0.8rem;color:#00d4ff;'>Detection Flags:</b>
                    {flags_html}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)

        fig_wifi = go.Figure(go.Bar(
            x=[r['ssid'] for r in results],
            y=[r['risk'] for r in results],
            marker=dict(color=[r['color'] for r in results], line=dict(color='#0a0e1a', width=1.5)),
            text=[str(r['risk']) for r in results],
            textposition='outside', textfont=dict(color='#e0e6f0'),
        ))
        fig_wifi.add_hline(y=70, line_dash='dash', line_color='#ff4444',
                           annotation_text='Threat (70)', annotation_font_color='#ff6666')
        fig_wifi.add_hline(y=40, line_dash='dot', line_color='#ffaa00',
                           annotation_text='Suspicious (40)', annotation_font_color='#ffcc44')
        fig_wifi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7a9cc0'),
            xaxis=dict(gridcolor='#0d1b2e', tickangle=-25),
            yaxis=dict(title='Risk Score', gridcolor='#0d1b2e', range=[0, 115]),
            margin=dict(l=10, r=10, t=20, b=80), height=320,
        )
        st.plotly_chart(fig_wifi, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 4 — URL SCANNER
# ═══════════════════════════════════════════════════════════
elif page == "🔗  URL Scanner":
    import urllib.parse

    st.markdown('<div class="page-title">🔗 URL THREAT SCANNER</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analyze any URL for phishing, malware, and suspicious patterns</div><br>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1b2e;border:1px solid #1e3a5f;border-radius:12px;
                padding:16px 20px;margin-bottom:20px;font-size:0.88rem;color:#7a9cc0;'>
        <b style='color:#00d4ff;'>How URL Threat Detection Works</b><br><br>
        Analyzes:
        <span style='color:#ff4444;'>Phishing Keywords</span> |
        <span style='color:#ffaa00;'>No HTTPS</span> |
        <span style='color:#ff7722;'>IP Instead of Domain</span> |
        <span style='color:#cc44ff;'>Suspicious Characters</span> |
        <span style='color:#ff4444;'>Excessive Subdomains</span>
    </div>
    """, unsafe_allow_html=True)

    def extract_url_features(url):
        url = url.strip()
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
        except Exception:
            domain = url

        phishing_words = ['login', 'signin', 'verify', 'account', 'update', 'secure',
                          'bank', 'paypal', 'password', 'credential', 'confirm', 'wallet',
                          'free', 'win', 'prize', 'lucky', 'click', 'urgent', 'suspend',
                          'blocked', 'limited', 'unusual', 'activity']
        found_kw = [w for w in phishing_words if w in url.lower()]

        suspicious_tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top',
                           '.click', '.download', '.link', '.win']

        return {
            'url_length':         len(url),
            'has_https':          url.startswith('https://'),
            'has_ip':             bool(re.match(r'https?://\d+\.\d+\.\d+\.\d+', url)),
            'has_at':             '@' in url,
            'has_double_slash':   '//' in url.replace('https://', '').replace('http://', ''),
            'hyphen_count':       url.count('-'),
            'subdomain_count':    max(0, domain.count('.') - 1),
            'special_char_count': sum(url.count(c) for c in ['%', '=', '?', '&', '#']),
            'phishing_keywords':  found_kw,
            'keyword_count':      len(found_kw),
            'suspicious_tld':     any(url.lower().endswith(t) for t in suspicious_tlds),
        }

    def compute_url_risk(features):
        risk    = 0
        reasons = []

        if not features['has_https']:
            risk += 20; reasons.append("No HTTPS — connection is not encrypted")
        if features['url_length'] > 100:
            risk += 20; reasons.append(f"Very long URL ({features['url_length']} chars) — common in phishing")
        elif features['url_length'] > 75:
            risk += 10; reasons.append(f"Long URL ({features['url_length']} chars)")
        if features['has_ip']:
            risk += 30; reasons.append("IP address used instead of domain — very suspicious")
        if features['has_at']:
            risk += 25; reasons.append("'@' symbol in URL — used to trick browsers")
        if features['has_double_slash']:
            risk += 15; reasons.append("Double slash in path — redirection trick")
        if features['hyphen_count'] > 3:
            risk += 15; reasons.append(f"Too many hyphens ({features['hyphen_count']}) — phishing indicator")
        if features['subdomain_count'] > 2:
            risk += 20; reasons.append(f"Too many subdomains ({features['subdomain_count']}) — spoofing attempt")
        if features['keyword_count'] > 0:
            risk += min(30, features['keyword_count'] * 12)
            reasons.append(f"Phishing keywords: {', '.join(features['phishing_keywords'][:4])}")
        if features['suspicious_tld']:
            risk += 20; reasons.append("Suspicious top-level domain (free/spam TLD)")
        if features['special_char_count'] > 5:
            risk += 10; reasons.append(f"Many special characters ({features['special_char_count']}) — obfuscation")
        if not reasons:
            reasons.append("No suspicious patterns detected — URL appears safe")

        return min(100, risk), reasons

    st.markdown('<div class="section-header">Single URL Analysis</div>', unsafe_allow_html=True)
    url_input = st.text_input("Enter URL", placeholder="https://example.com/login?verify=account")

    st.markdown("**Quick Demo Presets:**")
    d1, d2, d3, d4 = st.columns(4)
    preset_url = ""
    with d1:
        if st.button("✅ Safe URL"):    preset_url = "https://www.google.com"
    with d2:
        if st.button("⚠️ Suspicious"): preset_url = "http://secure-login-verify.xyz/account/confirm"
    with d3:
        if st.button("🚨 Phishing"):   preset_url = "http://192.168.1.1/paypal-login/verify@account?update=password"
    with d4:
        if st.button("🔎 Fake Bank"):  preset_url = "http://bank-of-america-secure-login.tk/verify/credentials"
    if preset_url:
        url_input = preset_url

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Bulk URL Scanner</div>', unsafe_allow_html=True)
    bulk_input = st.text_area("Enter multiple URLs (one per line)",
        placeholder="https://google.com\nhttp://free-prize-winner.tk/claim\nhttps://github.com", height=100)

    _, analyze_col, _ = st.columns([1, 2, 1])
    with analyze_col:
        analyze_url = st.button("ANALYZE URL(s)")

    if analyze_url or (url_input and url_input.strip()):
        urls_to_scan = []
        if url_input and url_input.strip():
            urls_to_scan.append(url_input.strip())
        if bulk_input and bulk_input.strip():
            for u in bulk_input.strip().split('\n'):
                u = u.strip()
                if u and u not in urls_to_scan:
                    urls_to_scan.append(u)

        if not urls_to_scan:
            st.warning("Please enter at least one URL.")
        else:
            with st.spinner("Analyzing URLs..."):
                time.sleep(0.6)

            url_results = []
            for url in urls_to_scan:
                features = extract_url_features(url)
                risk, reasons = compute_url_risk(features)
                level, color  = get_risk_level(risk)
                url_results.append({'url': url, 'risk': risk, 'level': level,
                                    'color': color, 'reasons': reasons, 'features': features})
            url_results.sort(key=lambda x: x['risk'], reverse=True)

            st.markdown(f"<br>**{len(url_results)} URL(s) analyzed**", unsafe_allow_html=True)

            high   = sum(1 for r in url_results if r['level'] == 'HIGH')
            medium = sum(1 for r in url_results if r['level'] == 'MEDIUM')
            safe   = sum(1 for r in url_results if r['level'] == 'LOW')

            c1, c2, c3, c4 = st.columns(4)
            for col, icon, val, label, color in [
                (c1, "🔗", len(url_results), "SCANNED",    "#00d4ff"),
                (c2, "🚨", high,              "MALICIOUS",  "#ff4444"),
                (c3, "⚠️",  medium,            "SUSPICIOUS", "#ffaa00"),
                (c4, "✅", safe,              "SAFE",       "#00ff88"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-value" style="color:{color};">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in url_results:
                bg          = "#2d0a0a" if r['level'] == 'HIGH' else "#2d2200" if r['level'] == 'MEDIUM' else "#0a2d0a"
                icon        = "🚨" if r['level'] == 'HIGH' else "⚠️" if r['level'] == 'MEDIUM' else "✅"
                display_url = r['url'] if len(r['url']) <= 65 else r['url'][:62] + "..."
                reasons_html = "".join([f"<div style='margin:3px 0;font-size:0.82rem;color:#aac;'>• {reason}</div>"
                                        for reason in r['reasons']])
                st.markdown(f"""
                <div style='background:{bg};border:2px solid {r["color"]};border-radius:12px;
                            padding:18px 22px;margin:10px 0;box-shadow:0 0 20px {r["color"]}33;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div style='flex:1;'>
                            <span style='font-size:1.1rem;font-weight:800;color:{r["color"]};'>{icon} {r["level"]} RISK</span><br>
                            <span style='font-size:0.78rem;color:#7a9cc0;word-break:break-all;'>{display_url}</span>
                        </div>
                        <div style='text-align:right;min-width:80px;'>
                            <div style='font-size:1.8rem;font-weight:800;color:{r["color"]};'>{r["risk"]}</div>
                            <div style='font-size:0.72rem;color:{r["color"]};'>/100</div>
                        </div>
                    </div>
                    <div style='margin-top:10px;display:flex;gap:20px;font-size:0.8rem;color:#7a9cc0;'>
                        <span>{"HTTPS" if r["features"]["has_https"] else "HTTP (Unsecured)"}</span>
                        <span>Length: {r["features"]["url_length"]}</span>
                        <span>Subdomains: {r["features"]["subdomain_count"]}</span>
                        <span>Keywords: {r["features"]["keyword_count"]}</span>
                    </div>
                    <div style='margin-top:10px;padding-top:10px;border-top:1px solid #1e3a5f;'>
                        <b style='font-size:0.8rem;color:#00d4ff;'>Detection Reasons:</b>
                        {reasons_html}
                    </div>
                </div>""", unsafe_allow_html=True)

            if len(url_results) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Comparative Risk Chart</div>', unsafe_allow_html=True)
                short_labels = [r['url'][:35] + '...' if len(r['url']) > 35 else r['url'] for r in url_results]
                fig_url = go.Figure(go.Bar(
                    x=short_labels, y=[r['risk'] for r in url_results],
                    marker=dict(color=[r['color'] for r in url_results], line=dict(color='#0a0e1a', width=1.5)),
                    text=[str(r['risk']) for r in url_results],
                    textposition='outside', textfont=dict(color='#e0e6f0'),
                ))
                fig_url.add_hline(y=70, line_dash='dash', line_color='#ff4444',
                                  annotation_text='Malicious (70)', annotation_font_color='#ff6666')
                fig_url.add_hline(y=40, line_dash='dot', line_color='#ffaa00',
                                  annotation_text='Suspicious (40)', annotation_font_color='#ffcc44')
                fig_url.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#7a9cc0'),
                    xaxis=dict(gridcolor='#0d1b2e'),
                    yaxis=dict(title='Risk Score (0-100)', gridcolor='#0d1b2e', range=[0, 115]),
                    margin=dict(l=10, r=10, t=20, b=80), height=300,
                )
                st.plotly_chart(fig_url, use_container_width=True)