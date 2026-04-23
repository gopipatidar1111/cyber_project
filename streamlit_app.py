"""
=======================================================
  AI-Based Cyber Threat Detection & Response System
  Streamlit App — 4 Pages
  Page 1: Dashboard (Overview)
  Page 2: Analyze Network Activity (Input Form)
  Page 3: WiFi Scanner (Real device scan + fallback)
  Page 4: URL Scanner
=======================================================

How to run:
    pip install streamlit plotly joblib scikit-learn pandas numpy
    streamlit run streamlit_app.py

    For real WiFi scan on Windows:
    → Run terminal as Administrator, then: streamlit run streamlit_app.py

Folder structure expected:
    streamlit_app.py
    model.pkl
    scaler.pkl
    encoders.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import time
import subprocess
import platform
import re

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
#  WIFI SCANNER FUNCTIONS
# ─────────────────────────────────────────────

def get_linux_wifi_interface():
    """Detect WiFi interface name on Linux."""
    try:
        result = subprocess.run(['iwconfig'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'IEEE 802.11' in line or 'ESSID' in line:
                return line.split()[0]
    except Exception:
        pass
    return None


def parse_windows_wifi(raw):
    """Parse Windows netsh wlan output into list of network dicts."""
    networks    = []
    current     = {}
    bssid_count = 0

    for line in raw.split('\n'):
        line = line.strip()

        if re.match(r'^SSID\s+\d+\s*:', line) and 'BSSID' not in line:
            if current.get('ssid'):
                networks.append(current)
            current     = {}
            bssid_count = 0
            parts = line.split(':', 1)
            if len(parts) == 2:
                current['ssid'] = parts[1].strip()

        elif 'BSSID' in line and bssid_count == 0:
            # BSSID format: AA:BB:CC:DD:EE:FF — need all colons
            m = re.search(r'([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})', line)
            if m:
                current['bssid'] = m.group(1).upper()
                bssid_count += 1

        elif 'Signal' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                try:
                    current['signal'] = int(parts[1].strip().replace('%', ''))
                except Exception:
                    current['signal'] = 50

        elif 'Authentication' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                current['security'] = parts[1].strip()

        elif 'Encryption' in line and 'encryption' not in current:
            parts = line.split(':', 1)
            if len(parts) == 2:
                current['encryption'] = parts[1].strip()

    if current.get('ssid'):
        networks.append(current)

    for n in networks:
        n.setdefault('bssid',      'N/A')
        n.setdefault('signal',     50)
        n.setdefault('security',   'Unknown')
        n.setdefault('encryption', 'Unknown')

    return [n for n in networks if n.get('ssid')]


def parse_linux_nmcli(raw):
    """Parse nmcli -t output."""
    networks = []
    for line in raw.strip().split('\n'):
        parts = re.split(r'(?<!\\):', line)
        if len(parts) >= 4:
            ssid     = parts[0].replace('\\:', ':').strip()
            bssid    = parts[1].replace('\\:', ':').strip()
            signal   = parts[2].strip()
            security = parts[3].replace('\\:', ':').strip()
            if not ssid:
                continue
            try:
                sig_val = int(signal)
            except Exception:
                sig_val = 50
            networks.append({
                'ssid':       ssid,
                'bssid':      bssid or 'N/A',
                'signal':     sig_val,
                'security':   security if security else 'Open',
                'encryption': 'CCMP' if 'WPA' in security else 'None',
            })
    return networks


def parse_linux_iwlist(raw):
    """Parse iwlist scan output."""
    networks = []
    current  = {}

    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith('Cell'):
            if current.get('ssid'):
                networks.append(current)
            current = {}
            m = re.search(r'Address:\s*([0-9A-F:]+)', line, re.IGNORECASE)
            if m:
                current['bssid'] = m.group(1)
        elif 'ESSID:' in line:
            m = re.search(r'ESSID:"([^"]*)"', line)
            if m:
                current['ssid'] = m.group(1)
        elif 'Signal level' in line:
            m = re.search(r'Signal level[=:](-?\d+)', line)
            if m:
                dbm = int(m.group(1))
                pct = max(0, min(100, 2 * (dbm + 100)))
                current['signal'] = pct
        elif 'Encryption key:' in line:
            current['encryption'] = 'CCMP' if 'on' in line.lower() else 'None'
        elif 'IE: WPA' in line or 'WPA2' in line:
            current['security'] = 'WPA2-Personal'

    if current.get('ssid'):
        networks.append(current)

    for n in networks:
        n.setdefault('bssid',      'N/A')
        n.setdefault('signal',     50)
        n.setdefault('security',   'WPA2-Personal' if n.get('encryption') == 'CCMP' else 'Open')
        n.setdefault('encryption', 'Unknown')

    return [n for n in networks if n.get('ssid')]


def scan_real_wifi():
    """
    Scan actual device WiFi networks.
    Supports Windows (netsh) and Linux (nmcli / iwlist).
    Returns list of network dicts, or None if scan fails.
    """
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.run(
                ['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
                capture_output=True, text=True, timeout=10,
                encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0 and result.stdout.strip():
                networks = parse_windows_wifi(result.stdout)
                if networks:
                    return networks
        except Exception:
            pass
        return None

    elif system == "Linux":
        # Try nmcli first
        try:
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'SSID,BSSID,SIGNAL,SECURITY', 'dev', 'wifi', 'list'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                networks = parse_linux_nmcli(result.stdout)
                if networks:
                    return networks
        except Exception:
            pass

        # Fallback: iwlist
        try:
            iface = get_linux_wifi_interface()
            if iface:
                result = subprocess.run(
                    ['sudo', 'iwlist', iface, 'scan'],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    networks = parse_linux_iwlist(result.stdout)
                    if networks:
                        return networks
        except Exception:
            pass
        return None

    return None  # macOS or unsupported


def generate_simulated_networks():
    """
    Fallback: randomized realistic networks when real scan is not possible.
    Uses timestamp seed so every scan gives fresh results.
    """
    seed = int(time.time()) % 100000
    rng  = np.random.default_rng(seed)

    safe_networks = [
        {'ssid': 'HomeNetwork_5G',      'security': 'WPA3-Personal',  'encryption': 'GCMP'},
        {'ssid': 'NETGEAR_Secure_2.4',  'security': 'WPA2-Personal',  'encryption': 'CCMP'},
        {'ssid': 'Office_Corp_WiFi',    'security': 'WPA2-Enterprise', 'encryption': 'CCMP'},
        {'ssid': 'Jio_Fiber_Home',      'security': 'WPA2-Personal',  'encryption': 'CCMP'},
        {'ssid': 'Airtel_Xstream_5GHz', 'security': 'WPA3-Personal',  'encryption': 'GCMP'},
        {'ssid': 'BSNL_Broadband_2G',   'security': 'WPA2-Personal',  'encryption': 'CCMP'},
        {'ssid': 'MyHome_WiFi_Secured',  'security': 'WPA2-Personal',  'encryption': 'CCMP'},
        {'ssid': 'TP-Link_AC1200',       'security': 'WPA2-Personal',  'encryption': 'CCMP'},
    ]
    suspicious_networks = [
        {'ssid': 'Free_Public_WiFi',   'security': 'Open', 'encryption': 'None'},
        {'ssid': 'Free_Internet_Here', 'security': 'Open', 'encryption': 'None'},
        {'ssid': 'TP-Link_Guest',      'security': 'Open', 'encryption': 'None'},
        {'ssid': 'HotelGuest_WiFi',    'security': 'Open', 'encryption': 'None'},
        {'ssid': 'Airport_FreeWiFi',   'security': 'Open', 'encryption': 'None'},
        {'ssid': 'D-Link_Default',     'security': 'WEP',  'encryption': 'WEP'},
        {'ssid': 'Linksys_Open',       'security': 'Open', 'encryption': 'None'},
        {'ssid': 'CafeWiFi_Unsecured', 'security': 'Open', 'encryption': 'None'},
    ]
    dangerous_networks = [
        {'ssid': 'HomeNetwork_5G',      'security': 'WPA2-Personal', 'encryption': 'CCMP'},
        {'ssid': 'Office_Corp_WiFi',    'security': 'Open',          'encryption': 'None'},
        {'ssid': 'PayPal_Verify_Login', 'security': 'Open',          'encryption': 'None'},
        {'ssid': 'BankSecure_Update',   'security': 'WPA2-Personal', 'encryption': 'CCMP'},
        {'ssid': 'HackNet_v2',          'security': 'Open',          'encryption': 'None'},
        {'ssid': 'WIN_FREE_PRIZE_NOW',  'security': 'Open',          'encryption': 'None'},
    ]

    n_safe = int(rng.integers(3, 6))
    n_susp = int(rng.integers(1, 4))
    n_dang = int(rng.integers(1, 3))

    chosen = (
        [safe_networks[i]       for i in rng.choice(len(safe_networks),       n_safe, replace=False)] +
        [suspicious_networks[i] for i in rng.choice(len(suspicious_networks), n_susp, replace=False)] +
        [dangerous_networks[i]  for i in rng.choice(len(dangerous_networks),  n_dang, replace=False)]
    )
    rng.shuffle(chosen)

    def random_bssid(rng):
        return ':'.join(f'{int(rng.integers(0, 256)):02X}' for _ in range(6))

    result = []
    for net in chosen:
        net = dict(net)
        net['bssid']  = random_bssid(rng)
        net['signal'] = int(rng.integers(70, 98)) if (
            net['security'] == 'Open' or any(w in net['ssid'] for w in ['Hack', 'WIN', 'Prize'])
        ) else int(rng.integers(35, 85))
        result.append(net)
    return result


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
    st.plotly_chart(fig_line, width="stretch")

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
        st.plotly_chart(fig_donut, width="stretch")

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
    st.plotly_chart(fig_bar, width="stretch")

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

        gauge_col, detail_col = st.columns([1, 1])
        with gauge_col:
            st.markdown('<div class="section-header">📊 Threat Risk Meter</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score,
                number=dict(font=dict(size=40, color=risk_color), suffix="/100"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor='#7a9cc0', tickfont=dict(color='#7a9cc0')),
                    bar=dict(color=risk_color, thickness=0.25),
                    bgcolor='#0d1b2e', bordercolor='#1e3a5f',
                    steps=[
                        dict(range=[0,  40], color='rgba(0,255,136,0.15)'),
                        dict(range=[40, 70], color='rgba(255,170,0,0.15)'),
                        dict(range=[70,100], color='rgba(255,68,68,0.2)'),
                    ],
                    threshold=dict(line=dict(color=risk_color, width=3), thickness=0.8, value=risk_score),
                ),
                title=dict(text=f"RISK LEVEL: <b>{risk_level}</b>", font=dict(size=16, color=risk_color)),
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                                     margin=dict(l=20, r=20, t=40, b=20), font=dict(color='#e0e6f0'))
            st.plotly_chart(fig_gauge, width="stretch")

        with detail_col:
            st.markdown('<div class="section-header">📋 Connection Summary</div>', unsafe_allow_html=True)
            details = {
                "Protocol": protocol, "Service": service.upper(), "Flag": flag,
                "Duration": f"{duration}s", "Src Bytes": f"{src_bytes:,}",
                "Dst Bytes": f"{dst_bytes:,}", "Request Count": count,
                "Confidence": f"{confidence*100:.1f}%",
            }
            for k, v in details.items():
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid #1e3a5f;font-size:0.9rem;'>
                    <span style='color:#7a9cc0;'>{k}</span>
                    <span style='color:#e0e6f0;font-weight:600;'>{v}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 Automated Response Recommendation</div>', unsafe_allow_html=True)

        action_col1, action_col2, action_col3 = st.columns(3)
        actions = {
            "HIGH":   ("action-block-high",   "🚫 BLOCK IP ADDRESS",   "Immediately terminate connection and blacklist source IP."),
            "MEDIUM": ("action-block-medium", "👁️  MONITOR & LOG",     "Flag for human review. Increase monitoring frequency."),
            "LOW":    ("action-block-low",    "✅ NO ACTION REQUIRED", "Connection appears safe. Continue normal operations."),
        }
        for col, (level, (css, action_title, action_desc)) in zip([action_col1, action_col2, action_col3], actions.items()):
            active = "border: 3px solid" if level == risk_level else "opacity: 0.35;"
            with col:
                st.markdown(f"""
                <div class="action-block {css}" style="{active}">
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
                width="stretch", hide_index=True)

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
#  PAGE 3 — WiFi THREAT SCANNER
# ═══════════════════════════════════════════════════════════
elif page == "📡  WiFi Scanner":

    st.markdown('<div class="page-title">📡 WiFi THREAT SCANNER</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Scan nearby WiFi networks and detect rogue access points & threats</div><br>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1b2e;border:1px solid #1e3a5f;border-radius:12px;
                padding:16px 20px;margin-bottom:20px;font-size:0.88rem;color:#7a9cc0;'>
        <b style='color:#00d4ff;'>How WiFi Threat Detection Works</b><br><br>
        Each nearby network is analyzed for:
        <span style='color:#ffaa00;'>Open Security</span> |
        <span style='color:#ff4444;'>Evil Twin / Duplicate SSID</span> |
        <span style='color:#ff7722;'>Suspicious Names</span> |
        <span style='color:#cc44ff;'>Weak Encryption (WEP)</span><br><br>
        <b style='color:#ffaa00;'>⚠️ Windows tip:</b> For real WiFi scan, run your terminal
        as <b>Administrator</b> before starting Streamlit.
    </div>
    """, unsafe_allow_html=True)

    _, scan_col, _ = st.columns([1, 2, 1])
    with scan_col:
        scan_clicked = st.button("📡  SCAN NEARBY WiFi NETWORKS")

    if scan_clicked:
        with st.spinner("🔍 Scanning nearby WiFi networks..."):
            time.sleep(1.0)
            real_networks = scan_real_wifi()
            used_real     = real_networks is not None and len(real_networks) > 0

        if used_real:
            networks = real_networks
            st.success(f"✅ **Real scan complete** — {len(networks)} networks detected from your device", icon="📡")
        else:
            networks = generate_simulated_networks()
            st.info(
                "ℹ️ **Simulated scan** — Real WiFi scan unavailable on this environment. "
                "On Windows, run the terminal as **Administrator** and ensure WiFi is enabled. "
                "Each simulated scan generates fresh randomized results.",
                icon="💡"
            )

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
        st.plotly_chart(fig_wifi, width="stretch")


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
                st.plotly_chart(fig_url, width="stretch")

                