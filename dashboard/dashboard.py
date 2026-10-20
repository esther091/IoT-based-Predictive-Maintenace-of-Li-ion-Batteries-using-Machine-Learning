"""
dashboard.py
Streamlit dashboard for real-time battery health monitoring.
Polls the FastAPI backend every 5 seconds and displays live data,
SOH gauge, RUL countdown, and alert banners.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time

# Backend address. Locally: 127.0.0.1, not "localhost" -- on Windows "localhost"
# tries IPv6 (::1) first, and since the backend only listens on IPv4 every
# request stalls ~2 s. Online: set BACKEND_URL (a Render environment variable) to the
# deployed backend, e.g. https://<name>.onrender.com
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REFRESH_INTERVAL = 5

st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .dashboard-header h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #0F2A47;
        margin: 0;
    }
    .dashboard-subtitle {
        color: #6B7684;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E7EAEE;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 42, 71, 0.06);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8A93A0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F2A47;
    }
    .metric-unit {
        font-size: 1rem;
        font-weight: 500;
        color: #9AA3AF;
        margin-left: 4px;
    }

    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F2A47;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .alert-banner {
        border-radius: 12px;
        padding: 16px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-urgent   { background: #FDECEC; color: #B3261E; border: 1px solid #F5C2C0; }
    .alert-critical { background: #FFF1E0; color: #B5540A; border: 1px solid #F7D6A8; }
    .alert-warning  { background: #FFF8E1; color: #8A6D00; border: 1px solid #F2E2A0; }
    .alert-normal   { background: #E9F7EF; color: #1E7E34; border: 1px solid #B7E4C7; }
    .alert-pending  { background: #EEF2F7; color: #4A5568; border: 1px solid #DCE2EA; }

    .section-gap { margin-top: 28px; margin-bottom: 10px; }

    .status-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #2ECC71; }
    .status-offline { background: #E74C3C; }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E7EAEE;
    }
</style>
""", unsafe_allow_html=True)


def fetch_latest():
    try:
        r = requests.get(f"{BACKEND_URL}/latest", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_history():
    try:
        r = requests.get(f"{BACKEND_URL}/history?limit=100", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def metric_card(label, value, unit):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        </div>
    """, unsafe_allow_html=True)


def soh_gauge(soh_value):
    if soh_value is None:
        soh_value = 0.0

    if soh_value >= 80:
        color = "#2ECC71"
    elif soh_value >= 60:
        color = "#F5B301"
    else:
        color = "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh_value,
        number={"suffix": "%", "font": {"size": 40, "color": "#0F2A47"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#B0B8C1", "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "#FDECEC"},
                {"range": [60, 80], "color": "#FFF6DD"},
                {"range": [80, 100], "color": "#E9F7EF"},
            ],
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(t=10, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


def styled_line_chart(df, y_col, title, color):
    fig = px.line(df, x="timestamp", y=y_col)
    fig.update_traces(line=dict(color=color, width=2.5))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#0F2A47")),
        height=230,
        margin=dict(t=36, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=None, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#F0F2F5", title=None, tickfont=dict(size=10)),
    )
    return fig


def main():
    health = fetch_health()
    latest = fetch_latest()
    history = fetch_history()

    reading = latest.get("reading") if latest else None
    prediction = latest.get("prediction") if latest else None
    cycle = latest.get("cycle") if latest else None
    # a PARTIAL_DISCHARGE row has no SOH/RUL values
    has_pred = bool(prediction) and prediction.get("soh_final") is not None

    # ---- Header ----
    status_class = "status-online" if health else "status-offline"
    status_text = "Backend connected" if health else "Backend offline"

    st.markdown(f"""
        <div class="dashboard-header">
            <span style="font-size: 2.2rem;">🔋</span>
            <h1>Battery Predictive Maintenance</h1>
        </div>
        <div class="dashboard-subtitle">
            <span class="status-dot {status_class}"></span>{status_text}
            &nbsp;·&nbsp; IoT-Based Predictive Maintenance of Li-ion Batteries
        </div>
    """, unsafe_allow_html=True)

    # ---- SAFETY alarms (shown above everything else) ----
    if health and health.get("is_discharging"):
        alarm_v = health.get("low_cell_alarm_v", 2.8)
        if reading and reading["current"] < -0.5 and reading["voltage"] < alarm_v:
            st.markdown(
                f'<div class="alert-banner alert-urgent" style="font-size:1.25rem;">🛑 SWITCH THE LOAD OFF NOW — '
                f'{reading["voltage"]:.2f} V per cell is below the safe limit ({alarm_v} V). The BMS should have cut off: '
                f'check its wiring before the next test.</div>', unsafe_allow_html=True)
        silent = health.get("seconds_since_reading") or 0
        if silent > 10 and reading and reading["voltage"] <= 3.2:
            st.markdown(
                f'<div class="alert-banner alert-normal">🔌 Readings stopped {silent:.0f} s ago at the end of the '
                f'discharge ({reading["voltage"]:.2f} V per cell): the BMS has cut off. Switch the load OFF. '
                f'The prediction appears within about 30 seconds.</div>', unsafe_allow_html=True)
        elif silent > 30:
            st.markdown(
                f'<div class="alert-banner alert-urgent">⚠️ No readings for {silent:.0f} s during a discharge. '
                f'Switch OFF and check the rig: the pack may be below 7 V (the PZEM then loses power), or the ESP32 / '
                f'WiFi is disconnected. After 3 minutes the run is closed as "interrupted" with no prediction.</div>',
                unsafe_allow_html=True)

    # ---- Live discharge progress ----
    if health and health.get("is_discharging"):
        st.markdown(
            f'<div class="alert-banner alert-pending">🔄 Discharging — {health.get("discharge_mah", 0):.0f} mAh '
            f'delivered in {health.get("discharge_minutes", 0):.0f} min. The prediction appears when the pack '
            f'reaches its cut-off.</div>', unsafe_allow_html=True)

    # ---- Alert banner ----
    if prediction and prediction.get("alert_level") == "NOT_STARTED_FULL":
        st.markdown(
            '<div class="alert-banner alert-pending">🔋 Last discharge did not start from a full pack, so no '
            'prediction was made. Charge every cell to 4.15–4.20 V before the next test.</div>',
            unsafe_allow_html=True)
    elif prediction and prediction.get("alert_level") == "INTERRUPTED":
        st.markdown(
            '<div class="alert-banner alert-pending">⛔ Last discharge was interrupted — readings stopped before it '
            'ended, so no prediction was made. Check the rig and repeat the test from a full charge.</div>',
            unsafe_allow_html=True)
    elif prediction and prediction.get("alert_level") == "PARTIAL_DISCHARGE":
        stop_v = f'{cycle["min_cell_voltage"]:.2f} V/cell' if cycle and cycle.get("min_cell_voltage") else "early"
        mah = f'{cycle["capacity_mah"]:.0f} mAh' if cycle and cycle.get("capacity_mah") is not None else "some charge"
        st.markdown(
            f'<div class="alert-banner alert-pending">⏸ Partial discharge — stopped at {stop_v} after {mah}. '
            f'SOH is only predicted from a complete discharge (down to the cut-off).</div>',
            unsafe_allow_html=True)
    elif prediction:
        alert = prediction.get("alert_level", "NORMAL")
        rul = prediction.get("rul_final", 0) or 0
        replace_by = (datetime.now() + timedelta(days=rul)).strftime("%d %b %Y")

        banners = {
            "URGENT":   ("alert-urgent", f"🚨 URGENT — Replace battery now. {rul:.0f} days remaining."),
            "CRITICAL": ("alert-critical", f"⚠️ CRITICAL — Schedule replacement soon. {rul:.0f} days remaining."),
            "WARNING":  ("alert-warning", f"⚡ WARNING — Battery ageing. Replace by {replace_by}."),
            "NORMAL":   ("alert-normal", f"✅ Battery healthy — estimated {rul:.0f} days remaining (replace by {replace_by})."),
        }
        if alert == "MODELS_NOT_LOADED":
            banners[alert] = ("alert-pending", "⚠️ ML models not loaded on the backend — run ml/train_models.py, then restart the backend.")
        # never fall back to the green "healthy" banner for an unrecognised level
        cls, msg = banners.get(alert, ("alert-pending", f"❔ Unknown alert level: {alert}"))
        st.markdown(f'<div class="alert-banner {cls}">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="alert-banner alert-pending">⏳ Waiting for first discharge cycle to complete...</div>',
            unsafe_allow_html=True
        )

    # ---- Live metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Voltage (per cell)", f"{reading['voltage']:.2f}" if reading else "--", "V")
    with c2: metric_card("Current", f"{reading['current']:.2f}" if reading else "--", "A")
    with c3: metric_card("Power (per cell)", f"{reading['power']:.1f}" if reading else "--", "W")
    with c4: metric_card("Energy (per cell)", f"{reading['energy']:.1f}" if reading else "--", "Wh")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- SOH / RUL / Model comparison ----
    col_soh, col_rul, col_models = st.columns([1.1, 1, 1.3])

    with col_soh:
        with st.container(border=True):
            st.markdown('<div class="panel-title">🔵 State of Health</div>', unsafe_allow_html=True)
            soh_val = prediction["soh_final"] if has_pred else 0
            st.plotly_chart(soh_gauge(soh_val), width="stretch", config={"displayModeBar": False})

    with col_rul:
        with st.container(border=True):
            st.markdown('<div class="panel-title">⏳ Remaining Useful Life</div>', unsafe_allow_html=True)
            if has_pred:
                rul = prediction.get("rul_final", 0) or 0
                replace_by = (datetime.now() + timedelta(days=rul)).strftime("%d %b %Y")
                st.markdown(f"""
                    <div style="text-align:center; padding-top: 18px;">
                        <div style="font-size: 2.6rem; font-weight: 800; color: #0F2A47;">{rul:.0f}</div>
                        <div style="color:#8A93A0; font-size:0.85rem; margin-top:-4px;">days remaining</div>
                        <div style="margin-top:16px; color:#4A5568; font-size:0.85rem;">Replace before <b>{replace_by}</b></div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="text-align:center; padding-top: 30px; color:#9AA3AF;">
                        No prediction yet
                    </div>
                """, unsafe_allow_html=True)

    with col_models:
        with st.container(border=True):
            st.markdown('<div class="panel-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
            if has_pred:
                df_models = pd.DataFrame({
                    "Model": ["Random Forest", "XGBoost", "Ensemble"],
                    "SOH (%)": [prediction.get("soh_rf"), prediction.get("soh_xgb"), prediction.get("soh_final")],
                    "RUL (days)": [prediction.get("rul_rf"), prediction.get("rul_xgb"), prediction.get("rul_final")],
                })
                st.dataframe(df_models, hide_index=True, width="stretch")
            else:
                st.markdown('<div style="color:#9AA3AF; padding-top:20px;">No data yet</div>', unsafe_allow_html=True)

    # ---- Validation: measured vs predicted (last discharge) ----
    if cycle and cycle.get("capacity_mah") is not None:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        v1, v2, v3, v4 = st.columns(4)
        measured = cycle.get("measured_soh")
        with v1: metric_card(f"Charge delivered (cycle {cycle['cycle_number']})", f"{cycle['capacity_mah']:.0f}", "mAh")
        with v2: metric_card("Measured SOH (mAh ÷ 2000)", f"{measured:.1f}" if measured is not None else "--", "%")
        with v3: metric_card("Predicted SOH (ensemble)", f"{prediction['soh_final']:.1f}" if has_pred else "--", "%")
        diff = (prediction["soh_final"] - measured) if (has_pred and measured is not None) else None
        with v4: metric_card("Prediction error", f"{diff:+.1f}" if diff is not None else "--", "pts")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- Time series charts ----
    if history:
        df = pd.DataFrame(history)
        # the backend stores UTC; show the laptop's local time (e.g. Nigeria = UTC+1)
        df["timestamp"] = (pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
                           .dt.tz_convert(datetime.now().astimezone().tzinfo).dt.tz_localize(None))

        ch1, ch2 = st.columns(2)
        with ch1:
            with st.container(border=True):
                st.plotly_chart(styled_line_chart(df, "voltage", "Voltage over Time", "#2563EB"),
                                 width="stretch", config={"displayModeBar": False})
        with ch2:
            with st.container(border=True):
                st.plotly_chart(styled_line_chart(df, "current", "Current over Time", "#D97706"),
                                 width="stretch", config={"displayModeBar": False})

        ch3, ch4 = st.columns(2)
        with ch3:
            with st.container(border=True):
                st.plotly_chart(styled_line_chart(df, "power", "Power over Time", "#7C3AED"),
                                 width="stretch", config={"displayModeBar": False})
        with ch4:
            with st.container(border=True):
                st.plotly_chart(styled_line_chart(df, "energy", "Energy over Time", "#059669"),
                                 width="stretch", config={"displayModeBar": False})

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel-title">📋 Raw Readings</div>', unsafe_allow_html=True)
            st.dataframe(
                df[["timestamp", "voltage", "current", "power", "energy"]],
                width="stretch", hide_index=True
            )
    else:
        st.info("No readings yet. Waiting for ESP32 data...")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} · Auto-refreshes every {REFRESH_INTERVAL}s")

    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()