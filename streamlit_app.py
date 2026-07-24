"""
streamlit_app.py
================
CSIR Thunderstorm Nowcast System — Operational Dashboard
Bengaluru Airport (IMD Station 43295)

Deploy to Streamlit Cloud:
  streamlit run streamlit_app.py

Author: Aprameya, CSIR Thunderstorm Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSIR Thunderstorm Nowcast — Bengaluru Airport",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

SLOT_NAMES  = {0:"0001–0600 IST",1:"0601–1200 IST",2:"1201–1800 IST",3:"1801–2400 IST"}
SLOT_LABELS = {0:"Late Night",1:"Morning",2:"Afternoon",3:"Evening"}
SLOT_COLORS = {0:"#4A90D9",1:"#27AE60",2:"#E67E22",3:"#8E44AD"}
SLOT_EMOJI  = {0:"🌙",1:"🌅",2:"☀️",3:"🌆"}

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .slot-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #E67E22;
        text-align: center;
        margin-bottom: 1rem;
    }
    .alert-banner {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .clear-banner {
        background: linear-gradient(135deg, #27ae60, #1e8449);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for slot_id in range(4):
        for suffix in ['_xgb_v3_calibrated','_xgb_v3','_xgb_v2_calibrated','_xgb_v2']:
            path = BASE / "models" / f"nowcast_slot{slot_id}{suffix}.pkl"
            if path.exists():
                models[slot_id] = joblib.load(path)
                break
    return models

@st.cache_data
def load_dataset():
    for fname in ['bengaluru_6hr_training_dataset_v3.csv',
                  'bengaluru_6hr_training_dataset_v2.csv']:
        path = BASE / "data" / fname
        if path.exists():
            return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()

@st.cache_data
def load_shap():
    for fname in ['shap_per_slot_importance_v3.csv','shap_per_slot_importance_v2.csv']:
        path = BASE / "results" / fname
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_metrics():
    path = BASE / "results" / "evaluation_results_per_slot_v3.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_synoptic():
    path = BASE / "results" / "synoptic_skill_per_regime.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def load_forecast_log():
    path = BASE / "data" / "forecast_log.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()

# ── PREDICTION ENGINE ─────────────────────────────────────────────────────────
def apply_calibrator(artifact, raw_prob):
    cal = artifact.get('calibrator')
    if cal is None: return raw_prob
    if artifact.get('calib_method') == 'sigmoid':
        return cal.predict_proba(np.array([[raw_prob]]))[0][1]
    return float(cal.predict([raw_prob])[0])

def compute_derived(obs, slot_id):
    m   = int(obs.get('month', datetime.now().month))
    doy = int(obs.get('doy', datetime.now().timetuple().tm_yday))

    obs['DTR']        = obs.get('MAX',30) - obs.get('MIN',20)
    obs['HA_flag']    = 0
    obs['RF_nonzero'] = 1 if obs.get('RF',0) > 0 else 0
    obs['SEASON']     = 1 if m in [3,4,5] else 2 if m in [6,7,8,9] else 3 if m in [10,11] else 0
    obs['MONTH_sin']  = math.sin(2*math.pi*m/12)
    obs['MONTH_cos']  = math.cos(2*math.pi*m/12)
    obs['DOY_sin']    = math.sin(2*math.pi*doy/365)
    obs['DOY_cos']    = math.cos(2*math.pi*doy/365)
    obs['doy_sin']    = obs['DOY_sin']
    obs['doy_cos']    = obs['DOY_cos']
    obs['slot_sin']   = math.sin(2*math.pi*slot_id/4)
    obs['slot_cos']   = math.cos(2*math.pi*slot_id/4)
    CLIM = {
        (4,0):0.025,(4,1):0.008,(4,2):0.129,(4,3):0.100,
        (5,0):0.129,(5,1):0.036,(5,2):0.194,(5,3):0.181,
        (6,0):0.042,(6,1):0.004,(6,2):0.096,(6,3):0.092,
        (7,0):0.028,(7,1):0.004,(7,2):0.032,(7,3):0.044,
        (8,0):0.020,(8,1):0.008,(8,2):0.052,(8,3):0.060,
        (9,0):0.083,(9,1):0.029,(9,2):0.079,(9,3):0.067,
        (10,0):0.077,(10,1):0.024,(10,2):0.077,(10,3):0.077,
    }
    obs['slot_month_clim'] = CLIM.get((m,slot_id), 0.02)
    obs['ERA5_CAPE'] = obs.get('ERA5_CAPE', obs.get('CAPE', 0.0))
    obs['CIN']       = obs.get('CIN', 0.0)

    q850=obs.get('ERA5_q_850hPa',0.013); q700=obs.get('ERA5_q_700hPa',0.009)
    q500=obs.get('ERA5_q_500hPa',0.003); t850=obs.get('ERA5_t_850hPa',293.0)
    t500=obs.get('ERA5_t_500hPa',268.0); u850=obs.get('ERA5_u_850hPa',-3.0)
    v850=obs.get('ERA5_v_850hPa',2.0);   u700=obs.get('ERA5_u_700hPa',2.0)
    v700=obs.get('ERA5_v_700hPa',1.0);   u500=obs.get('ERA5_u_500hPa',5.0)
    v500=obs.get('ERA5_v_500hPa',2.0)
    CAPE=obs.get('CAPE',0.0); K=obs.get('K_INDEX',30.0)
    LI=obs.get('LIFTED_INDEX',-2.0); TT=obs.get('TOTALS_TOTALS',44.0)

    obs['cape_x_kindex']      = CAPE * K
    obs['li_x_totals']        = abs(LI) * TT
    obs['q_gradient_500_850'] = q850 - q500
    obs['thetae_850']         = t850 + 2491 * q850
    obs['wind_shear_500_850'] = ((u500-u850)**2 + (v500-v850)**2)**0.5
    obs['wind_shear_700_850'] = ((u700-u850)**2 + (v700-v850)**2)**0.5
    obs['moisture_flux_850']  = q850 * (u850**2 + v850**2)**0.5
    obs['moisture_flux_700']  = q700 * (u700**2 + v700**2)**0.5
    obs['thickness_500_850']  = t850 - t500
    obs['mid_level_drying']   = q700 / (q850 + 1e-9)
    return obs

def predict_all_slots(models, obs_base):
    results = {}
    for slot_id in range(4):
        artifact     = models.get(slot_id)
        if artifact is None:
            results[slot_id] = {'probability':0.0,'predicted':False,'threshold':0.5}
            continue
        feature_cols = artifact['feature_cols']
        threshold    = artifact['threshold']
        model        = artifact['model']
        obs = {**obs_base, 'slot': slot_id}
        obs = compute_derived(obs, slot_id)
        X   = np.array([[float(obs.get(c, 0.0)) for c in feature_cols]])
        raw = float(model.predict_proba(X)[0][1])
        cal = apply_calibrator(artifact, raw)
        results[slot_id] = {
            'probability': float(cal),
            'predicted':   float(cal) >= threshold,
            'threshold':   threshold,
        }
    return results

# ── GAUGE CHART ───────────────────────────────────────────────────────────────
def make_gauge(prob, slot_id, slot_name):
    pct = prob * 100
    if pct < 20:   color = "#27AE60"
    elif pct < 40: color = "#F39C12"
    elif pct < 60: color = "#E67E22"
    else:           color = "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1,
                     'tickcolor': "#666", 'nticks': 6},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 20],  'color': '#d5f5e3'},
                {'range': [20, 40], 'color': '#fef9e7'},
                {'range': [40, 60], 'color': '#fdebd0'},
                {'range': [60, 100],'color': '#fdedec'},
            ],
            'threshold': {
                'line': {'color': "#2C3E50", 'width': 3},
                'thickness': 0.75,
                'value': 45,
            }
        },
        title={'text': f"{SLOT_EMOJI[slot_id]} {slot_name}<br>"
                       f"<span style='font-size:12px;color:#888'>"
                       f"{SLOT_LABELS[slot_id]}</span>",
               'font': {'size': 14}},
    ))
    fig.update_layout(
        height=220, margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# ── TIMELINE BAR ──────────────────────────────────────────────────────────────
def make_timeline(results):
    fig = go.Figure()
    for slot_id in range(4):
        prob  = results[slot_id]['probability'] * 100
        color = "#E74C3C" if results[slot_id]['predicted'] else "#27AE60"
        fig.add_trace(go.Bar(
            x=[prob], y=[SLOT_NAMES[slot_id]],
            orientation='h',
            marker_color=color,
            text=[f"{prob:.1f}%"],
            textposition='inside',
            name=SLOT_NAMES[slot_id],
            showlegend=False,
        ))
    fig.add_vline(x=45, line_dash="dash", line_color="#2C3E50",
                  annotation_text="Default threshold", annotation_position="top right")
    fig.update_layout(
        height=200, margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(range=[0,100], title="Probability (%)"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,1)',
        barmode='overlay',
    )
    return fig

# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:1.8rem">⛈️ CSIR Thunderstorm Nowcast System</h1>
        <p style="margin:0.5rem 0 0 0;opacity:0.85;font-size:1rem">
            Kempegowda International Airport — IMD Station 43295, Bengaluru
        </p>
        <p style="margin:0.3rem 0 0 0;opacity:0.65;font-size:0.85rem">
            XGBoost v3 Calibrated Models | 6-Hourly ERA5 | Real-Time GFS Pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    with st.spinner("Loading models..."):
        models  = load_models()
        dataset = load_dataset()
        shap_df = load_shap()
        metrics_df = load_metrics()
        synoptic_df = load_synoptic()

    if len(models) == 0:
        st.error("No models found. Make sure model .pkl files are in the models/ folder.")
        return

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/b2/CSIR_logo.png/220px-CSIR_logo.png",
                 width=120)
        st.markdown("### ⚙️ Forecast Settings")

        mode = st.radio("Mode", ["📅 Date Lookup", "✏️ Manual Input"],
                        help="Date Lookup uses historical data. Manual Input lets you enter observations.")

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown(f"**Models loaded:** {len(models)}/4")
        st.markdown("**Version:** Calibrated v3")
        st.markdown("**Features:** 64 (ERA5 6-hrly + derived)")
        st.markdown("**Training:** 2015–2022")
        st.markdown("**Test:** 2023–2025")

        st.markdown("---")
        st.markdown("### 🎯 Thresholds")
        if len(models) > 0:
            for slot_id in range(4):
                if slot_id in models:
                    t = models[slot_id]['threshold']
                    st.markdown(f"Slot {slot_id}: **{t}**")

        st.markdown("---")
        st.caption("CSIR Thunderstorm Prediction System\nCollaboration with IMD Bengaluru\nDr. Geeta Agnihotri, Scientist F")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "⛈️ Forecast", "📈 Performance", "🔬 SHAP Analysis", "🗺️ Synoptic Regimes"
    ])

    # ── TAB 1: FORECAST ───────────────────────────────────────────────────────
    with tab1:
        if mode == "📅 Date Lookup" and len(dataset) > 0:
            col_date, col_info = st.columns([2,3])
            with col_date:
                min_date = dataset[dataset['year']>=2023]['date'].min().date()
                max_date = dataset['date'].max().date()
                selected_date = st.date_input(
                    "Select date",
                    value=datetime(2023, 4, 29).date(),
                    min_value=min_date, max_value=max_date,
                    help="Dates from 2023-2025 are test set (model never saw these during training)"
                )
            with col_info:
                st.info(f"📌 Showing historical forecast for **{selected_date}**\n\n"
                        f"Green badges = correct | Red = missed | Orange = false alarm")

            date_str  = str(selected_date)
            day_data  = dataset[dataset['date']==date_str]
            if len(day_data) == 0:
                st.warning(f"No data found for {date_str}")
                return

            FEATURE_COLS = [c for c in dataset.columns if c not in
                            ['date','year','month','slot','slot_label','ts_label']]
            obs_base = {c: float(day_data.iloc[0][c]) for c in FEATURE_COLS
                        if c in day_data.columns}
            obs_base['month'] = int(pd.Timestamp(date_str).month)
            obs_base['doy']   = int(pd.Timestamp(date_str).dayofyear)

            actual_labels = {int(r['slot']): int(r['ts_label'])
                             for _, r in day_data.iterrows()}

        else:
            st.markdown("#### Enter Today's Observations")
            c1, c2, c3 = st.columns(3)
            with c1:
                MAX   = st.number_input("Max Temp (°C)", 20.0, 45.0, 29.0, 0.5)
                MIN   = st.number_input("Min Temp (°C)", 10.0, 35.0, 21.0, 0.5)
                RF    = st.number_input("Rainfall (mm)", 0.0, 200.0, 0.0, 0.5)
            with c2:
                CAPE  = st.number_input("CAPE (J/kg)", 0.0, 5000.0, 300.0, 50.0)
                K_IDX = st.number_input("K-Index", 0.0, 50.0, 35.0, 0.5)
                LI    = st.number_input("Lifted Index", -10.0, 5.0, -2.0, 0.5)
            with c3:
                TT    = st.number_input("Totals-Totals", 30.0, 60.0, 45.0, 0.5)
                T2M   = st.number_input("ERA5 T2M (K)", 280.0, 315.0, 299.0, 0.5)
                ECAPE = st.number_input("ERA5 CAPE (J/kg)", 0.0, 5000.0, 300.0, 50.0)

            obs_base = {
                'MAX':MAX,'MIN':MIN,'RF':RF,'AW':3.0,'EVP':5.0,'DRNRF':0.0,'SSH':300.0,
                'RF_3d':0.0,'RF_7d':0.0,'MAX_3d_avg':MAX,'MIN_3d_avg':MIN,'DTR_3d_avg':MAX-MIN,
                'RF_lag1':0.0,'MAX_lag1':MAX,'MIN_lag1':MIN,'LABEL_lag1':0,
                'CAPE':CAPE,'K_INDEX':K_IDX,'LIFTED_INDEX':LI,'TOTALS_TOTALS':TT,
                'PRECIP_WATER':40.0,'ERA5_T2M':T2M,'ERA5_D2M':293.0,
                'ERA5_U10':-3.0,'ERA5_V10':2.0,'ERA5_CAPE':ECAPE,'ERA5_SP':91500.0,
                'ERA5_t_500hPa':268.0,'ERA5_t_700hPa':283.0,'ERA5_t_850hPa':293.0,
                'ERA5_q_500hPa':0.003,'ERA5_q_700hPa':0.009,'ERA5_q_850hPa':0.013,
                'ERA5_u_500hPa':5.0,'ERA5_u_700hPa':2.0,'ERA5_u_850hPa':-3.0,
                'ERA5_v_500hPa':2.0,'ERA5_v_700hPa':1.0,'ERA5_v_850hPa':2.0,
                'ts_label_lag1_slot':0,'ts_any_yesterday':0,'CIN':0.0,
                'month': datetime.now().month,'doy': datetime.now().timetuple().tm_yday,
            }
            actual_labels = None
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Run prediction
        with st.spinner("Running forecast..."):
            results = predict_all_slots(models, obs_base)

        # Alert banner
        alert_slots = [s for s, r in results.items() if r['predicted']]
        if alert_slots:
            peak = max(alert_slots, key=lambda s: results[s]['probability'])
            st.markdown(f"""
            <div class="alert-banner">
                🔴 THUNDERSTORM ALERT — {len(alert_slots)} window(s) at risk |
                Peak: {SLOT_NAMES[peak]} ({results[peak]['probability']*100:.1f}%)
            </div>""", unsafe_allow_html=True)
        else:
            max_slot = max(results, key=lambda s: results[s]['probability'])
            st.markdown(f"""
            <div class="clear-banner">
                🟢 ALL CLEAR — No thunderstorm predicted |
                Highest: {results[max_slot]['probability']*100:.1f}% ({SLOT_NAMES[max_slot]})
            </div>""", unsafe_allow_html=True)

        # Gauge charts
        cols = st.columns(4)
        for slot_id in range(4):
            with cols[slot_id]:
                res   = results[slot_id]
                prob  = res['probability']
                fig   = make_gauge(prob, slot_id, SLOT_NAMES[slot_id])
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{slot_id}")

                # Badge
                if res['predicted']:
                    st.markdown(f"<div style='text-align:center;background:#e74c3c;color:white;"
                                f"border-radius:20px;padding:4px 12px;font-weight:bold;"
                                f"font-size:0.9rem'>⚠ ALERT</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center;background:#27ae60;color:white;"
                                f"border-radius:20px;padding:4px 12px;font-weight:bold;"
                                f"font-size:0.9rem'>✓ CLEAR</div>", unsafe_allow_html=True)

                # Actual label if available
                if actual_labels:
                    actual = actual_labels.get(slot_id, 0)
                    pred   = int(res['predicted'])
                    if pred==1 and actual==1:   badge = "✓ HIT"
                    elif pred==0 and actual==1: badge = "✗ MISS"
                    elif pred==1 and actual==0: badge = "✗ FALSE ALARM"
                    else:                        badge = "✓ CORRECT NO"
                    color = "#27ae60" if badge.startswith("✓") else "#e74c3c"
                    st.markdown(f"<div style='text-align:center;color:{color};"
                                f"font-size:0.85rem;margin-top:4px;font-weight:bold'>"
                                f"{badge}</div>", unsafe_allow_html=True)

        # Timeline bar
        st.markdown("#### 24-Hour Risk Timeline")
        st.plotly_chart(make_timeline(results), use_container_width=True)

        # JSON output
        with st.expander("📋 JSON Output (for API integration)"):
            import json
            output = [
                {"date": date_str, "slot": s,
                 "slot_label": SLOT_NAMES[s],
                 "ts_probability": round(r['probability'], 4),
                 "ts_predicted": r['predicted'],
                 "threshold_used": r['threshold'],
                 "model": "calibrated_v3"}
                for s, r in results.items()
            ]
            st.json(output)

    # ── TAB 2: PERFORMANCE ────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Model Performance — Test Set 2023–2025")

        if len(metrics_df) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Per-Slot Metrics")
                display_metrics = metrics_df[['Slot_label','AUROC','POD','FAR','CSI','HSS','Threshold']].copy()
                display_metrics.columns = ['Slot','AUROC','POD','FAR','CSI','HSS','Threshold']
                st.dataframe(display_metrics.style.background_gradient(
                    subset=['AUROC','HSS'], cmap='Greens').background_gradient(
                    subset=['FAR'], cmap='Reds_r'), use_container_width=True)

            with col2:
                st.markdown("#### vs Daily Baseline")
                comparison = pd.DataFrame([
                    {'Model':'Daily XGBoost (baseline)','AUROC':0.8715,'POD':0.500,'FAR':0.586,'HSS':0.389},
                    {'Model':'Ensemble (A9)',            'AUROC':0.8456,'POD':0.385,'FAR':0.538,'HSS':0.365},
                    {'Model':'Slot 2 v3 calibrated',    'AUROC':0.833, 'POD':0.356,'FAR':0.623,'HSS':0.318},
                ])
                fig = px.bar(comparison, x='Model', y=['AUROC','POD','HSS'],
                             barmode='group', color_discrete_sequence=['#2C3E50','#27AE60','#E67E22'])
                fig.update_layout(height=300, margin=dict(t=20,b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Forecast log
        forecast_log = load_forecast_log()
        if len(forecast_log) > 0:
            st.markdown("#### Recent Forecast Log")
            st.dataframe(forecast_log.tail(20), use_container_width=True)

    # ── TAB 3: SHAP ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### SHAP Feature Importance — v3 Models")

        if len(shap_df) > 0:
            slot_sel = st.selectbox("Select slot", [0,1,2,3],
                                    format_func=lambda x: f"Slot {x} — {SLOT_NAMES[x]}")
            slot_shap = shap_df[shap_df['slot']==slot_sel].head(15)

            NEW_FEATURES = ['cape_x_kindex','li_x_totals','q_gradient_500_850',
                            'thetae_850','wind_shear_700_850','wind_shear_500_850',
                            'moisture_flux_850','moisture_flux_700','mid_level_drying',
                            'thickness_500_850']
            colors = ['#E74C3C' if f in NEW_FEATURES else SLOT_COLORS[slot_sel]
                      for f in slot_shap['feature'].values[::-1]]

            fig = go.Figure(go.Bar(
                x=slot_shap['mean_shap'].values[::-1],
                y=slot_shap['feature'].values[::-1],
                orientation='h',
                marker_color=colors,
                marker_opacity=0.85,
            ))
            fig.update_layout(
                title=f"Top 15 Features — Slot {slot_sel} ({SLOT_NAMES[slot_sel]})<br>"
                      f"<span style='font-size:12px'>Red bars = new derived features (v3)</span>",
                height=450, margin=dict(t=60,b=20,l=20,r=20),
                xaxis_title="Mean |SHAP value|",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,1)',
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("🔴 Red bars are new derived features added in v3 "
                    "(interaction terms, moisture flux, θe). "
                    "These directly target the pre-monsoon convective burst regime.")
        else:
            st.info("SHAP importance data not found. Run A4_shap_analysis_v3.py first.")

    # ── TAB 4: SYNOPTIC REGIMES ───────────────────────────────────────────────
    with tab4:
        st.markdown("### Synoptic Weather Regime Analysis")
        st.markdown("KMeans clustering of 3,819 days into 5 atmospheric regimes (ERA5 features)")

        if len(synoptic_df) > 0:
            col1, col2 = st.columns([3,2])

            with col1:
                fig = px.bar(
                    synoptic_df.sort_values('ts_rate', ascending=True),
                    x='ts_rate', y='regime_name',
                    orientation='h',
                    color='AUROC',
                    color_continuous_scale='RdYlGn',
                    title='Thunderstorm Frequency & Model Skill per Regime',
                    labels={'ts_rate':'TS Rate (%)','regime_name':'Regime',
                            'AUROC':'Model AUROC'},
                    text='ts_rate',
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                fig.update_layout(height=350, margin=dict(t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Key Findings")
                for _, row in synoptic_df.sort_values('ts_rate',ascending=False).iterrows():
                    auroc = row.get('AUROC', 0)
                    ts    = row['ts_rate']
                    color = "🔴" if ts > 30 else "🟠" if ts > 15 else "🟡" if ts > 5 else "🟢"
                    st.markdown(f"{color} **{row['regime_name']}**")
                    st.markdown(f"   TS rate: **{ts:.1f}%** | AUROC: **{auroc:.3f}**")

                st.info("**Key insight:** The pre-monsoon convective burst regime "
                        "(52% TS rate) is where the model struggles most (AUROC 0.773). "
                        "October post-monsoon storms are the primary miss pattern.")
        else:
            st.info("Synoptic regime data not found. Run A11_synoptic_clustering.py first.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.8rem'>"
        "CSIR Thunderstorm Prediction System | IMD Station 43295 Bengaluru Airport | "
        "Collaboration with Dr. Geeta Agnihotri, Scientist F, IMD Bengaluru"
        "</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
