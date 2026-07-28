"""
forecast_action.py
==================
Runs inside GitHub Actions to generate forecast.json
Called by .github/workflows/forecast_update.yml
"""

import pandas as pd
import numpy as np
import joblib
import json
import math
from pathlib import Path
from datetime import datetime

BASE   = Path('.')
MODELS = BASE / 'models'
DATA   = BASE / 'data'

SLOT_NAMES  = {0:"0001-0600 IST",1:"0601-1200 IST",2:"1201-1800 IST",3:"1801-2400 IST"}
SLOT_LABELS = {0:"Late Night",1:"Morning",2:"Afternoon",3:"Evening"}
THRESHOLDS  = {0:0.24, 1:0.38, 2:0.16, 3:0.39}

def apply_calibrator(artifact, raw_prob):
    cal = artifact.get('calibrator')
    if cal is None:
        return raw_prob
    if artifact.get('calib_method') == 'sigmoid':
        return float(cal.predict_proba(np.array([[raw_prob]]))[0][1])
    return float(cal.predict([raw_prob])[0])

def compute_derived(obs, slot_id):
    m   = int(obs.get('month', datetime.now().month))
    doy = int(obs.get('doy', datetime.now().timetuple().tm_yday))
    obs['DTR']        = obs.get('MAX', 30) - obs.get('MIN', 20)
    obs['HA_flag']    = 0
    obs['RF_nonzero'] = 1 if obs.get('RF', 0) > 0 else 0
    if m in [3,4,5]:     obs['SEASON'] = 1
    elif m in [6,7,8,9]: obs['SEASON'] = 2
    elif m in [10,11]:   obs['SEASON'] = 3
    else:                 obs['SEASON'] = 0
    obs['MONTH_sin']  = math.sin(2*math.pi*m/12)
    obs['MONTH_cos']  = math.cos(2*math.pi*m/12)
    obs['DOY_sin']    = math.sin(2*math.pi*doy/365)
    obs['DOY_cos']    = math.cos(2*math.pi*doy/365)
    obs['doy_sin']    = obs['DOY_sin']
    obs['doy_cos']    = obs['DOY_cos']
    obs['slot_sin']   = math.sin(2*math.pi*slot_id/4)
    obs['slot_cos']   = math.cos(2*math.pi*slot_id/4)
    CLIM = {
        (4,2):0.129,(5,2):0.194,(6,2):0.096,(7,2):0.032,
        (8,2):0.052,(9,2):0.079,(10,2):0.077,
    }
    obs['slot_month_clim'] = CLIM.get((m, slot_id), 0.02)
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

    # v5 temporal features — atmospheric lags and rolling stats
    # These use met_history loaded from forecast_log.csv
    cape_now  = obs.get('ERA5_CAPE', obs.get('CAPE', 0.0))
    ki_now    = obs.get('K_INDEX', 30.0)
    pwat_now  = obs.get('PRECIP_WATER', 40.0)
    shear_now = obs.get('shear_850_500',
                        ((obs.get('ERA5_u_500hPa',5)-obs.get('ERA5_u_850hPa',-3))**2 +
                         (obs.get('ERA5_v_500hPa',2)-obs.get('ERA5_v_850hPa',2))**2)**0.5)

    for w in [3, 7, 14]:
        obs[f'CAPE_roll{w}']  = get_roll('ERA5_CAPE', w, cape_now)
        obs[f'KI_roll{w}']    = get_roll('K_INDEX',   w, ki_now)
        obs[f'PWAT_roll{w}']  = get_roll('PRECIP_WATER', w, pwat_now)
        obs[f'shear_roll{w}'] = get_roll('shear_850_500', w, shear_now)
    for lag in [1, 2, 3]:
        obs[f'CAPE_lag{lag}']  = get_lag('ERA5_CAPE',    lag, cape_now)
        obs[f'KI_lag{lag}']    = get_lag('K_INDEX',      lag, ki_now)
        obs[f'LI_lag{lag}']    = get_lag('LIFTED_INDEX', lag, obs.get('LIFTED_INDEX',-2.0))
        obs[f'PWAT_lag{lag}']  = get_lag('PRECIP_WATER', lag, pwat_now)
        obs[f'shear_lag{lag}'] = get_lag('shear_850_500', lag, shear_now)
    obs['CAPE_trend3'] = cape_now - obs.get('CAPE_roll3', cape_now)
    obs['KI_trend3']   = ki_now   - obs.get('KI_roll3',   ki_now)
    obs['PWAT_trend3'] = pwat_now - obs.get('PWAT_roll3',  pwat_now)
    return obs

now      = datetime.now()
month    = now.month
doy      = now.timetuple().tm_yday
date_str = now.strftime('%Y-%m-%d')

# Load upper air data if available
upper_air = {}
ua_path   = DATA / 'upperair_realtime_43295.csv'
if ua_path.exists():
    ua_df   = pd.read_csv(ua_path)
    ua_today = ua_df[ua_df['date'] == date_str]
    for _, row in ua_today.iterrows():
        upper_air[int(row['slot'])] = row.to_dict()
    print(f"Upper air loaded: {list(upper_air.keys())}")

# Load rolling met history for temporal features (last 14 days from forecast_log)
met_history = {}  # {feature: [val_t-1, val_t-2, val_t-3]} oldest first
flog_path = DATA / 'forecast_log.csv'
if flog_path.exists():
    try:
        flog = pd.read_csv(flog_path)
        flog['date'] = pd.to_datetime(flog['date'])
        # Get last 14 days of Slot 2 (most complete met data)
        recent = flog[flog['slot']==2].sort_values('date').tail(14)
        for feat in ['ERA5_CAPE','K_INDEX','LIFTED_INDEX','PRECIP_WATER']:
            if feat in recent.columns:
                vals = recent[feat].fillna(0).tolist()
                met_history[feat] = vals  # oldest to newest
    except Exception as _e:
        pass

def get_lag(feat, lag, default=0.0):
    vals = met_history.get(feat, [])
    if len(vals) >= lag:
        return float(vals[-lag])
    return default

def get_roll(feat, window, default=0.0):
    vals = met_history.get(feat, [])
    if len(vals) >= 1:
        return float(np.mean(vals[-window:]))
    return default

# ── Monsoon phase detection ──────────────────────────────────────────────────
monsoon_phase    = 'NEUTRAL'
monsoon_index    = 0.0
phase_detector   = {}
phase_det_path   = MODELS / 'monsoon_phase_detector.json'
if phase_det_path.exists():
    try:
        with open(phase_det_path) as _f:
            phase_detector = json.load(_f)
        # Compute monsoon index from latest GFS/upper-air values
        # Will be updated after GFS loads — placeholder here
        monsoon_phase = 'NEUTRAL'
    except Exception as _e:
        pass

# Load GFS data if available
gfs_df   = pd.DataFrame()
gfs_path = DATA / 'gfs_realtime_43295.csv'
if gfs_path.exists():
    gfs_df = pd.read_csv(gfs_path)
    print(f"GFS loaded: {len(gfs_df)} rows, cycle: {gfs_df.get('gfs_cycle', ['unknown'])[0] if len(gfs_df) else 'N/A'}")

slots_output = []
results      = {}

for slot_id in range(4):
    # Best model per slot (empirically validated on 2024-2025 test set):
    # Slot 0: v4_ensemble  AUROC=0.8484
    # Slot 1: v5_temporal  AUROC=0.8317
    # Slot 2: v3_calibrated AUROC=0.8710 (best)
    # Slot 3: v3_calibrated AUROC=0.8710 (best)
    v4_path = MODELS / f'nowcast_slot{slot_id}_xgb_v4_ensemble.pkl'
    v5_path = MODELS / f'nowcast_slot{slot_id}_xgb_v5_temporal.pkl'
    v3_path = MODELS / f'nowcast_slot{slot_id}_xgb_v3_calibrated.pkl'

    # Slot 1 uses v5; Slots 2/3 use v3; Slot 0 uses v4
    # IMPORTANT: store full artifact dict in artifact_model, not just calibrated model
    if slot_id == 1 and v5_path.exists():
        artifact_model = joblib.load(v5_path)  # full dict with 'calibrated'+'features'
        model_path = None
        auroc_str = f"{artifact_model.get('auroc',0):.4f}" if isinstance(artifact_model,dict) else '?'
        print(f'  [v5 temporal] Slot {slot_id} | AUROC={auroc_str} | features={len(artifact_model.get("features",[]))}')
    elif slot_id in [2, 3] and v3_path.exists():
        model_path     = v3_path
        artifact_model = None
        print(f'  [v3 calibrated] Slot {slot_id} | AUROC=0.8710')
    elif v4_path.exists():
        artifact_model = joblib.load(v4_path)  # full dict with 'calibrated'+'features'
        model_path = None
        auroc_str = f"{artifact_model.get('auroc',0):.4f}" if isinstance(artifact_model,dict) else '?'
        print(f'  [v4 ensemble] Slot {slot_id} | AUROC={auroc_str}')
    else:
        model_path     = v3_path
        artifact_model = None
        if not model_path.exists():
            model_path = MODELS / f'nowcast_slot{slot_id}_xgb_v2_calibrated.pkl'
        print(f'  [v3 fallback] Slot {slot_id}')

    if model_path is not None and not model_path.exists():
        clim = {0:0.037, 1:0.011, 2:0.063, 3:0.059}
        prob = clim[slot_id]
        results[slot_id] = prob
        slots_output.append({
            "slot": slot_id, "label": SLOT_LABELS[slot_id],
            "time": SLOT_NAMES[slot_id],
            "ts_probability": round(prob, 4),
            "ts_predicted": prob >= THRESHOLDS[slot_id],
            "threshold": THRESHOLDS[slot_id],
            "primary": slot_id == 2,
            "source": "climatology",
        })
        continue

    artifact     = artifact_model if artifact_model is not None else joblib.load(model_path)
    # Artifact formats:
    # v3: dict with keys 'model', 'feature_cols', 'threshold'
    # v4/v5: dict with keys 'calibrated', 'features', 'auroc'  OR CalibratedClassifierCV directly
    if isinstance(artifact, dict) and 'model' in artifact:
        # v3 format
        model        = artifact['model']
        feature_cols = artifact['feature_cols']
        threshold    = artifact.get('threshold', THRESHOLDS[slot_id])
    elif isinstance(artifact, dict) and 'calibrated' in artifact:
        # v4/v5 format
        model        = artifact['calibrated']
        feature_cols = artifact.get('features', None)
        threshold    = THRESHOLDS[slot_id]
    else:
        # CalibratedClassifierCV directly (old v4 without dict wrapper)
        model        = artifact
        feature_cols = None
        threshold    = THRESHOLDS[slot_id]

    obs = {
        'month':month,'doy':doy,
        'MAX':30.0,'MIN':20.0,'RF':0.0,'AW':3.0,'EVP':5.0,'DRNRF':0.0,'SSH':300.0,
        'RF_3d':0.0,'RF_7d':0.0,'MAX_3d_avg':30.0,'MIN_3d_avg':20.0,'DTR_3d_avg':10.0,
        'RF_lag1':0.0,'MAX_lag1':30.0,'MIN_lag1':20.0,'LABEL_lag1':0,
        'CAPE':0.0,'CIN':0.0,'K_INDEX':30.0,'LIFTED_INDEX':-2.0,
        'TOTALS_TOTALS':44.0,'PRECIP_WATER':40.0,
        'ERA5_T2M':299.0,'ERA5_D2M':293.0,'ERA5_U10':-2.0,'ERA5_V10':1.0,
        'ERA5_CAPE':0.0,'ERA5_SP':91500.0,
        'ERA5_t_500hPa':268.0,'ERA5_t_700hPa':283.0,'ERA5_t_850hPa':293.0,
        'ERA5_q_500hPa':0.003,'ERA5_q_700hPa':0.009,'ERA5_q_850hPa':0.013,
        'ERA5_u_500hPa':5.0,'ERA5_u_700hPa':2.0,'ERA5_u_850hPa':-3.0,
        'ERA5_v_500hPa':2.0,'ERA5_v_700hPa':1.0,'ERA5_v_850hPa':2.0,
        'ts_label_lag1_slot':0,'ts_any_yesterday':0,
    }

    if len(gfs_df) > 0:
        row = gfs_df.iloc[0]
        for col in ['ERA5_T2M','ERA5_D2M','ERA5_U10','ERA5_V10','ERA5_CAPE','ERA5_SP',
                    'ERA5_t_500hPa','ERA5_t_700hPa','ERA5_t_850hPa',
                    'ERA5_q_500hPa','ERA5_q_700hPa','ERA5_q_850hPa',
                    'ERA5_u_500hPa','ERA5_u_700hPa','ERA5_u_850hPa',
                    'ERA5_v_500hPa','ERA5_v_700hPa','ERA5_v_850hPa']:
            if col in row.index and pd.notna(row[col]):
                obs[col] = float(row[col])
        for col in ['K_INDEX','TOTALS_TOTALS','LIFTED_INDEX','CAPE','PRECIP_WATER']:
            if col in row.index and pd.notna(row[col]):
                obs[col] = float(row[col])

    # ── Update monsoon phase using actual GFS values ─────────────────────────
    if phase_detector and len(gfs_df) > 0:
        try:
            _row    = gfs_df.iloc[0]
            _pwat   = float(_row.get('PRECIP_WATER', phase_detector.get('pwat_mean', 40)))
            _ki     = float(_row.get('K_INDEX', phase_detector.get('ki_mean', 36)))
            _q500   = float(_row.get('ERA5_q_500hPa', phase_detector.get('q500_mean', 0.003)))
            _mi = (0.40 * (_pwat - phase_detector['pwat_mean']) / (phase_detector['pwat_std'] + 1e-9) +
                   0.30 * (_ki   - phase_detector['ki_mean'])   / (phase_detector['ki_std']  + 1e-9) +
                   0.30 * (_q500 - phase_detector['q500_mean']) / (phase_detector['q500_std'] + 1e-9))
            monsoon_index = round(float(_mi), 3)
            monsoon_phase = ('ACTIVE'  if _mi > phase_detector['active_threshold']
                            else 'BREAK' if _mi < phase_detector['break_threshold']
                            else 'NEUTRAL')
            print(f"  [MONSOON] Phase={monsoon_phase} Index={monsoon_index:.2f} "
                  f"PWAT={_pwat:.1f} KI={_ki:.1f}")
        except Exception as _e:
            print(f"  [MONSOON] Phase detection error: {_e}")

    if slot_id in upper_air:
        ua = upper_air[slot_id]
        for col in ['CAPE','CIN','K_INDEX','LIFTED_INDEX','TOTALS_TOTALS','PRECIP_WATER',
                    'ERA5_u_500hPa','ERA5_v_500hPa','ERA5_u_850hPa','ERA5_v_850hPa']:
            if col in ua and pd.notna(ua[col]):
                obs[col] = float(ua[col])

    obs  = compute_derived(obs, slot_id)
    # v4 ensemble: feature_cols is None — use model's own feature names if available
    if feature_cols is None:
        # v4 ensemble trained on numpy arrays — no feature names stored
        # Use the v4 feature list (same 54 features used in Cell 2 training)
        fc = [
            'ERA5_CAPE','ERA5_T2M','ERA5_D2M','ERA5_U10','ERA5_V10','ERA5_SP',
            'ERA5_t_500hPa','ERA5_t_700hPa','ERA5_t_850hPa',
            'ERA5_q_500hPa','ERA5_q_700hPa','ERA5_q_850hPa',
            'ERA5_u_500hPa','ERA5_u_700hPa','ERA5_u_850hPa',
            'ERA5_v_500hPa','ERA5_v_700hPa','ERA5_v_850hPa',
            'K_INDEX','LIFTED_INDEX','TOTALS_TOTALS','PRECIP_WATER','CIN',
            'MAX','MIN','DTR','RF','EVP','SSH','DRNRF',
            'RF_3d','RF_7d','MAX_3d_avg','MIN_3d_avg','RF_lag1','MAX_lag1','MIN_lag1','LABEL_lag1',
            'MONTH_sin','MONTH_cos','DOY_sin','DOY_cos','SEASON',
            'cold_top_proxy','mid_moisture','lapse_rate_850_500','shear_850_500',
            'cape_x_ki','cape_x_tt','rh_proxy','wind_speed_850','wind_speed_500',
            'theta_e_proxy','moisture_flux_850',
        ]
    else:
        fc = feature_cols
    X    = np.array([[float(obs.get(c, 0.0)) for c in fc]])
    # v3: dict with 'model' key — needs apply_calibrator
    # v4/v5: dict with 'calibrated' key OR direct CalibratedClassifierCV — already calibrated
    if isinstance(artifact, dict) and 'model' in artifact:
        raw = float(model.predict_proba(X)[0][1])
        cal = apply_calibrator(artifact, raw)
    else:
        cal = float(model.predict_proba(X)[0][1])
    results[slot_id] = cal

    slots_output.append({
        "slot": slot_id, "label": SLOT_LABELS[slot_id],
        "time": SLOT_NAMES[slot_id],
        "ts_probability": round(float(cal), 4),
        "ts_predicted": float(cal) >= threshold,
        "threshold": threshold,
        "primary": slot_id == 2,
        "source": "gfs+upperair" if slot_id in upper_air else "model_defaults",
        "cape": round(obs.get('CAPE', 0), 1),
        "k_index": round(obs.get('K_INDEX', 0), 1),
        "lifted_index": round(obs.get('LIFTED_INDEX', 0), 2),
        "totals_totals": round(obs.get('TOTALS_TOTALS', 0), 1),
    })
    print(f"Slot {slot_id}: {cal*100:.1f}%")

alert_active     = any(s["ts_predicted"] for s in slots_output)
peak_slot        = max(results, key=results.get)
peak_probability = results[peak_slot]
met_slot         = next((s for s in slots_output if s["slot"] == 2), slots_output[0])

# ── Pull wind values for met_parameters ──────────────────────────────────────
# Try Slot 2 first (afternoon peak), fall back through 3→1→0 so real values
# are used whenever any slot has been fetched today (Aprameya 2026-07-26).
_ua_wind = (upper_air.get(2) or upper_air.get(3) or
            upper_air.get(1) or upper_air.get(0) or {})

def _wind(key):
    v = _ua_wind.get(key)
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0

himawari_override_active = False
himawari_override_slots  = []
himawari_boost_value     = 0.0

forecast = {
    "date":             date_str,
    "generated_at":     now.strftime('%Y-%m-%d %H:%M IST'),
    "alert_active":     alert_active,
    "peak_slot":        peak_slot,
    "peak_probability": round(float(peak_probability), 4),
    "model_version":    "v4_ensemble_A100",
    "himawari_override_active": himawari_override_active,
    "himawari_override_slots":  himawari_override_slots,
    "himawari_boost_value":     himawari_boost_value,
    "monsoon_phase":            monsoon_phase,
    "monsoon_index":            monsoon_index,
    "slots":            slots_output,
    "met_parameters": {
        "ua_cape_jkg":       met_slot.get('cape', 0),
        "ua_cape_raw":       met_slot.get('cape', 0),
        "ua_k_index":        met_slot.get('k_index', 0),
        "ua_lifted_index":   met_slot.get('lifted_index', 0),
        "ua_totals_totals":  met_slot.get('totals_totals', 0),
        # GFS wind components at 500/850 hPa (m/s) — named ERA5_* to match training schema
        "ERA5_u_500hPa":     _wind('ERA5_u_500hPa'),
        "ERA5_v_500hPa":     _wind('ERA5_v_500hPa'),
        "ERA5_u_850hPa":     _wind('ERA5_u_850hPa'),
        "ERA5_v_850hPa":     _wind('ERA5_v_850hPa'),
        "instability_level": (
            "Extreme" if met_slot.get('cape', 0) >= 3000 else
            "High"    if met_slot.get('cape', 0) >= 1500 else
            "Moderate" if met_slot.get('cape', 0) >= 500 else
            "Marginal" if met_slot.get('cape', 0) >= 100 else
            "Stable"
        ),
    },
}

# Load Himawari satellite signal
himawari = {}
himawari_history = []
himawari_path = DATA / 'himawari_realtime.json'
himawari_hist_path = DATA / 'himawari_history.json'

if himawari_path.exists():
    try:
        with open(himawari_path) as f:
            himawari = json.load(f)
        print(f"Himawari loaded: storm_detected={himawari.get('storm_detected')} min_bt={himawari.get('min_bt_50km')}°C")
    except Exception as e:
        print(f"Himawari load error: {e}")

if himawari_hist_path.exists():
    try:
        with open(himawari_hist_path) as f:
            himawari_history = json.load(f)
    except Exception:
        pass

# ── HIMAWARI CORRECTION MODEL (TRAINED, CV AUROC=0.8812) ─────────────────────
# Replaces heuristic boost with calibrated logistic correction model
# Trained on 2015-2025 ERA5+IMD data. ERA5 proxies for Himawari BT.
CORRECTION_MODEL_PATH = BASE / 'models' / 'himawari_correction_model.pkl'
CORRECTION_META_PATH  = BASE / 'models' / 'correction_model_meta.json'
HIMAWARI_BT_THRESHOLD = -45.0
HIMAWARI_PIXEL_MIN    = 50
HIMAWARI_DIST_MAX_KM  = 100.0
CORRECTION_MIN_UPLIFT = 0.05    # only override if correction gives ≥5% uplift

# Load trained correction model
_correction_model = None
_t500_clim_mean   = 268.80
_correction_auroc = 0.0
try:
    _correction_model = joblib.load(CORRECTION_MODEL_PATH)
    with open(CORRECTION_META_PATH) as _f:
        _cmeta = json.load(_f)
    _t500_clim_mean   = _cmeta.get('t500_climatological_mean', 268.80)
    _correction_auroc = _cmeta.get('cv_auroc_mean', 0.0)
    print(f"[CORRECTION MODEL] Loaded — CV AUROC={_correction_auroc:.4f}")
except Exception as _e:
    print(f"[CORRECTION MODEL] Not found ({_e}) — using heuristic fallback")

def _build_correction_features(ua_row):
    era5_cape  = float(ua_row.get('ERA5_CAPE', ua_row.get('CAPE', 0)) or 0)
    era5_t500  = float(ua_row.get('ERA5_t_500hPa', _t500_clim_mean) or _t500_clim_mean)
    era5_t850  = float(ua_row.get('ERA5_t_850hPa', 294.0) or 294.0)
    era5_q700  = float(ua_row.get('ERA5_q_700hPa', 0.008) or 0.008)
    era5_u500  = float(ua_row.get('ERA5_u_500hPa', 0) or 0)
    era5_v500  = float(ua_row.get('ERA5_v_500hPa', 0) or 0)
    era5_u850  = float(ua_row.get('ERA5_u_850hPa', 0) or 0)
    era5_v850  = float(ua_row.get('ERA5_v_850hPa', 0) or 0)
    k_idx      = float(ua_row.get('K_INDEX', 30) or 30)
    li         = float(ua_row.get('LIFTED_INDEX', -2) or -2)
    tt         = float(ua_row.get('TOTALS_TOTALS', 42) or 42)
    pwat       = float(ua_row.get('PRECIP_WATER', 40) or 40)
    m          = now.month
    month_sin  = math.sin(2 * math.pi * m / 12)
    month_cos  = math.cos(2 * math.pi * m / 12)
    cold_top   = _t500_clim_mean - era5_t500
    mid_moist  = era5_q700 * 1000
    lapse      = era5_t850 - era5_t500
    shear      = math.sqrt((era5_u500-era5_u850)**2 + (era5_v500-era5_v850)**2)
    cape_x_ki  = era5_cape * k_idx
    return [era5_cape, cold_top, mid_moist, lapse, shear,
            k_idx, li, tt, pwat, cape_x_ki, month_sin, month_cos]

def _heuristic_boost(bt, cpx, dist):
    bt_f = min(max((abs(bt)-45)/25, 0), 1.0)
    px_f = min(max((cpx-50)/250, 0), 1.0)
    d_f  = max(1.0 - dist/100.0, 0)
    return round(min(0.35 + 0.40*(bt_f*0.5 + px_f*0.3 + d_f*0.2), 0.75), 3)

if himawari:
    h_storm = himawari.get("storm_detected", False)
    h_bt    = himawari.get("min_bt_50km") or 0
    h_cpx   = himawari.get("cold_pixels_count") or 0
    h_dist  = himawari.get("nearest_pixel_dist_km") or 999

    if (h_storm and h_bt < HIMAWARI_BT_THRESHOLD and
            h_cpx >= HIMAWARI_PIXEL_MIN and h_dist < HIMAWARI_DIST_MAX_KM):

        himawari_override_active = True

        # Get best available upper-air row for feature extraction
        _ua = upper_air.get(2) or upper_air.get(3) or upper_air.get(1) or upper_air.get(0) or {}

        # Compute corrected probability
        if _correction_model is not None:
            try:
                _feat = _build_correction_features(_ua)
                _corrected = float(_correction_model.predict_proba([_feat])[0][1])
                _method = f"trained_model(AUROC={_correction_auroc:.3f})"
            except Exception as _ex:
                _corrected = _heuristic_boost(h_bt, h_cpx, h_dist)
                _method = f"heuristic_fallback({_ex})"
        else:
            _corrected = _heuristic_boost(h_bt, h_cpx, h_dist)
            _method = "heuristic_boost"

        ist_hour = now.hour
        if   0  <= ist_hour < 6:  affected = [3]
        elif 6  <= ist_hour < 12: affected = [0, 1]
        elif 12 <= ist_hour < 18: affected = [2]
        else:                     affected = [2, 3]

        for s in slots_output:
            if s["slot"] in affected:
                original_prob = s["ts_probability"]
                boosted_prob  = round(min(max(original_prob, _corrected), 0.95), 4)
                if boosted_prob - original_prob >= CORRECTION_MIN_UPLIFT:
                    s["ts_probability"]    = boosted_prob
                    s["ts_predicted"]      = boosted_prob >= s["threshold"]
                    s["himawari_override"] = True
                    s["himawari_boost"]    = boosted_prob
                    s["original_prob"]     = original_prob
                    s["correction_method"] = _method
                    himawari_override_slots.append(s["slot"])
                    results[s["slot"]]     = boosted_prob
                    himawari_boost_value   = boosted_prob
                    print(f"  [HIMAWARI CORRECTION] Slot {s['slot']}: {original_prob*100:.1f}% -> {boosted_prob*100:.1f}%"
                          f" | {_method} | BT={h_bt:.1f}C px={h_cpx} dist={h_dist:.1f}km")
                else:
                    print(f"  [HIMAWARI] Slot {s['slot']}: correction={_corrected*100:.1f}% no uplift over base={original_prob*100:.1f}%")

        if himawari_override_slots:
            print(f"[HIMAWARI] Correction active — slots {himawari_override_slots} | method={_method}")
        else:
            print(f"[HIMAWARI] Storm detected but no slots needed correction")
    else:
        print(f"[HIMAWARI] No correction — storm={h_storm} bt={h_bt:.1f}C px={h_cpx} dist={h_dist:.1f}km")

    # Update alert and peak after correction
    alert_active     = any(s["ts_predicted"] for s in slots_output)
    peak_slot        = max(results, key=results.get)
    peak_probability = results[peak_slot]
    # Patch forecast dict with post-override values
    forecast["alert_active"]              = alert_active
    forecast["peak_slot"]                 = peak_slot
    forecast["peak_probability"]          = round(float(peak_probability), 4)
    forecast["himawari_override_active"]  = himawari_override_active
    forecast["himawari_override_slots"]   = himawari_override_slots
    forecast["himawari_boost_value"]      = himawari_boost_value

forecast["satellite"] = {
    "himawari9": {
        "timestamp_utc":         himawari.get("timestamp_utc"),
        "timestamp_ist":         himawari.get("timestamp_ist"),
        "vobl_bt_celsius":       himawari.get("vobl_bt_celsius"),
        "min_bt_50km":           himawari.get("min_bt_50km"),
        "mean_bt_50km":          himawari.get("mean_bt_50km"),
        "cold_pixels_count":     himawari.get("cold_pixels_count", 0),
        "storm_detected":        himawari.get("storm_detected", False),
        "nearest_pixel_dist_km": himawari.get("nearest_pixel_dist_km"),
        "threshold_celsius":     himawari.get("threshold_celsius", -40.0),
        "data_source":           "Himawari-9 Band 13 (10.4um) via NOAA AWS S3",
        "available":             bool(himawari),
    },
    "history": himawari_history[-6:] if himawari_history else [],
}

# Load verification report if available
verification = {}
verif_path = BASE / 'results' / 'verification_report.json'
if verif_path.exists():
    try:
        with open(verif_path) as f:
            verification = json.load(f)
        print(f"Verification loaded: POD={verification.get('pod')} HSS={verification.get('hss')}")
    except Exception as e:
        print(f"Verification load error: {e}")

# Convective Initiation Timer
import math as _math
now_hour_ist = now.hour + now.minute / 60.0

gfs_row = gfs_df.iloc[0] if len(gfs_df) > 0 else {}
cape_now = float(gfs_row.get('CAPE', 0)) if len(gfs_df) > 0 else 0
ki_now   = float(gfs_row.get('K_INDEX', 30)) if len(gfs_df) > 0 else 30
li_now   = float(gfs_row.get('LIFTED_INDEX', 0)) if len(gfs_df) > 0 else 0
tt_now   = float(gfs_row.get('TOTALS_TOTALS', 44)) if len(gfs_df) > 0 else 44

cape_score = min(cape_now / 2000.0 * 40, 40)
ki_score   = max(0, min((ki_now - 20) / 20.0 * 30, 30))
li_score   = max(0, min((-li_now) / 6.0 * 20, 20))
tt_score   = max(0, min((tt_now - 40) / 10.0 * 10, 10))
instability_score = round(cape_score + ki_score + li_score + tt_score, 1)

PEAK_START = 13.0
PEAK_END   = 18.0

if now_hour_ist < PEAK_START:
    hours_to_peak = PEAK_START - now_hour_ist
    initiation_status = "PRE-CONVECTIVE"
    initiation_message = f"Peak convective window in {hours_to_peak:.1f}h (1300-1800 IST)"
elif PEAK_START <= now_hour_ist <= PEAK_END:
    hours_to_peak = 0
    initiation_status = "CONVECTIVE WINDOW ACTIVE"
    initiation_message = "Currently in peak thunderstorm window (1300-1800 IST)"
else:
    hours_to_peak = 24 - now_hour_ist + PEAK_START
    initiation_status = "POST-CONVECTIVE"
    initiation_message = f"Next peak window in {hours_to_peak:.1f}h (tomorrow 1300 IST)"

if instability_score >= 70:
    initiation_risk = "HIGH"
elif instability_score >= 45:
    initiation_risk = "MODERATE"
elif instability_score >= 25:
    initiation_risk = "LOW"
else:
    initiation_risk = "MINIMAL"

forecast["convective_initiation"] = {
    "instability_score":  instability_score,
    "initiation_status":  initiation_status,
    "initiation_message": initiation_message,
    "initiation_risk":    initiation_risk,
    "hours_to_peak":      round(hours_to_peak, 1),
    "cape_now":           round(cape_now, 1),
    "ki_now":             round(ki_now, 2),
    "li_now":             round(li_now, 2),
    "tt_now":             round(tt_now, 2),
    "peak_window_ist":    "1300-1800 IST",
    "computed_at":        now.strftime('%Y-%m-%d %H:%M IST'),
}
print(f"Convective initiation: {initiation_status} | Score: {instability_score} | Risk: {initiation_risk}")

# Multi-day outlook
multiday_path = DATA / 'gfs_multiday_43295.json'
multiday_outlook = []

if multiday_path.exists():
    try:
        with open(multiday_path) as f:
            multiday_data = json.load(f)

        for day_row in multiday_data:
            day_cape = float(day_row.get('CAPE', 0))
            day_ki   = float(day_row.get('K_INDEX', 30))
            day_li   = float(day_row.get('LIFTED_INDEX', 0))
            day_tt   = float(day_row.get('TOTALS_TOTALS', 44))

            d_cape_score = min(day_cape / 2000.0 * 40, 40)
            d_ki_score   = max(0, min((day_ki - 20) / 20.0 * 30, 30))
            d_li_score   = max(0, min((-day_li) / 6.0 * 20, 20))
            d_tt_score   = max(0, min((day_tt - 40) / 10.0 * 10, 10))
            d_score      = round(d_cape_score + d_ki_score + d_li_score + d_tt_score, 1)
            d_prob       = min(round(d_score / 100.0 * 0.6, 3), 0.95)
            risk = "HIGH" if d_score >= 70 else "MODERATE" if d_score >= 45 else "LOW" if d_score >= 25 else "MINIMAL"

            multiday_outlook.append({
                "date":              day_row.get('date'),
                "day_label":         day_row.get('day_label'),
                "cape":              round(day_cape, 1),
                "k_index":           round(day_ki, 2),
                "lifted_index":      round(day_li, 2),
                "totals_totals":     round(day_tt, 2),
                "instability_score": d_score,
                "ts_probability_slot2": d_prob,
                "risk_level":        risk,
                "peak_window":       "1300-1800 IST",
            })
            print(f"  {day_row.get('day_label')}: score={d_score} prob={d_prob*100:.1f}% risk={risk}")

    except Exception as e:
        print(f"  Multiday outlook error: {e}")

forecast["multiday_outlook"] = multiday_outlook

# Historical analogs
analogs = []
try:
    features_path = BASE / 'data' / 'bengaluru_thunderstorm_features_merged.csv'
    if not features_path.exists():
        features_path = BASE / 'bengaluru_thunderstorm_features_merged.csv'
    if features_path.exists():
        import pandas as pd_ana
        df_ana = pd_ana.read_csv(features_path, parse_dates=['date'])

        today_cape  = cape_now
        today_ki    = ki_now
        today_li    = li_now
        today_month = month

        months = [(today_month - 1) % 12 or 12, today_month, (today_month % 12) + 1]
        df_filtered = df_ana[df_ana['date'].dt.month.isin(months)].copy()

        if 'CAPE' in df_filtered.columns and 'K_INDEX' in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=['CAPE', 'K_INDEX'])
            cape_rng = df_filtered['CAPE'].max() - df_filtered['CAPE'].min() + 1e-9
            ki_rng   = df_filtered['K_INDEX'].max() - df_filtered['K_INDEX'].min() + 1e-9
            li_rng   = df_filtered['LIFTED_INDEX'].max() - df_filtered['LIFTED_INDEX'].min() + 1e-9 if 'LIFTED_INDEX' in df_filtered.columns else 1

            df_filtered['_score'] = (
                2.0 * (df_filtered['CAPE'] - today_cape).abs() / cape_rng +
                1.5 * (df_filtered['K_INDEX'] - today_ki).abs() / ki_rng +
                1.0 * (df_filtered['LIFTED_INDEX'] - today_li).abs() / li_rng if 'LIFTED_INDEX' in df_filtered.columns else 0
            )

            top_analogs = df_filtered.nsmallest(5, '_score')
            for _, row_a in top_analogs.iterrows():
                analogs.append({
                    'date':         str(row_a['date'])[:10],
                    'cape':         round(float(row_a.get('CAPE', 0)), 1),
                    'k_index':      round(float(row_a.get('K_INDEX', 0)), 1),
                    'lifted_index': round(float(row_a.get('LIFTED_INDEX', 0)), 2) if 'LIFTED_INDEX' in row_a.index else None,
                    'thunderstorm': bool(row_a.get('LABEL', 0)),
                    'month':        int(row_a['date'].month),
                })
            ts_count = sum(1 for a in analogs if a['thunderstorm'])
            print(f"Analogs found: {len(analogs)} days, {ts_count} with TS")
except Exception as e:
    print(f"Analog search error: {e}")

forecast["analogs"] = {
    "top_5":       analogs,
    "ts_rate":     round(sum(1 for a in analogs if a['thunderstorm']) / len(analogs), 2) if analogs else None,
    "query_cape":  round(cape_now, 1),
    "query_ki":    round(ki_now, 2),
    "query_li":    round(li_now, 2),
    "computed_at": now.strftime('%Y-%m-%d %H:%M IST'),
}

# Airport Impact Score
SLOT_DEPARTURES   = {0: 8, 1: 45, 2: 52, 3: 38}
DISRUPTION_FACTOR = 0.60

impact_slots     = []
total_disrupted  = 0
total_departures = 0

for s in slots_output:
    slot_id   = s['slot']
    prob      = s.get('ts_probability', 0) or 0
    deps      = SLOT_DEPARTURES.get(slot_id, 0)
    disrupted = round(prob * deps * DISRUPTION_FACTOR)
    total_disrupted  += disrupted
    total_departures += deps
    impact_slots.append({
        'slot':           slot_id,
        'label':          s.get('label', ''),
        'ts_probability': round(prob, 4),
        'departures':     deps,
        'disrupted_est':  disrupted,
        'impact_pct':     round(prob * DISRUPTION_FACTOR * 100, 1),
    })

overall_risk = 'HIGH' if total_disrupted >= 20 else 'MODERATE' if total_disrupted >= 8 else 'LOW' if total_disrupted >= 2 else 'MINIMAL'

forecast["airport_impact"] = {
    "total_departures_today": total_departures,
    "total_disrupted_est":    total_disrupted,
    "disruption_pct":         round(total_disrupted / total_departures * 100, 1) if total_departures else 0,
    "overall_risk":           overall_risk,
    "disruption_factor":      DISRUPTION_FACTOR,
    "slots":                  impact_slots,
    "computed_at":            now.strftime('%Y-%m-%d %H:%M IST'),
}
print(f"Airport impact: {total_disrupted} disrupted of {total_departures} departures ({overall_risk})")

# Synoptic Regime Auto-Detection
try:
    t2m_c = (float(gfs_row.get('ERA5_T2M', 302)) - 273.15) if len(gfs_df) > 0 else 28.0

    if ki_now >= 38 and cape_now >= 800:
        regime_id = 'R5'; regime_name = 'Pre-Monsoon Convective Burst'
        regime_ts_rate = 52.1; regime_auroc = 0.773; regime_color = 'red'
        regime_desc = 'Severe convective instability — most challenging and highest TS rate regime'
    elif ki_now >= 35 and cape_now >= 300 and month in [5,6,7,8,9]:
        regime_id = 'R2'; regime_name = 'Moist Monsoon'
        regime_ts_rate = 9.3; regime_auroc = 0.934; regime_color = 'yellow'
        regime_desc = 'Monsoonal westerly surge with high skill forecast'
    elif ki_now >= 32 and cape_now >= 100 and t2m_c >= 28:
        regime_id = 'R4'; regime_name = 'Strong Solar Heating'
        regime_ts_rate = 9.8; regime_auroc = 0.900; regime_color = 'orange'
        regime_desc = 'Strong surface heating with mid-level moisture'
    elif cape_now < 100 and ki_now < 30 and month in [6,7,8,9]:
        regime_id = 'R3'; regime_name = 'Break Monsoon'
        regime_ts_rate = 10.2; regime_auroc = 0.798; regime_color = 'blue'
        regime_desc = 'Break-monsoon stratiform clouding, suppressed convection'
    else:
        regime_id = 'R1'; regime_name = 'Hot Pre-Monsoon / Stable'
        regime_ts_rate = 1.2; regime_auroc = 1.000; regime_color = 'green'
        regime_desc = 'Dry thermal low baseline, low storm occurrence'

    forecast["synoptic_regime"] = {
        "regime_id":   regime_id,
        "regime_name": regime_name,
        "ts_rate":     regime_ts_rate,
        "auroc":       regime_auroc,
        "description": regime_desc,
        "color":       regime_color,
        "cape_used":   round(cape_now, 1),
        "ki_used":     round(ki_now, 2),
        "t2m_c":       round(t2m_c, 1),
        "month":       month,
        "computed_at": now.strftime('%Y-%m-%d %H:%M IST'),
    }
    print(f"Synoptic regime: {regime_id} — {regime_name} (TS rate: {regime_ts_rate}%)")
except Exception as e:
    print(f"Regime detection error: {e}")
    forecast["synoptic_regime"] = {}

# Verification metrics
slot2_30d = verification.get("metrics_30day", {}).get("2", {})
forecast["verification"] = {
    "pod":           round(float(slot2_30d.get("POD", 0)), 3) if slot2_30d else None,
    "far":           round(float(slot2_30d.get("FAR", 0)), 3) if slot2_30d else None,
    "hss":           round(float(slot2_30d.get("HSS", 0)), 3) if slot2_30d else None,
    "brier":         round(float(slot2_30d.get("Brier", 0)), 4) if slot2_30d else None,
    "csi":           round(float(slot2_30d.get("CSI", 0)), 3) if slot2_30d else None,
    "n_days":        slot2_30d.get("n_days"),
    "n_ts":          slot2_30d.get("n_ts"),
    "date_verified": verification.get("generated_at"),
    "slot":          2,
    "window":        "30-day",
    "available":     bool(slot2_30d),
    "all_slots_30d": {
        str(k): {
            "pod":   round(float(v.get("POD", 0)), 3),
            "far":   round(float(v.get("FAR", 0)), 3),
            "hss":   round(float(v.get("HSS", 0)), 3),
            "brier": round(float(v.get("Brier", 0)), 4),
        }
        for k, v in verification.get("metrics_30day", {}).items()
    }
}

# Trend comparison with previous forecast
prev_probs = {}
if Path('forecast.json').exists():
    try:
        with open('forecast.json') as f:
            prev = json.load(f)
        for s in prev.get('slots', []):
            prev_probs[s['slot']] = s.get('ts_probability', 0) or 0
    except Exception:
        pass

for s in forecast['slots']:
    slot_id = s['slot']
    curr    = s.get('ts_probability') or 0
    prev_p  = prev_probs.get(slot_id, curr)
    diff    = round(curr - prev_p, 3)
    s['trend']            = 'up' if diff > 0.01 else 'down' if diff < -0.01 else 'stable'
    s['trend_diff']       = diff
    s['prev_probability'] = round(prev_p, 3)

# Real-time SHAP
try:
    import subprocess
    subprocess.run(['python', 'compute_realtime_shap.py'], check=True, timeout=120)
    shap_path = DATA / 'realtime_shap.json'
    if shap_path.exists():
        with open(shap_path) as f:
            realtime_shap = json.load(f)
        forecast['realtime_shap'] = realtime_shap
        print(f"SHAP computed: {len(realtime_shap)} slots")
except Exception as e:
    print(f"SHAP computation error: {e}")

with open('forecast.json', 'w') as f:
    json.dump(forecast, f, indent=2)

print(f"forecast.json updated: alert={alert_active} peak=Slot{peak_slot} {peak_probability*100:.1f}%")
print(f"met_parameters wind keys: u500={forecast['met_parameters']['ERA5_u_500hPa']} v500={forecast['met_parameters']['ERA5_v_500hPa']} u850={forecast['met_parameters']['ERA5_u_850hPa']} v850={forecast['met_parameters']['ERA5_v_850hPa']}")