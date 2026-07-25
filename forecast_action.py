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

# Load GFS data if available
gfs_df   = pd.DataFrame()
gfs_path = DATA / 'gfs_realtime' / f"gfs_{date_str}.csv"
if gfs_path.exists():
    gfs_df = pd.read_csv(gfs_path)
    print(f"GFS loaded: {len(gfs_df)} rows")

slots_output = []
results      = {}

for slot_id in range(4):
    model_path = MODELS / f'nowcast_slot{slot_id}_xgb_v3_calibrated.pkl'
    if not model_path.exists():
        model_path = MODELS / f'nowcast_slot{slot_id}_xgb_v2_calibrated.pkl'

    if not model_path.exists():
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

    artifact     = joblib.load(model_path)
    model        = artifact['model']
    feature_cols = artifact['feature_cols']
    threshold    = artifact['threshold']

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
        slot_gfs = gfs_df[gfs_df['slot'] == slot_id]
        if len(slot_gfs) > 0:
            row = slot_gfs.iloc[0]
            for col in ['ERA5_T2M','ERA5_D2M','ERA5_U10','ERA5_V10','ERA5_CAPE','ERA5_SP',
                        'ERA5_t_500hPa','ERA5_t_700hPa','ERA5_t_850hPa',
                        'ERA5_q_500hPa','ERA5_q_700hPa','ERA5_q_850hPa',
                        'ERA5_u_500hPa','ERA5_u_700hPa','ERA5_u_850hPa',
                        'ERA5_v_500hPa','ERA5_v_700hPa','ERA5_v_850hPa']:
                if col in row and pd.notna(row[col]):
                    obs[col] = float(row[col])

    if slot_id in upper_air:
        ua = upper_air[slot_id]
        for col in ['CAPE','CIN','K_INDEX','LIFTED_INDEX','TOTALS_TOTALS','PRECIP_WATER']:
            if col in ua and pd.notna(ua[col]):
                obs[col] = float(ua[col])

    obs  = compute_derived(obs, slot_id)
    X    = np.array([[float(obs.get(c, 0.0)) for c in feature_cols]])
    raw  = float(model.predict_proba(X)[0][1])
    cal  = apply_calibrator(artifact, raw)
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

alert_active     = any(s['predicted'] for s in slots_output)
peak_slot        = max(results, key=results.get)
peak_probability = results[peak_slot]
met_slot         = next((s for s in slots_output if s['slot'] == 2), slots_output[0])

forecast = {
    "date":             date_str,
    "generated_at":     now.strftime('%Y-%m-%d %H:%M IST'),
    "alert_active":     alert_active,
    "peak_slot":        peak_slot,
    "peak_probability": round(float(peak_probability), 4),
    "model_version":    "v3_calibrated",
    "slots":            slots_output,
    "met_parameters": {
        "ua_cape_jkg":       met_slot.get('cape', 0),
        "ua_k_index":        met_slot.get('k_index', 0),
        "ua_lifted_index":   met_slot.get('lifted_index', 0),
        "ua_totals_totals":  met_slot.get('totals_totals', 0),
        "instability_level": (
            "Extreme" if met_slot.get('cape', 0) >= 3000 else
            "High"    if met_slot.get('cape', 0) >= 1500 else
            "Moderate" if met_slot.get('cape', 0) >= 500 else
            "Marginal" if met_slot.get('cape', 0) >= 100 else
            "Stable"
        ),
    },
}

with open('forecast.json', 'w') as f:
    json.dump(forecast, f, indent=2)

print(f"forecast.json updated: alert={alert_active} peak=Slot{peak_slot} {peak_probability*100:.1f}%")
