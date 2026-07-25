"""
compute_realtime_shap.py
========================
Computes real-time SHAP values for today's GFS inputs
against the v3 calibrated slot models.
Saves results to data/realtime_shap.json

Usage:
  python compute_realtime_shap.py
"""

import json
import math
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from datetime import datetime
import pytz

BASE   = Path('.')
MODELS = BASE / 'models'
DATA   = BASE / 'data'
OUT    = DATA / 'realtime_shap.json'

IST = pytz.timezone('Asia/Kolkata')

THRESHOLDS = {0: 0.24, 1: 0.38, 2: 0.16, 3: 0.39}

def build_obs_from_gfs(gfs_row, slot_id):
    """Build full 54-feature vector from GFS row."""
    now = datetime.now(IST)
    m   = now.month
    doy = now.timetuple().tm_yday

    obs = {
        'month': m, 'doy': doy,
        'MAX': 30.0, 'MIN': 20.0, 'RF': 0.0, 'AW': 3.0,
        'EVP': 5.0, 'DRNRF': 0.0, 'SSH': 300.0,
        'RF_3d': 0.0, 'RF_7d': 0.0, 'MAX_3d_avg': 30.0,
        'MIN_3d_avg': 20.0, 'DTR_3d_avg': 10.0,
        'RF_lag1': 0.0, 'MAX_lag1': 30.0, 'MIN_lag1': 20.0,
        'LABEL_lag1': 0, 'CAPE': 0.0, 'CIN': 0.0,
        'K_INDEX': 30.0, 'LIFTED_INDEX': -2.0,
        'TOTALS_TOTALS': 44.0, 'PRECIP_WATER': 40.0,
        'ERA5_T2M': 299.0, 'ERA5_D2M': 293.0,
        'ERA5_U10': -2.0, 'ERA5_V10': 1.0,
        'ERA5_CAPE': 0.0, 'ERA5_SP': 91500.0,
        'ERA5_t_500hPa': 268.0, 'ERA5_t_700hPa': 283.0,
        'ERA5_t_850hPa': 293.0,
        'ERA5_q_500hPa': 0.003, 'ERA5_q_700hPa': 0.009,
        'ERA5_q_850hPa': 0.013,
        'ERA5_u_500hPa': 5.0, 'ERA5_u_700hPa': 2.0,
        'ERA5_u_850hPa': -3.0, 'ERA5_v_500hPa': 2.0,
        'ERA5_v_700hPa': 1.0, 'ERA5_v_850hPa': 2.0,
        'ts_label_lag1_slot': 0, 'ts_any_yesterday': 0,
    }

    # Override with real GFS values
    if gfs_row is not None:
        for col in gfs_row.index:
            if col in obs and pd.notna(gfs_row[col]):
                obs[col] = float(gfs_row[col])
        # Stability indices
        for col in ['K_INDEX', 'TOTALS_TOTALS', 'LIFTED_INDEX', 'CAPE', 'PRECIP_WATER']:
            if col in gfs_row.index and pd.notna(gfs_row[col]):
                obs[col] = float(gfs_row[col])

    # Derived features
    DTR = obs['MAX'] - obs['MIN']
    obs['DTR'] = DTR
    obs['HA_flag'] = 0
    obs['RF_nonzero'] = 1 if obs['RF'] > 0 else 0
    obs['SEASON'] = 1 if m in [3,4,5] else 2 if m in [6,7,8,9] else 3 if m in [10,11] else 0
    obs['MONTH_sin'] = math.sin(2*math.pi*m/12)
    obs['MONTH_cos'] = math.cos(2*math.pi*m/12)
    obs['DOY_sin']   = math.sin(2*math.pi*doy/365)
    obs['DOY_cos']   = math.cos(2*math.pi*doy/365)
    obs['doy_sin']   = obs['DOY_sin']
    obs['doy_cos']   = obs['DOY_cos']
    obs['slot_sin']  = math.sin(2*math.pi*slot_id/4)
    obs['slot_cos']  = math.cos(2*math.pi*slot_id/4)
    CLIM = {(4,2):0.129,(5,2):0.194,(6,2):0.096,(7,2):0.032,(8,2):0.052,(9,2):0.079,(10,2):0.077}
    obs['slot_month_clim'] = CLIM.get((m, slot_id), 0.02)
    obs['ERA5_CAPE'] = obs.get('ERA5_CAPE', obs['CAPE'])

    q850=obs['ERA5_q_850hPa']; q700=obs['ERA5_q_700hPa']
    q500=obs['ERA5_q_500hPa']; t850=obs['ERA5_t_850hPa']
    t500=obs['ERA5_t_500hPa']; u850=obs['ERA5_u_850hPa']
    v850=obs['ERA5_v_850hPa']; u700=obs['ERA5_u_700hPa']
    v700=obs['ERA5_v_700hPa']; u500=obs['ERA5_u_500hPa']
    v500=obs['ERA5_v_500hPa']
    CAPE=obs['CAPE']; K=obs['K_INDEX']
    LI=obs['LIFTED_INDEX']; TT=obs['TOTALS_TOTALS']

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


def compute_shap_for_slot(slot_id, gfs_row):
    model_path = MODELS / f'nowcast_slot{slot_id}_xgb_v3_calibrated.pkl'
    if not model_path.exists():
        return None

    artifact     = joblib.load(model_path)
    model        = artifact['model']
    feature_cols = artifact['feature_cols']
    threshold    = artifact['threshold']

    obs = build_obs_from_gfs(gfs_row, slot_id)
    X   = np.array([[float(obs.get(c, 0.0)) for c in feature_cols]])
    df  = pd.DataFrame(X, columns=feature_cols)

    # Compute SHAP values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # For binary classification, take class 1 SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    prob = float(model.predict_proba(X)[0][1])

    # Get top 10 features by absolute SHAP value
    feature_shap = list(zip(feature_cols, sv.tolist()))
    feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
    top10 = feature_shap[:10]

    return {
        'slot':       slot_id,
        'prob':       round(prob, 4),
        'threshold':  threshold,
        'base_value': round(float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value), 4),
        'top_features': [
            {
                'feature': f,
                'shap':    round(v, 5),
                'value':   round(float(obs.get(f, 0.0)), 4),
                'direction': 'increases_risk' if v > 0 else 'decreases_risk',
            }
            for f, v in top10
        ],
        'computed_at': datetime.now(IST).strftime('%Y-%m-%d %H:%M IST'),
    }


def main():
    print('=' * 60)
    print('  compute_realtime_shap.py — Real-Time SHAP')
    print('=' * 60)

    # Load GFS data
    gfs_path = DATA / 'gfs_realtime_43295.csv'
    gfs_row  = None
    if gfs_path.exists():
        df = pd.read_csv(gfs_path)
        if len(df) > 0:
            gfs_row = df.iloc[0]
            print(f'  GFS loaded: cycle={gfs_row.get("gfs_cycle")}')

    results = {}
    for slot_id in range(4):
        print(f'\n  Computing SHAP for Slot {slot_id}...')
        try:
            result = compute_shap_for_slot(slot_id, gfs_row)
            if result:
                results[str(slot_id)] = result
                print(f'  ✓ Top feature: {result["top_features"][0]["feature"]} (SHAP={result["top_features"][0]["shap"]:.4f})')
        except Exception as e:
            print(f'  ✗ Slot {slot_id} error: {e}')

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {OUT}')


if __name__ == '__main__':
    main()