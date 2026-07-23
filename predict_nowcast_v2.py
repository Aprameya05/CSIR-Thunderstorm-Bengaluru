"""
predict_nowcast_v2.py
=====================
6-Hour Thunderstorm Nowcast for Bengaluru Airport (Station 43295).
Uses CALIBRATED v2 models (6-hourly ERA5).

Four modes:

  1. DEMO mode (default) — real test case 2023-04-29
     python predict_nowcast_v2.py

  2. DATE mode — look up from training dataset
     python predict_nowcast_v2.py --date 2023-10-11

  3. LIVE-GFS mode — load today's GFS real-time file automatically
     python predict_nowcast_v2.py --live-gfs

  4. LIVE-GFS for specific date
     python predict_nowcast_v2.py --live-gfs --date 2026-07-16

For --live-gfs to work, run first:
  python fetch_gfs_realtime.py
  python fetch_upperair_realtime.py --slot 2

Author: Aprameya, CSIR Thunderstorm Project
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.isotonic import IsotonicRegression

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\Aprameya\OneDrive\Pictures\Desktop\CSIR_Thunderstorm")
MODELS    = BASE / "models"
DATA      = BASE / "data" / "bengaluru_6hr_training_dataset_v2.csv"
GFS_DIR   = BASE / "data" / "gfs_realtime"
UPPER_AIR = BASE / "data" / "upperair_realtime_43295.csv"

SLOT_NAMES = {
    0: "0001-0600 IST",
    1: "0601-1200 IST",
    2: "1201-1800 IST",
    3: "1801-2400 IST",
}
SLOT_EMOJI = {0: "🌙", 1: "🌅", 2: "☀️ ", 3: "🌆"}

# ── DEMO INPUT (2023-04-29 — confirmed thunderstorm day) ──────────────────────
DEMO_OBS = {
    "date": "2023-04-29",
    "MAX": 34.5, "MIN": 22.2, "AW": 3.0, "RF": 0.0,
    "EVP": 6.0, "DRNRF": 0.0, "SSH": 426.0,
    "RF_3d": 0.0, "RF_7d": 6.8,
    "MAX_3d_avg": 33.27, "MIN_3d_avg": 22.97, "DTR_3d_avg": 10.3,
    "RF_lag1": 0.0, "MAX_lag1": 33.5, "MIN_lag1": 22.0, "LABEL_lag1": 0,
    "CAPE": 177.25, "K_INDEX": 39.38, "LIFTED_INDEX": -6.15,
    "TOTALS_TOTALS": 46.78, "PRECIP_WATER": 40.55,
    "ERA5_T2M": 300.12, "ERA5_D2M": 294.73,
    "ERA5_U10": -3.96, "ERA5_V10": 2.39,
    "ERA5_CAPE": 177.25, "ERA5_SP": 91197.0,
    "ERA5_t_500hPa": 268.44, "ERA5_t_700hPa": 283.67, "ERA5_t_850hPa": 293.79,
    "ERA5_q_500hPa": 0.00235, "ERA5_q_700hPa": 0.00947, "ERA5_q_850hPa": 0.01404,
    "ERA5_u_500hPa": 5.16, "ERA5_u_700hPa": 0.44, "ERA5_u_850hPa": -7.54,
    "ERA5_v_500hPa": 2.34, "ERA5_v_700hPa": -0.14, "ERA5_v_850hPa": 4.21,
    "ts_label_lag1_slot": 0, "ts_any_yesterday": 0,
}
DEMO_ACTUAL = {0: 0, 1: 0, 2: 1, 3: 1}

# ── LOAD CALIBRATED MODELS ────────────────────────────────────────────────────
def load_models():
    models = {}
    for slot_id in range(4):
        path = MODELS / f"nowcast_slot{slot_id}_xgb_v2_calibrated.pkl"
        if not path.exists():
            path = MODELS / f"nowcast_slot{slot_id}_xgb_v2.pkl"
            print(f"  Slot {slot_id}: calibrated not found, using v2")
        if not path.exists():
            raise FileNotFoundError(f"No model found for slot {slot_id}")
        models[slot_id] = joblib.load(path)
    return models

def apply_calibrator(artifact, raw_prob):
    calibrator   = artifact.get('calibrator')
    calib_method = artifact.get('calib_method', 'none')
    if calibrator is None:
        return raw_prob
    if calib_method == 'sigmoid':
        return calibrator.predict_proba(raw_prob.reshape(-1,1))[:,1]
    return calibrator.predict(raw_prob)

# ── DERIVED FEATURES ──────────────────────────────────────────────────────────
def compute_derived(obs: dict, slot_id: int) -> dict:
    import math
    date  = pd.Timestamp(obs['date'])
    doy   = date.dayofyear
    month = date.month

    obs['DTR']      = obs['MAX'] - obs['MIN']
    obs['HA_flag']  = int(obs.get('HA_flag', 0))
    obs['RF_nonzero'] = 1 if obs.get('RF', 0) > 0 else 0

    m = month
    if m in [3,4,5]:     obs['SEASON'] = 1
    elif m in [6,7,8,9]: obs['SEASON'] = 2
    elif m in [10,11]:   obs['SEASON'] = 3
    else:                 obs['SEASON'] = 0

    obs['MONTH_sin'] = math.sin(2*math.pi*m/12)
    obs['MONTH_cos'] = math.cos(2*math.pi*m/12)
    obs['DOY_sin']   = math.sin(2*math.pi*doy/365)
    obs['DOY_cos']   = math.cos(2*math.pi*doy/365)
    obs['doy_sin']   = obs['DOY_sin']
    obs['doy_cos']   = obs['DOY_cos']

    obs['slot_sin'] = math.sin(2*math.pi*slot_id/4)
    obs['slot_cos'] = math.cos(2*math.pi*slot_id/4)

    CLIM = {
        (1,0):0.000,(1,1):0.000,(1,2):0.000,(1,3):0.000,
        (2,0):0.000,(2,1):0.000,(2,2):0.009,(2,3):0.013,
        (3,0):0.008,(3,1):0.004,(3,2):0.036,(3,3):0.028,
        (4,0):0.025,(4,1):0.008,(4,2):0.129,(4,3):0.100,
        (5,0):0.129,(5,1):0.036,(5,2):0.194,(5,3):0.181,
        (6,0):0.042,(6,1):0.004,(6,2):0.096,(6,3):0.092,
        (7,0):0.028,(7,1):0.004,(7,2):0.032,(7,3):0.044,
        (8,0):0.020,(8,1):0.008,(8,2):0.052,(8,3):0.060,
        (9,0):0.083,(9,1):0.029,(9,2):0.079,(9,3):0.067,
        (10,0):0.077,(10,1):0.024,(10,2):0.077,(10,3):0.077,
        (11,0):0.008,(11,1):0.008,(11,2):0.021,(11,3):0.017,
        (12,0):0.000,(12,1):0.000,(12,2):0.000,(12,3):0.000,
    }
    obs['slot_month_clim'] = CLIM.get((m, slot_id), 0.0)
    if 'ERA5_CAPE' not in obs or obs['ERA5_CAPE'] is None:
        obs['ERA5_CAPE'] = obs.get('CAPE', 0.0)
    obs['CIN'] = obs.get('CIN', 0.0)
    return obs

def build_feature_vector(obs, feature_cols):
    return np.array([[float(obs.get(col, 0.0)) for col in feature_cols]])

# ── LOAD GFS REALTIME DATA ────────────────────────────────────────────────────
def load_gfs_data(date_str: str) -> dict:
    """Load GFS real-time CSV for given date and merge with upper-air if available."""
    gfs_file = GFS_DIR / f"gfs_{date_str}.csv"
    if not gfs_file.exists():
        raise FileNotFoundError(
            f"GFS file not found: {gfs_file}\n"
            f"Run: python fetch_gfs_realtime.py --date {date_str}"
        )

    gfs_df = pd.read_csv(gfs_file)
    print(f"  GFS file loaded: {gfs_file.name} ({len(gfs_df)} rows)")

    # Load upper-air stability indices if available
    upper_air_by_slot = {}
    if UPPER_AIR.exists():
        ua_df = pd.read_csv(UPPER_AIR)
        ua_today = ua_df[ua_df['date'] == date_str]
        if len(ua_today) > 0:
            for _, row in ua_today.iterrows():
                upper_air_by_slot[int(row['slot'])] = row.to_dict()
            print(f"  Upper-air data loaded for {len(upper_air_by_slot)} slots")
        else:
            print(f"  ⚠ No upper-air data for {date_str} — stability indices will be estimated")
    else:
        print(f"  ⚠ No upper-air file found — run fetch_upperair_realtime.py")

    # Build per-slot observation dicts
    slot_obs = {}
    for slot_id in range(4):
        slot_row = gfs_df[gfs_df['slot'] == slot_id]
        if len(slot_row) == 0:
            print(f"  ⚠ Slot {slot_id}: no GFS data")
            continue

        row = slot_row.iloc[0]
        obs = {
            'date':     date_str,
            'ERA5_T2M': float(row.get('ERA5_T2M', 299.0)),
            'ERA5_D2M': float(row.get('ERA5_D2M', 293.0)),
            'ERA5_U10': float(row.get('ERA5_U10', -2.0)),
            'ERA5_V10': float(row.get('ERA5_V10', 1.0)),
            'ERA5_CAPE':float(row.get('ERA5_CAPE', 0.0)),
            'ERA5_SP':  float(row.get('ERA5_SP', 91500.0)),
            'ERA5_t_500hPa': float(row.get('ERA5_t_500hPa', 268.0)),
            'ERA5_t_700hPa': float(row.get('ERA5_t_700hPa', 283.0)),
            'ERA5_t_850hPa': float(row.get('ERA5_t_850hPa', 293.0)),
            'ERA5_q_500hPa': float(row.get('ERA5_q_500hPa', 0.003)),
            'ERA5_q_700hPa': float(row.get('ERA5_q_700hPa', 0.009)),
            'ERA5_q_850hPa': float(row.get('ERA5_q_850hPa', 0.013)),
            'ERA5_u_500hPa': float(row.get('ERA5_u_500hPa', 5.0)),
            'ERA5_u_700hPa': float(row.get('ERA5_u_700hPa', 2.0)),
            'ERA5_u_850hPa': float(row.get('ERA5_u_850hPa', -3.0)),
            'ERA5_v_500hPa': float(row.get('ERA5_v_500hPa', 2.0)),
            'ERA5_v_700hPa': float(row.get('ERA5_v_700hPa', 1.0)),
            'ERA5_v_850hPa': float(row.get('ERA5_v_850hPa', 2.0)),
            # Surface obs — use ERA5 T2M as proxy if IMD not available
            'MAX':    float(row.get('ERA5_T2M', 302.0)) - 273.15,
            'MIN':    float(row.get('ERA5_T2M', 302.0)) - 278.15,
            'AW': 0.0, 'RF': 0.0, 'EVP': 5.0, 'DRNRF': 0.0, 'SSH': 300.0,
            'RF_3d': 0.0, 'RF_7d': 0.0,
            'MAX_3d_avg': float(row.get('ERA5_T2M', 302.0)) - 273.15,
            'MIN_3d_avg': float(row.get('ERA5_T2M', 302.0)) - 278.15,
            'DTR_3d_avg': 8.0,
            'RF_lag1': 0.0, 'MAX_lag1': 0.0, 'MIN_lag1': 0.0,
            'LABEL_lag1': 0, 'ts_label_lag1_slot': 0, 'ts_any_yesterday': 0,
            # Stability indices — from upper-air if available
            'CAPE': 0.0, 'CIN': 0.0, 'K_INDEX': 30.0,
            'LIFTED_INDEX': -2.0, 'TOTALS_TOTALS': 44.0, 'PRECIP_WATER': 40.0,
        }

        # Override with real upper-air data if available
        if slot_id in upper_air_by_slot:
            ua = upper_air_by_slot[slot_id]
            obs['CAPE']          = float(ua.get('CAPE', obs['CAPE']))
            obs['CIN']           = float(ua.get('CIN', obs['CIN']))
            obs['K_INDEX']       = float(ua.get('K_INDEX', obs['K_INDEX']))
            obs['LIFTED_INDEX']  = float(ua.get('LIFTED_INDEX', obs['LIFTED_INDEX']))
            obs['TOTALS_TOTALS'] = float(ua.get('TOTALS_TOTALS', obs['TOTALS_TOTALS']))
            obs['PRECIP_WATER']  = float(ua.get('PRECIP_WATER', obs['PRECIP_WATER']))
            print(f"  Slot {slot_id}: upper-air merged — CAPE={obs['CAPE']:.0f} K={obs['K_INDEX']:.1f}")

        slot_obs[slot_id] = obs

    return slot_obs

# ── PRINT FORECAST ────────────────────────────────────────────────────────────
def print_forecast(date_str, results, mode, actual_labels=None):
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     CSIR THUNDERSTORM NOWCAST — Bengaluru Airport (43295)   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Date   : {date_str:<51}║")
    print(f"║  Mode   : {mode:<51}║")
    print(f"║  Models : Calibrated v2 (6-hourly ERA5)               ║")
    print(f"║  Run    : {datetime.now().strftime('%Y-%m-%d %H:%M IST'):<51}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {'Slot':<6} {'Window':<16} {'Prob':>6}  {'Bar':<20} {'Call':<8}║")
    print("╠══════════════════════════════════════════════════════════════╣")

    alert_slots = []
    for slot_id, res in results.items():
        prob      = res['probability']
        predicted = res['predicted']
        emoji     = SLOT_EMOJI[slot_id]
        filled    = int(prob * 20)
        bar       = "█" * filled + "░" * (20 - filled)
        call      = "⚠ YES" if predicted else "  NO "
        if predicted:
            alert_slots.append(slot_id)

        actual_str = ""
        if actual_labels:
            actual = actual_labels.get(slot_id)
            if actual == 1:
                actual_str = " ✓HIT" if predicted else " ✗MISS"
            else:
                actual_str = " ✗FA " if predicted else " ✓TN "

        print(f"║  {emoji} {slot_id}  {SLOT_NAMES[slot_id]:<16} "
              f"{prob*100:>5.1f}%  {bar}  {call}{actual_str:<5}║")

    print("╠══════════════════════════════════════════════════════════════╣")
    if alert_slots:
        peak = max(alert_slots, key=lambda s: results[s]['probability'])
        print(f"║  🔴 ALERT: Thunderstorm likely in {len(alert_slots)} window(s)"
              f"{'':>25}║")
        print(f"║     Peak risk: {SLOT_NAMES[peak]:<47}║")
    else:
        print(f"║  🟢 ALL CLEAR: No thunderstorm predicted today"
              f"{'':>15}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print()
    print("  Calibrated probabilities:")
    for slot_id, res in results.items():
        bar = "█" * int(res['probability'] * 40)
        print(f"  Slot {slot_id}  {res['probability']*100:5.1f}%  {bar}")

    print("\n  JSON output:")
    print("  [")
    for slot_id, res in results.items():
        comma = "," if slot_id < 3 else ""
        print(f'    {{"date":"{date_str}","slot":{slot_id},'
              f'"slot_label":"{SLOT_NAMES[slot_id]}",'
              f'"ts_probability":{res["probability"]:.4f},'
              f'"ts_predicted":{"true" if res["predicted"] else "false"},'
              f'"threshold_used":{res["threshold"]},'
              f'"model":"calibrated_v2"}}{comma}')
    print("  ]")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSIR 6-Hour Thunderstorm Nowcast v2")
    parser.add_argument('--date',     type=str, help="Date YYYY-MM-DD")
    parser.add_argument('--live-gfs', action='store_true',
                        help="Load today's GFS real-time data from data/gfs_realtime/")
    args = parser.parse_args()

    print("\nLoading calibrated v2 models...")
    models = load_models()
    print("All 4 models loaded.")

    # ── Get observations ──────────────────────────────────────────────────────
    actual_labels = None
    gfs_mode      = False

    if args.live_gfs:
        # Determine operational date
        if args.date:
            date_str = args.date
        else:
            ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
            date_str = ist.strftime("%Y-%m-%d")

        print(f"\nLIVE-GFS mode — date: {date_str}")
        print("Loading GFS real-time data...")
        try:
            slot_obs_map = load_gfs_data(date_str)
            gfs_mode = True
        except FileNotFoundError as e:
            print(f"\n✗ {e}")
            return

    elif args.date:
        print(f"\nLooking up {args.date} in training dataset...")
        df = pd.read_csv(DATA, parse_dates=['date'])
        day_data = df[df['date'] == args.date]
        if len(day_data) == 0:
            print(f"Date {args.date} not found in dataset.")
            return
        FEATURE_COLS = [c for c in df.columns if c not in
                        ['date','year','month','slot','slot_label','ts_label']]
        slot_obs_map = {}
        for _, row in day_data.iterrows():
            obs = {c: row[c] for c in FEATURE_COLS}
            obs['date'] = args.date
            slot_obs_map[int(row['slot'])] = obs
        actual_labels = {int(r['slot']): int(r['ts_label'])
                         for _, r in day_data.iterrows()}
        date_str = args.date
        print(f"Found. Actual labels: {actual_labels}")

    else:
        print("\nRunning DEMO mode — date: 2023-04-29 (real thunderstorm day)")
        slot_obs_map  = {s: DEMO_OBS.copy() for s in range(4)}
        actual_labels = DEMO_ACTUAL
        date_str      = "2023-04-29"

    # ── Run predictions ───────────────────────────────────────────────────────
    results = {}
    for slot_id in range(4):
        if slot_id not in slot_obs_map:
            print(f"  ⚠ Slot {slot_id}: no data available")
            results[slot_id] = {'probability': 0.0, 'predicted': False,
                                'threshold': 0.5}
            continue

        artifact     = models[slot_id]
        model        = artifact['model']
        feature_cols = artifact['feature_cols']
        threshold    = artifact['threshold']

        obs = slot_obs_map[slot_id].copy()
        obs = compute_derived(obs, slot_id)
        X   = build_feature_vector(obs, feature_cols)

        raw_prob  = float(model.predict_proba(X)[0][1])
        cal_prob  = float(apply_calibrator(artifact,
                          np.array([raw_prob]))[0])
        predicted = cal_prob >= threshold

        results[slot_id] = {
            'probability': cal_prob,
            'predicted':   predicted,
            'threshold':   threshold,
        }

    # ── Print ─────────────────────────────────────────────────────────────────
    mode_str = "LIVE-GFS (real-time)" if gfs_mode else \
               ("Dataset lookup" if args.date else "Demo")
    print_forecast(date_str, results, mode_str, actual_labels)

if __name__ == "__main__":
    main()
