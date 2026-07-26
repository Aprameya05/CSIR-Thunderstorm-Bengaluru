import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# File paths
BASE = Path(__file__).resolve().parent
IMD_FILE = BASE / "43295_Table_2_Daily_NDCQ202607153.csv"
FORECAST_LOG = BASE / "data" / "forecast_log.csv"
OUTPUT_JSON = BASE / "data" / "verification_today.json"

# Read yesterday's forecasts
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
df = pd.read_csv(FORECAST_LOG)

# Add predicted column if missing (threshold = 0.30)
if 'predicted' not in df.columns:
    df['predicted'] = (df['ts_probability'] >= 0.30).astype(int)

yesterday_rows = df[df['date'] == yesterday].tail(4)

if yesterday_rows.empty:
    print(f"⚠️ Warning: No forecasts found for {yesterday} in forecast_log.csv.")
    # Write empty verification so forecast.json still gets updated
    summary = {
        "date_verified": yesterday,
        "slots_verified": 0,
        "thunderstorm_observed": None,
        "slot_results": [],
        "rolling_30d": {"POD": 0.0, "FAR": 0.0, "HSS": 0.0, "Brier": 0.0}
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))
    # Also inject into forecast.json
    FORECAST_JSON = BASE / "forecast.json"
    try:
        with open(FORECAST_JSON, "r") as f:
            fc = json.load(f)
        fc["verification"] = summary
        with open(FORECAST_JSON, "w") as f:
            json.dump(fc, f, indent=2)
        print("✓ Also written to forecast.json")
    except Exception as e:
        print(f"⚠️ Could not write to forecast.json: {e}")
    exit(0)  # graceful exit

# Read IMD file
imd = pd.read_csv(IMD_FILE)

# Build a proper DATE column from YEAR, MN, DT
imd['DATE'] = pd.to_datetime(
    imd[['YEAR', 'MN', 'DT']].rename(
        columns={'YEAR': 'year', 'MN': 'month', 'DT': 'day'}
    )
).dt.strftime('%Y-%m-%d')

if yesterday not in imd['DATE'].values:
    print(f"⚠️ Warning: {yesterday} not found in IMD file.")
    summary = {
        "date_verified": yesterday,
        "slots_verified": 0,
        "thunderstorm_observed": None,
        "slot_results": [],
        "rolling_30d": {"POD": 0.0, "FAR": 0.0, "HSS": 0.0, "Brier": 0.0}
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))
    # Also inject into forecast.json
    FORECAST_JSON = BASE / "forecast.json"
    try:
        with open(FORECAST_JSON, "r") as f:
            fc = json.load(f)
        fc["verification"] = summary
        with open(FORECAST_JSON, "w") as f:
            json.dump(fc, f, indent=2)
        print("✓ Also written to forecast.json")
    except Exception as e:
        print(f"⚠️ Could not write to forecast.json: {e}")
    exit(0)

# TH flag is column index 22 (0-based)
th_flag = int(imd.loc[imd['DATE'] == yesterday].iloc[0, 22])

# Determine actual outcomes
actuals = []
for _, row in yesterday_rows.iterrows():
    slot = row['slot']
    if th_flag == 1 and slot in [2, 3]:
        actual = 1
    else:
        actual = 0
    actuals.append(actual)
    df.loc[(df['date'] == yesterday) & (df['slot'] == slot), 'actual'] = actual

# Compute verification metrics for the last 30 days (4 slots/day = 120 rows)
last30 = df.tail(120)
hits = ((last30['predicted'] == 1) & (last30['actual'] == 1)).sum()
misses = ((last30['predicted'] == 0) & (last30['actual'] == 1)).sum()
false_alarms = ((last30['predicted'] == 1) & (last30['actual'] == 0)).sum()
correct_negatives = ((last30['predicted'] == 0) & (last30['actual'] == 0)).sum()

POD = hits / (hits + misses) if (hits + misses) > 0 else 0.0
FAR = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
HSS = 2 * (hits * correct_negatives - misses * false_alarms) / (
    (hits + misses) * (misses + correct_negatives) +
    (hits + false_alarms) * (false_alarms + correct_negatives)
) if (hits + misses + false_alarms + correct_negatives) > 0 else 0.0
Brier = ((last30['predicted'] - last30['actual']) ** 2).mean()

# Build summary JSON
summary = {
    "date_verified": yesterday,
    "slots_verified": len(yesterday_rows),
    "thunderstorm_observed": bool(th_flag),
    "slot_results": [
        {
            "slot": int(row['slot']),
            "predicted": bool(row['predicted']),
            "actual": bool(act),
            "correct": bool(row['predicted'] == act)
        }
        for row, act in zip(yesterday_rows.to_dict('records'), actuals)
    ],
    "rolling_30d": {
        "POD": round(POD, 3),
        "FAR": round(FAR, 3),
        "HSS": round(HSS, 3),
        "Brier": round(Brier, 3)
    }
}

OUTPUT_JSON.write_text(json.dumps(summary, indent=2))
print(" Verification complete. Summary saved to verification_today.json")

# Also inject into forecast.json
FORECAST_JSON = BASE / "forecast.json"
try:
    with open(FORECAST_JSON, "r") as f:
        fc = json.load(f)
    fc["verification"] = summary
    with open(FORECAST_JSON, "w") as f:
        json.dump(fc, f, indent=2)
    print("✓ Also written to forecast.json")
except Exception as e:
    print(f"⚠️ Could not write to forecast.json: {e}")
