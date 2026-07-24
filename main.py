"""
main.py
=======
CSIR Thunderstorm Prediction System — FastAPI Application
Bengaluru Airport (Station 43295)

Endpoints:
  GET  /                          — health check
  POST /predict                   — daily thunderstorm prediction
  POST /nowcast/predict/slot/{id} — 6-hour slot prediction
  POST /nowcast/predict/all       — all 4 slots in one call
  GET  /nowcast/slots/info        — slot metadata

Run:
  uvicorn main:app --reload

Swagger UI:
  http://127.0.0.1:8000/docs

Author: Satvik (Deployment), Aprameya (ML), CSIR Thunderstorm Project
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CSIR Thunderstorm Prediction System",
    description="AI-based thunderstorm forecasting for Bengaluru Airport (IMD Station 43295)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
MODELS    = BASE / "models"

# ── SLOT METADATA ─────────────────────────────────────────────────────────────
SLOT_INFO = {
    0: {"label": "0001-0600 IST", "era5_utc": "00Z", "description": "Late night / early morning"},
    1: {"label": "0601-1200 IST", "era5_utc": "06Z", "description": "Morning"},
    2: {"label": "1201-1800 IST", "era5_utc": "12Z", "description": "Afternoon (peak TS window)"},
    3: {"label": "1801-2400 IST", "era5_utc": "18Z", "description": "Evening"},
}

# ── LOAD MODELS AT STARTUP ────────────────────────────────────────────────────
daily_model_artifact  = None
nowcast_slot_artifacts = {}

@app.on_event("startup")
def load_models():
    global daily_model_artifact, nowcast_slot_artifacts

    # Daily model
    daily_path = MODELS / "thunderstorm_model.pkl"
    if daily_path.exists():
        daily_model_artifact = joblib.load(daily_path)
        print(f"✓ Daily model loaded: {daily_path.name}")
    else:
        print(f"⚠ Daily model not found at {daily_path}")

    # Nowcast slot models
    for slot_id in range(4):
        slot_path = MODELS / f"nowcast_slot{slot_id}_xgb_v3_calibrated.pkl"
        if slot_path.exists():
            nowcast_slot_artifacts[slot_id] = joblib.load(slot_path)
            print(f"✓ Slot {slot_id} model loaded: {slot_path.name}")
        else:
            print(f"⚠ Slot {slot_id} model not found at {slot_path}")

# ── SCHEMAS — DAILY ───────────────────────────────────────────────────────────
class DailyInput(BaseModel):
    date:       str   = Field(..., example="2026-07-16")
    MAX:        float = Field(..., example=29.0)
    MIN:        float = Field(..., example=21.0)
    AW:         float = Field(..., example=4.0)
    RF:         float = Field(..., example=2.1)
    SSH:        float = Field(..., example=180.0)
    RF_lag1:    float = Field(0.0, example=0.0)
    MAX_lag1:   float = Field(..., example=28.5)
    MIN_lag1:   float = Field(..., example=20.8)
    LABEL_lag1: int   = Field(0,   example=0)

class DailyOutput(BaseModel):
    date:                    str
    thunderstorm_probability: float
    alert_level:             str
    prediction:              bool
    threshold:               float
    message:                 str

# ── SCHEMAS — NOWCAST ─────────────────────────────────────────────────────────
class NowcastInput(BaseModel):
    date:       str = Field(..., example="2026-07-16")
    slot:       int = Field(..., ge=0, le=3)

    # Surface
    MAX:        float = Field(..., example=29.0)
    MIN:        float = Field(..., example=21.0)
    AW:         float = Field(0.0, example=4.0)
    RF:         float = Field(0.0, example=0.0)
    EVP:        float = Field(5.0, example=5.0)
    DRNRF:      float = Field(0.0, example=0.0)
    SSH:        float = Field(300.0, example=300.0)

    # Rolling / lag
    RF_3d:       float = Field(0.0)
    RF_7d:       float = Field(0.0)
    MAX_3d_avg:  float = Field(None)
    MIN_3d_avg:  float = Field(None)
    DTR_3d_avg:  float = Field(None)
    RF_lag1:     float = Field(0.0)
    MAX_lag1:    float = Field(None)
    MIN_lag1:    float = Field(None)
    LABEL_lag1:  int   = Field(0)

    # Stability indices
    CAPE:          float = Field(..., example=500.0)
    K_INDEX:       float = Field(..., example=35.0)
    LIFTED_INDEX:  float = Field(..., example=-2.0)
    TOTALS_TOTALS: float = Field(..., example=45.0)
    PRECIP_WATER:  float = Field(40.0, example=40.0)

    # ERA5
    ERA5_T2M:      float = Field(..., example=299.0)
    ERA5_D2M:      float = Field(293.0)
    ERA5_U10:      float = Field(-2.0)
    ERA5_V10:      float = Field(1.0)
    ERA5_CAPE:     float = Field(None)
    ERA5_SP:       float = Field(91500.0)
    ERA5_t_500hPa: float = Field(268.0)
    ERA5_t_700hPa: float = Field(283.0)
    ERA5_t_850hPa: float = Field(293.0)
    ERA5_q_500hPa: float = Field(0.003)
    ERA5_q_700hPa: float = Field(0.009)
    ERA5_q_850hPa: float = Field(0.013)
    ERA5_u_500hPa: float = Field(5.0)
    ERA5_u_700hPa: float = Field(2.0)
    ERA5_u_850hPa: float = Field(-3.0)
    ERA5_v_500hPa: float = Field(2.0)
    ERA5_v_700hPa: float = Field(1.0)
    ERA5_v_850hPa: float = Field(2.0)

    # Slot lag
    ts_label_lag1_slot: int   = Field(0)
    ts_any_yesterday:   int   = Field(0)

class NowcastOutput(BaseModel):
    date:              str
    slot:              int
    slot_label:        str
    ts_probability:    float
    ts_predicted:      bool
    threshold_used:    float
    alert_level:       str

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_alert_level(prob: float) -> str:
    if prob < 0.20:   return "GREEN"
    elif prob < 0.45: return "YELLOW"
    elif prob < 0.70: return "ORANGE"
    else:             return "RED"

def derive_nowcast_features(p: NowcastInput) -> dict:
    """Fill derived fields and defaults from raw input."""
    import math
    date  = pd.Timestamp(p.date)
    doy   = date.dayofyear
    month = date.month
    slot  = p.slot

    DTR = p.MAX - p.MIN
    m   = month

    season = 1 if m in [3,4,5] else 2 if m in [6,7,8,9] else 3 if m in [10,11] else 0

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

    obs =  {
        "MAX": p.MAX, "MIN": p.MIN, "DTR": DTR,
        "AW": p.AW, "RF": p.RF, "EVP": p.EVP,
        "DRNRF": p.DRNRF, "SSH": p.SSH,
        "RF_3d": p.RF_3d, "RF_7d": p.RF_7d,
        "MAX_3d_avg": p.MAX_3d_avg or p.MAX,
        "MIN_3d_avg": p.MIN_3d_avg or p.MIN,
        "DTR_3d_avg": p.DTR_3d_avg or DTR,
        "RF_lag1": p.RF_lag1,
        "MAX_lag1": p.MAX_lag1 or p.MAX,
        "MIN_lag1": p.MIN_lag1 or p.MIN,
        "LABEL_lag1": p.LABEL_lag1,
        "MONTH_sin": math.sin(2*math.pi*m/12),
        "MONTH_cos": math.cos(2*math.pi*m/12),
        "DOY_sin":   math.sin(2*math.pi*doy/365),
        "DOY_cos":   math.cos(2*math.pi*doy/365),
        "SEASON": season,
        "HA_flag": 0, "RF_nonzero": 1 if p.RF > 0 else 0,
        "CAPE": p.CAPE, "CIN": 0.0,
        "K_INDEX": p.K_INDEX, "LIFTED_INDEX": p.LIFTED_INDEX,
        "TOTALS_TOTALS": p.TOTALS_TOTALS, "PRECIP_WATER": p.PRECIP_WATER,
        "ERA5_T2M": p.ERA5_T2M, "ERA5_D2M": p.ERA5_D2M,
        "ERA5_U10": p.ERA5_U10, "ERA5_V10": p.ERA5_V10,
        "ERA5_CAPE": p.ERA5_CAPE or p.CAPE, "ERA5_SP": p.ERA5_SP,
        "ERA5_t_500hPa": p.ERA5_t_500hPa, "ERA5_t_700hPa": p.ERA5_t_700hPa,
        "ERA5_t_850hPa": p.ERA5_t_850hPa,
        "ERA5_q_500hPa": p.ERA5_q_500hPa, "ERA5_q_700hPa": p.ERA5_q_700hPa,
        "ERA5_q_850hPa": p.ERA5_q_850hPa,
        "ERA5_u_500hPa": p.ERA5_u_500hPa, "ERA5_u_700hPa": p.ERA5_u_700hPa,
        "ERA5_u_850hPa": p.ERA5_u_850hPa,
        "ERA5_v_500hPa": p.ERA5_v_500hPa, "ERA5_v_700hPa": p.ERA5_v_700hPa,
        "ERA5_v_850hPa": p.ERA5_v_850hPa,
        "slot_sin": math.sin(2*math.pi*slot/4),
        "slot_cos": math.cos(2*math.pi*slot/4),
        "slot_month_clim": CLIM.get((m, slot), 0.0),
        "doy_sin": math.sin(2*math.pi*doy/365),
        "doy_cos": math.cos(2*math.pi*doy/365),
        "ts_label_lag1_slot": p.ts_label_lag1_slot,
        "ts_any_yesterday":   p.ts_any_yesterday,
    }
    obs['cape_x_kindex']      = p.CAPE * p.K_INDEX
    obs['li_x_totals']        = abs(p.LIFTED_INDEX) * p.TOTALS_TOTALS
    obs['q_gradient_500_850'] = p.ERA5_q_850hPa - p.ERA5_q_500hPa
    obs['thetae_850']         = p.ERA5_t_850hPa + 2491 * p.ERA5_q_850hPa
    obs['wind_shear_500_850'] = ((p.ERA5_u_500hPa-p.ERA5_u_850hPa)**2 + (p.ERA5_v_500hPa-p.ERA5_v_850hPa)**2)**0.5
    obs['wind_shear_700_850'] = ((p.ERA5_u_700hPa-p.ERA5_u_850hPa)**2 + (p.ERA5_v_700hPa-p.ERA5_v_850hPa)**2)**0.5
    obs['moisture_flux_850']  = p.ERA5_q_850hPa * (p.ERA5_u_850hPa**2 + p.ERA5_v_850hPa**2)**0.5
    obs['moisture_flux_700']  = p.ERA5_q_700hPa * (p.ERA5_u_700hPa**2 + p.ERA5_v_700hPa**2)**0.5
    obs['thickness_500_850']  = p.ERA5_t_850hPa - p.ERA5_t_500hPa
    obs['mid_level_drying']   = p.ERA5_q_700hPa / (p.ERA5_q_850hPa + 1e-9)

    return obs

# ── ROUTES ────────────────────────────────────────────────────────────────────
# @app.get("/")
# def health():
#     return {
#         "status": "ok",
#         "system": "CSIR Thunderstorm Prediction System",
#         "station": "Bengaluru Airport — IMD 43295",
#         "daily_model":   daily_model_artifact is not None,
#         "nowcast_models": {s: s in nowcast_slot_artifacts for s in range(4)},
#     }

@app.get("/")
def health():
    nowcast_info = {}
    for s in range(4):
        if s in nowcast_slot_artifacts:
            a = nowcast_slot_artifacts[s]
            nowcast_info[str(s)] = {
                "loaded":    True,
                "version":   "v3_calibrated",
                "threshold": a.get("threshold", "unknown"),
                "slot_name": a.get("slot_name", SLOT_INFO[s]["label"]),
            }
        else:
            nowcast_info[str(s)] = {"loaded": False}
    return {
        "status":         "ok",
        "system":         "CSIR Thunderstorm Prediction System",
        "station":        "Bengaluru Airport — IMD 43295",
        "daily_model":    daily_model_artifact is not None,
        "nowcast_models": nowcast_info,
    }

@app.post("/predict", response_model=DailyOutput)
def predict_daily(payload: DailyInput):
    if daily_model_artifact is None:
        raise HTTPException(503, "Daily model not loaded")

    model    = daily_model_artifact.get("model") or daily_model_artifact
    features = daily_model_artifact.get("feature_cols", [])
    threshold = daily_model_artifact.get("threshold", 0.45)

    data = payload.dict()
    vec  = np.array([[data.get(f, 0.0) for f in features]])
    prob = float(model.predict_proba(vec)[0][1])
    pred = prob >= threshold

    return DailyOutput(
        date=payload.date,
        thunderstorm_probability=round(prob, 4),
        alert_level=get_alert_level(prob),
        prediction=pred,
        threshold=threshold,
        message=f"Thunderstorm probability: {prob*100:.1f}%",
    )

@app.get("/nowcast/slots/info")
def slot_info():
    return {"slots": SLOT_INFO}

@app.post("/nowcast/predict/slot/{slot_id}", response_model=NowcastOutput)
def predict_slot(slot_id: int, payload: NowcastInput):
    if slot_id not in range(4):
        raise HTTPException(400, "slot_id must be 0, 1, 2, or 3")
    if slot_id not in nowcast_slot_artifacts:
        raise HTTPException(503, f"Slot {slot_id} model not loaded")
    if payload.slot != slot_id:
        raise HTTPException(400, "URL slot_id must match payload slot field")
    if payload.CAPE is None or payload.CAPE == 0.0:
        raise HTTPException(
            status_code=422,
            detail="CAPE is required and cannot be zero"
        )

    if payload.ERA5_T2M is None or payload.ERA5_T2M == 0.0:
        raise HTTPException(
            status_code=422,
            detail="ERA5_T2M is required and cannot be zero"
        )

    artifact     = nowcast_slot_artifacts[slot_id]
    model        = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold    = artifact["threshold"]

    obs = derive_nowcast_features(payload)
    vec = np.array([[obs.get(f, 0.0) for f in feature_cols]])

    prob = float(model.predict_proba(vec)[0][1])
    pred = prob >= threshold

    return NowcastOutput(
        date=payload.date,
        slot=slot_id,
        slot_label=SLOT_INFO[slot_id]["label"],
        ts_probability=round(prob, 4),
        ts_predicted=pred,
        threshold_used=threshold,
        alert_level=get_alert_level(prob),
    )

@app.post("/nowcast/predict/all")
def predict_all(payloads: list[NowcastInput]):
    if len(payloads) != 4:
        raise HTTPException(400, "Provide exactly 4 inputs, one per slot (0-3)")
    results = []
    for p in sorted(payloads, key=lambda x: x.slot):
        try:
            results.append(predict_slot(p.slot, p))
        except HTTPException as e:
            results.append({"slot": p.slot, "error": e.detail})
    return {"date": payloads[0].date, "predictions": results}
