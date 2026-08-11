<div align="center">

<img src="https://img.shields.io/badge/CSIR-Research%20Project-blue?style=for-the-badge" />

# CSIR Thunderstorm Prediction System

**Operational AI Thunderstorm Nowcasting for Bengaluru Airport (VOBL)**  
IMD Station 43295 · Kempegowda International Airport · Bengaluru, India

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-v6%20Temporal%20%2B%20Himawari-success" />
  <img src="https://img.shields.io/badge/Himawari--9-Band%2013%20IR-blueviolet" />
  <img src="https://img.shields.io/badge/Cloudflare%20Pages-Live-F38020?logo=cloudflare&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-5--cron%20CI%2FCD-2088FF?logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Real--Time%20Explainability-red" />
  <img src="https://img.shields.io/badge/Status-Operational%20%E2%9C%85-brightgreen" />
</p>

**[Live Dashboard](https://csir-thunderstorm-bengaluru.pages.dev)** &nbsp;·&nbsp; **[Live API](https://csir-thunderstorm-api.onrender.com)** &nbsp;·&nbsp; **[API Docs](https://csir-thunderstorm-api.onrender.com/docs)**

</div>

---

## What This Is

This is a fully operational AI system that predicts thunderstorm probability at Bengaluru Airport (VOBL) across four 6-hour windows every day. It runs on a 5-cron GitHub Actions pipeline that pulls real-time atmospheric data, runs XGBoost inference, computes SHAP explanations, and updates a public dashboard on Cloudflare Pages, all without any human intervention.

The goal is an operational nowcasting product that the Indian Meteorological Department and ATC can trust and eventually put their name on.

**What makes this different from a typical ML project:**

- It runs in production today, not just in a notebook
- It ingests four live data sources on every pipeline run
- The model explains every prediction in real-time using SHAP
- It handles missing data, stale sources, and fallback chains gracefully
- Verification metrics are computed daily against actual IMD observations

---

## Forecast Slots

The day is split into four 6-hour windows. Each slot has its own XGBoost model trained specifically on that window's historical data.

| Slot | Window (IST) | Period | Production Model | CV AUROC | Threshold | Notes |
|:----:|:------------:|:------:|:----------------:|:--------:|:---------:|:-----:|
| 0 | 00:01 - 06:00 | Late Night | v6+v4 Ensemble | 0.8484 | 0.24 | Low event rate |
| 1 | 06:01 - 12:00 | Morning | v6 Temporal | 0.8317 | 0.15 | 30 lag features |
| **2** | **12:01 - 18:00** | **Afternoon** | **v6 Temporal** | **0.8710** | **0.16** | **Primary operational slot** |
| 3 | 18:01 - 24:00 | Evening | v6 Temporal | 0.8710 | 0.39 | High base threshold |

**Slot 2 is the primary operational slot.** The 1300-1800 IST window captures Bengaluru's dominant thunderstorm mechanism: solar heating of the Deccan Plateau driving afternoon convection, often triggered by orographic uplift on the eastern slopes of the Western Ghats.

**October threshold fix:** Slot 2 threshold is automatically lowered from 0.16 to 0.10 in October. SHAP analysis showed that the DOY_sin feature suppresses output probabilities during the post-monsoon transition, causing systematic under-prediction. The threshold fix restores POD from 0.379 to 0.621 on the 2015-2025 test set.

**Monsoon regime adjustment:** All thresholds are now scaled dynamically by a monsoon phase factor. BREAK monsoon conditions raise thresholds by 30% to suppress false alarms during stratiform clouding periods. ACTIVE and CONVECTIVE_BURST regimes lower thresholds to catch more events. The regime is detected from real-time CAPE and K-Index each run.

---

## System Architecture

```
GitHub Actions (5 crons/day)
        |
        +-- gfs_fetcher.py
        |       NOAA NOMADS GFS 0.25 deg (anonymous, Chrome User-Agent required)
        |       - Surface fields: CAPE, CIN, K-Index, LI, TT, PW
        |       - Profile: T/q/u/v at 500/700/850 hPa
        |       - Multi-hour TMP: f006, f012, f018, f024 (for Tmax/Tmin)
        |       - 48h outlook: f024, f048 per-day aggregation
        |       - History: gfs_history_43295.json (CAPE tendency computation)
        |       Output: gfs_realtime_43295.csv, upperair_realtime_43295.csv
        |               gfs_multiday_43295.json
        |
        +-- fetch_himawari_realtime.py
        |       Himawari-9 Band 13 (10.4 um IR) via NOAA S3 (anonymous)
        |       Falls back to JAXA P-Tree if S3 unavailable
        |       - Downloads 3 segments covering VOBL's 50km radius box
        |       - Parsed with satpy (ahi_hsd reader) into lat/lon BT grid
        |       - Computes: min_bt_50km, cold_pixels_count, storm_detected,
        |                   nearest_pixel_dist_km, vobl_bt_celsius
        |       - bt_trend_1h: compares current frame to the frame ~60 min ago
        |         from 6-frame rolling history (negative = anvil cooling)
        |       Output: himawari_realtime.json, himawari_history.json (6 frames)
        |
        +-- forecast_action.py
        |       XGBoost inference, all downstream computation
        |       - Monsoon regime detection (R1-R5 rule-based)
        |       - Regime-aware threshold adjustment per slot
        |       - CAPE tendency from gfs_history_43295.json (dCAPE/dt in J/kg/h)
        |       - 84-feature vector: base obs + 30 temporal lags + derived fields
        |       - Historical analog search on 2015-2025 training data
        |       - Convective initiation score (CAPE + KI + LI + TT composite)
        |       - 48h multi-day outlook with per-day instability scores
        |       - Airport impact: disrupted departures estimate per slot
        |       - Pipeline health tracker (data freshness per source)
        |       - METAR TS override: if live METAR confirms thunderstorm,
        |         forces alert_active=True and floors slot probabilities to 0.85
        |       - SIGMET bulletin: auto-generates ICAO-format advisory text
        |         when any slot predicts TS (LIGHT/MODERATE/SEVERE by probability)
        |       Output: forecast.json
        |
        +-- fetch_metar.py
        |       aviationweather.gov JSON API (VOBL with VOBG fallback)
        |       Parses: T, Td, RH, wind, visibility, sky cover, TS flag
        |       Injected into forecast.json under "metar" key
        |
        +-- compute_realtime_shap.py
        |       SHAP TreeExplainer on the production model for each slot
        |       Top 12 features by absolute SHAP value
        |       Output: data/realtime_shap.json
        |
        +-- verify_today.py
        |       Compares previous-day forecast against IMD VOBL observations
        |       Computes rolling 30-day WMO metrics: POD, FAR, CSI, HSS, Brier
        |       Output: data/verification_today.json
        |
        +-- populate_skill_scores.py
        |       Aggregates verification history into structured skill score output
        |       Falls back through: forecast_log.csv -> verification_report.json
        |       -> verification_today.json in priority order
        |       Output: data/skill_scores.json
        |
        +-- forecast.json (committed back to main)
                Cloudflare Pages auto-deploys on every commit
                Dashboard fetches this file directly (no backend needed)
```

**Pipeline run time:** approximately 90 seconds end-to-end. The Himawari step takes the longest due to downloading three satellite segments from NOAA S3.

---

## Cron Schedule

Five GitHub Actions crons per day. The GFS cycle selection follows the **t+12 rule**: the cycle fetched is 12 hours before the slot valid time, ensuring a ~6.5 hour posting buffer before the window opens.

| UTC Cron | IST Time | Purpose | GFS Cycle |
|:--------:|:--------:|:-------:|:---------:|
| `45 17 * * *` | 23:15 IST | Slot 0 forecast | prev-day 06Z |
| `45 23 * * *` | 05:15 IST | Slot 1 forecast | prev-day 12Z |
| `45  5 * * *` | 11:15 IST | Slot 2 forecast (primary) | prev-day 18Z |
| `15 10 * * *` | 15:45 IST | Slot 3 forecast | same-day 00Z |
| `15 16 * * *` | 21:45 IST | Dashboard refresh | same-day 00Z |

---

## Features and Model Versions

### Feature Engineering

The v5/v6 models use 84 features built from four categories:

**Base meteorological features (54)**
- GFS surface: CAPE, CIN, K-Index, LI, Total Totals, PW, T2m, Td2m
- ERA5-derived multi-level fields: T/q/u/v at 500, 700, 850 hPa
- Daily IMD obs: Tmax, Tmin, rainfall, sunshine hours, evaporation

**Temporal lag features (30)**
- 1-day and 3-day lags on Tmax, Tmin, rainfall, storm label
- Rolling means: RF_3d, RF_7d, MAX_3d_avg, DTR_3d_avg
- LABEL_lag1: whether a storm occurred in the same slot yesterday

**Derived thermodynamic features (10)**
- `cape_x_kindex`: product of CAPE and K-Index (best single predictor in SHAP)
- `thetae_850`: equivalent potential temperature at 850 hPa
- `wind_shear_500_850`, `wind_shear_700_850`: vertical shear vectors
- `moisture_flux_850/700`: |wind| * specific humidity
- `q_gradient_500_850`: moisture availability at mid-levels
- `thickness_500_850`: thermal thickness proxy for lapse rate

**Cyclic time encodings**
- `MONTH_sin/cos`, `DOY_sin/cos`, `slot_sin/cos`: prevents discontinuities at month/year boundaries
- `slot_month_clim`: per-slot monthly climatological storm rate (base rate anchor)

### Model Version History

| Version | Training Data | Key Features | Notes |
|---------|:-------------:|:------------:|-------|
| v2 | 2015-2022 | 54 base features | Baseline calibrated |
| v3 | 2015-2022 | 54 + calibration | Isotonic calibration |
| v4 Ensemble | 2015-2023 | 54 + ensemble stacking | 3-model ensemble for Slot 0 |
| **v5 Temporal** | **2015-2023** | **84 (+ 30 lag features)** | **Slots 1-3 production** |
| **v6 Temporal** | **2015-2024** | **84 + 10 new derived** | **Slot 2 production, A100 trained** |

**v6 training details:**
- 100 Optuna trials per slot on A100 GPU (`tree_method="hist", device="cuda"`)
- Walk-forward cross-validation: train on years < test_year, test on test_year
- F-beta=1.5 threshold tuning: recall weighted 1.5x over precision (missed storm is a safety event)
- October-specific class weight x2 on positive samples for Slot 2
- Models saved as both `.pkl` (joblib) and `.ubj` (XGBoost Booster, version-stable binary)

**Model fallback chain per slot:**
```
v6 Himawari > v6 Temporal > v5 Temporal > v4 Ensemble/Calibrated > v3 Calibrated > v2 Calibrated > Climatology
```

---

## Real-Time Intelligence Features

Beyond a simple model prediction, each pipeline run now generates several operational intelligence layers:

**Brightness Temperature Trend (`bt_trend_1h`)**  
Compares the current Himawari frame's minimum BT within 50km of VOBL against the frame from approximately 60 minutes prior. A negative trend (cooling) indicates an anvil growing toward the airport, typically a 30-60 minute precursor to surface thunderstorm activity. Available in `forecast.json` under `satellite.himawari9.bt_trend_1h`.

**CAPE Tendency**  
Computes the rate of change of CAPE between the last two GFS cycles stored in `gfs_history_43295.json`. Units are J/kg/h. A positive tendency above +50 J/kg/h is flagged as BUILDING in the convective initiation section and also sets `INTSF` (intensifying) in the SIGMET bulletin.

**Monsoon Regime Detection**  
Classifies the current synoptic environment into one of five regimes using rule-based logic on real-time CAPE, K-Index, T2m, and month:

| Regime | Condition | Threshold Factor | Interpretation |
|:------:|:---------:|:----------------:|:--------------|
| CONVECTIVE_BURST | KI >= 38, CAPE >= 800 | 0.80 | Peak thunderstorm environment |
| ACTIVE | KI >= 35, CAPE >= 300, monsoon months | 0.88 | Active monsoon westerlies |
| ACTIVE_MODERATE | KI >= 32, CAPE >= 100, T2m >= 28 | 0.95 | Moderate instability |
| NEUTRAL | Default | 1.00 | No adjustment |
| BREAK | KI < 30, CAPE < 100, monsoon months | 1.30 | Suppress false alarms |

**METAR Thunderstorm Override**  
If the live METAR for VOBL reports an active thunderstorm (`TS` in wx_string), the pipeline forces `alert_active=True` regardless of model output and sets all slot probabilities to a minimum of 0.85. This handles the case where a storm has already initiated but the model has not yet updated.

**SIGMET Bulletin Generation**  
When any slot exceeds its prediction threshold, the pipeline auto-generates an ICAO-format SIGMET advisory text under `forecast.json["sigmet_bulletin"]`. Intensity is LIGHT/MODERATE/SEVERE based on peak probability, and the INTSF flag fires when CAPE tendency is positive. Clearly marked as advisory-only and not for operational use without meteorologist review.

---

## Verification and Skill Scores

Verification runs daily via `verify_today.py` comparing the previous day's forecasts against IMD VOBL surface observations. Rolling 30-day metrics are written to `forecast.json` and displayed on the dashboard.

| Metric | Formula | What it measures | Target |
|--------|:-------:|:----------------:|:------:|
| POD | TP / (TP + FN) | Fraction of actual storms that were predicted | >= 0.70 |
| FAR | FP / (TP + FP) | Fraction of predicted storms that did not occur | <= 0.35 |
| CSI | TP / (TP + FP + FN) | Combined hit/miss/false-alarm score | >= 0.40 |
| HSS | 2(TP*TN - FP*FN) / [...] | Skill relative to random chance (-1 to +1) | >= 0.45 |
| Brier Score | mean((P - O)^2) | Probabilistic accuracy | <= 0.08 |
| BSS | 1 - Brier/Brier_clim | Skill relative to climatological base rate | >= 0.10 |

The `populate_skill_scores.py` script aggregates these into `data/skill_scores.json` with per-slot breakdown and weighted summary statistics.

---

## Data Sources

| Source | Variables | Update Lag | Auth |
|--------|-----------|:----------:|:----:|
| NOAA NOMADS (GFS 0.25 deg) | CAPE, CIN, KI, LI, TT, PW, T/q/u/v profile, TMP f006-f048 | ~4h after cycle | None (Chrome UA required) |
| Himawari-9 via NOAA S3 | Band 13 BT at 10.4 um, 3 segments, 50km box | ~10 min | None (anonymous S3) |
| JAXA P-Tree | Same as above | ~15 min | None (HTTP) |
| aviationweather.gov | METAR: T, Td, wind, vis, ceiling, TS flag | ~1h | None |
| IMD VOBL surface obs | Daily thunderstorm occurrence (TH flag) | Daily | Institutional |

**GFS fetch note:** NOAA NOMADS returns 403 on bare Python requests. A Chrome User-Agent header is required. All GFS paths use UTC dates only; IST conversion happens only for display.

---

## Key Files

```
forecast_action.py              Main pipeline: inference, all downstream sections
gfs_fetcher.py                  GFS NOMADS fetcher (surface + profile + multi-hour TMP + outlook)
fetch_himawari_realtime.py      Himawari-9 BT via satpy (NOAA S3 -> JAXA fallback)
fetch_metar.py                  METAR from aviationweather.gov (VOBL + VOBG fallback)
compute_realtime_shap.py        SHAP TreeExplainer, top 12 features per slot
verify_today.py                 Daily IMD verification (WMO metrics, 30-day rolling)
populate_skill_scores.py        Aggregates skill scores from forecast log into skill_scores.json
resave_models.py                XGBoost version-agnostic model resave using Booster.save_model
train_v6_slot_models.py         v6 training script (A100/GPU, Optuna, walk-forward CV)
backtest_himawari.py            Historical Himawari BT retrieval for v6 training data
october_threshold_fix.py        October Slot 2 threshold override (DOY_sin artefact)
index.html                      Dashboard (React + Tailwind, fetches forecast.json directly)

models/
  nowcast_slot*_xgb_v6_temporal.pkl      v6 production models (4 slots)
  nowcast_slot*_xgb_v5_temporal.pkl      v5 temporal, 30 lag features
  nowcast_slot0_xgb_v4_ensemble.pkl      v4 ensemble (Slot 0 only)
  nowcast_slot*_xgb_v3_calibrated.pkl    v3 fallback
  boosters/nowcast_slot*_v6.ubj          XGBoost Booster binary (version-stable)

data/
  gfs_realtime_43295.csv                 Today's GFS surface + upper-air
  gfs_history_43295.json                 Per-cycle GFS history (CAPE tendency source)
  himawari_realtime.json                 Latest Himawari frame + bt_trend_1h
  himawari_history.json                  Last 6 Himawari frames
  realtime_shap.json                     SHAP values per slot
  verification_today.json                Daily WMO metrics
  skill_scores.json                      Aggregated skill scores
  pipeline_health.json                   Data freshness per source

forecast.json                            Live output committed by GitHub Actions
.github/workflows/forecast_update.yml   5-cron CI/CD pipeline
```

---

## forecast.json Schema

The full output written after each pipeline run. The dashboard reads this file directly.

```json
{
  "date": "2026-08-11",
  "generated_at": "2026-08-11 11:15 IST",
  "alert_active": true,
  "peak_slot": 2,
  "peak_probability": 0.41,
  "model_version": "v6_temporal_v5_temporal_v4_ensemble",
  "slots": [
    {
      "slot": 2,
      "label": "Afternoon",
      "time": "1201-1800 IST",
      "ts_probability": 0.41,
      "ts_predicted": true,
      "threshold": 0.14,
      "primary": true,
      "source": "gfs+upperair",
      "model_used": "nowcast_slot2_xgb_v6_temporal.pkl",
      "model_version": "v6_temporal",
      "raw_probability": 0.39,
      "cape": 1240.5,
      "k_index": 36.2,
      "lifted_index": -2.4,
      "totals_totals": 48.1,
      "trend": "up",
      "trend_diff": 0.045,
      "prev_probability": 0.365,
      "prob_delta": 0.045,
      "regime_adjustment": 0.88
    }
  ],
  "met_parameters": {
    "ua_cape_jkg": 1240.5,
    "ua_k_index": 36.2,
    "ua_lifted_index": -2.4,
    "ua_totals_totals": 48.1,
    "ERA5_u_500hPa": 5.2,
    "ERA5_u_850hPa": -3.1,
    "instability_level": "Moderate"
  },
  "satellite": {
    "himawari9": {
      "min_bt_50km": -52.3,
      "vobl_bt_celsius": -18.4,
      "cold_pixels_count": 47,
      "storm_detected": true,
      "nearest_pixel_dist_km": 31.2,
      "bt_trend_1h": -8.4,
      "alert_level": "ORANGE",
      "available": true
    },
    "history": ["...last 6 frames..."]
  },
  "convective_initiation": {
    "instability_score": 67.4,
    "initiation_risk": "MODERATE",
    "cape_tendency_jkgh": 120.5,
    "cape_trend": "BUILDING",
    "monsoon_regime": "ACTIVE",
    "regime_thresh_factor": 0.88,
    "hours_to_peak": 1.5,
    "peak_window_ist": "1300-1800 IST"
  },
  "sigmet_bulletin": "VCBB SIGMET X01 VALID 1115/1715 UTC 11/AUG/2026\nVOBL FIR THUNDERSTORM FCST\nTS MODERATE OBS/FCST AT 1115 UTC\nTOP FL350 CB\nMOV NE 10KT\nINTENSITY INTSF\nFCST AT 1715 UTC TS MODERATE STNR\nNC=",
  "metar_ts_override": false,
  "multiday_outlook": ["...48h per-day outlook..."],
  "analogs": {
    "top_5": ["...5 most similar historical days..."],
    "ts_rate": 0.6
  },
  "airport_impact": {
    "total_departures_today": 143,
    "total_disrupted_est": 12,
    "overall_risk": "MODERATE"
  },
  "synoptic_regime": {
    "regime_id": "R2",
    "regime_name": "Moist Monsoon",
    "ts_rate": 9.3,
    "auroc": 0.934
  },
  "verification": {
    "pod": 0.714,
    "far": 0.286,
    "hss": 0.523,
    "brier": 0.0812,
    "window": "30-day"
  },
  "realtime_shap": {
    "2": {
      "top_features": [
        {"feature": "cape_x_kindex", "shap": 1.24, "direction": "increases_risk"},
        "..."
      ]
    }
  },
  "pipeline_health": {
    "components": {
      "gfs": {"status": "OK", "cycle": "18Z", "staleness": "FRESH"},
      "himawari9": {"status": "OK", "storm_detected": true, "staleness": "FRESH"},
      "metar": {"status": "OK"}
    }
  },
  "gfs_tmax_c": 31.4,
  "gfs_tmin_c": 21.8,
  "gfs_rainfall_mm": 4.2
}
```

---

## Deployment

### Normal push
```bash
git pull origin main
git add <files>
git commit -m "type: description"
git push origin main --force-with-lease
```

### forecast.json merge conflict (race with Actions)
```bash
git pull origin main --no-rebase
git checkout --theirs forecast.json   # auto-generated, never edit manually
git add forecast.json
git commit -m "merge: resolve forecast.json conflict"
git push origin main --force-with-lease
```

**Never use bare `--force`.** Always `--force-with-lease`.

### Trigger a manual pipeline run
GitHub repo -> Actions -> "Update Forecast JSON" -> Run workflow

### Cloudflare Pages
- Project: `csir-thunderstorm-bengaluru`
- Deploys automatically after every commit to `main` via `cloudflare/wrangler-action@v3`
- The dashboard is a static HTML file that fetches `forecast.json` from the raw GitHub URL

### Model resave after retraining
After any retraining on Colab or the A100, run:
```bash
python resave_models.py --all
```
This saves each model using `Booster.save_model()` into `.ubj` (XGBoost binary JSON) format, which is stable across XGBoost versions. The old `.pkl` files remain for backward compatibility with the fallback chain.

---

## Local Development

```bash
git clone https://github.com/Aprameya05/csir-thunderstorm-bengaluru.git
cd csir-thunderstorm-bengaluru

pip install -r requirements.txt

# Fetch today's GFS data
python gfs_fetcher.py

# Fetch latest Himawari scene
python fetch_himawari_realtime.py

# Run the forecast pipeline
python forecast_action.py

# Compute SHAP explanations
python compute_realtime_shap.py

# Aggregate skill scores
python populate_skill_scores.py

# Open index.html in a browser to see the dashboard
```

---

## Free-Tier Stack

The entire system runs at zero cost.

| Service | Use | Cost |
|---------|-----|:----:|
| NOAA NOMADS | GFS 0.25 deg GRIB2 | Free |
| NOAA AWS S3 | Himawari-9 HSD files (anonymous) | Free |
| JAXA P-Tree | Himawari fallback (HTTP) | Free |
| aviationweather.gov | METAR API | Free |
| Cloudflare Pages | Dashboard hosting | Free |
| GitHub Actions | CI/CD (public repo) | Free |
| Render | FastAPI backend | Free (cold start ~30s) |
| Groq (Llama-3.3-70b) | RAG-based met explanation | Free tier |

---

## Key Engineering Decisions

**Why XGBoost over deep learning?**  
LSTM and CNN architectures were tested and achieved around 0.79 AUROC. XGBoost at 0.871 AUROC outperformed them across all seasons. The dataset has approximately 3,800 days of training data with a 6-8% positive rate, which is too small to train deep models effectively.

**Why walk-forward validation?**  
Random k-fold cross-validation leaks future information through the lag features (RF_lag1, LABEL_lag1 etc.), inflating apparent AUROC by 4-6 points. Walk-forward CV trains on all years before the test year and evaluates forward, which is how the model will actually be used.

**Why F-beta=1.5 threshold tuning?**  
A missed thunderstorm at an airport is a safety event. A false alarm costs delay time and fuel. The F-beta=1.5 criterion weights recall 1.5x over precision in threshold selection, deliberately accepting higher FAR to improve POD.

**Why not retrain on ERA5 reanalysis?**  
ERA5 data has higher quality than real-time GFS, which creates a train/serve skew. GFS-based training means the model has seen the same artifacts (bias in CAPE, coarser profile resolution) it will encounter at inference time.

**Why a static dashboard with no backend?**  
The entire dashboard state lives in `forecast.json` which is committed to the repo after every pipeline run. Cloudflare Pages serves it as a static file. There is no database, no backend API required for the dashboard, and no infrastructure to maintain. The Render FastAPI is an optional layer for programmatic access only.

---

## Things to Know

- GFS NOMADS requires a Chrome User-Agent header. Bare Python requests return 403.
- All GFS date paths are in UTC. Never use IST dates in NOMADS URLs.
- `satpy` requires the `ahi_hsd` reader for Himawari HSD files.
- The v6 models were trained with DataFrame input (not numpy arrays) to preserve named feature columns in XGBoost. SHAP feature names will show real column names, not f0/f54 etc.
- `forecast.json` is auto-generated. Never manually edit it. Merge conflicts should always resolve with `git checkout --theirs forecast.json`.
- The `gfs_history_43295.json` file grows by one entry per pipeline run and is used for CAPE tendency computation. It is not trimmed automatically.
- SHAP values are computed using TreeExplainer, not KernelExplainer. This is fast enough for GitHub Actions (under 5 seconds per slot) and gives exact Shapley values for tree models.

---

## Team

| Member | Role |
|--------|------|
| **Aprameya** | ML Lead, project architect |
| **Atul Denny** | Upper-air data, GFS pipeline, Himawari satellite fetching |
| **Satvik** | FastAPI backend on Render |
| **Vidhi** | ERA5 historical data |
| **Sneha** | Verification pipeline, visualisation |

**Institutional collaborator:** Dr. Geeta Agnihotri, Scientist F, IMD Bengaluru. Agreed to share 2026 VOBL raw surface observation data and is a candidate co-author on a joint technical report.

---

## What Is Next

**Near-term:**
- Himawari backtest (`python backtest_himawari.py --per-slot --start 2015-07-01`) to generate per-slot BT training data for proper v6 Himawari model training. Currently the v6 Himawari models use a limited BT feature set.
- Re-calibrate v5 models on 2024-2025 data using isotonic calibration without full retraining.
- Integrate Dr. Agnihotri's 2026 raw VOBL surface data when received.
- Add Damini lightning network feed (pending IMD agreement).

**Medium-term:**
- API rate limiting and authentication for IMD production deployment.
- Automated retraining trigger when new IMD annual data arrives.
- Mobile-responsive dashboard audit.

---

*This is a production system, not a research prototype. Everything described above runs today.*
