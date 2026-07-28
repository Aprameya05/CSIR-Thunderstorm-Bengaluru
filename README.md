<div align="center">

<img src="https://img.shields.io/badge/CSIR-Research%20Project-blue?style=for-the-badge" />

# ⛈️ CSIR Thunderstorm Prediction System

### AI-Powered Operational Thunderstorm Nowcasting for Bengaluru Airport
#### IMD Station 43295 — Kempegowda International Airport (VOBL)

*Developed in collaboration with Dr. Geeta Agnihotri, Scientist F, IMD Bengaluru*

<p>
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-v5%20Temporal%20Ensemble-success" />
  <img src="https://img.shields.io/badge/LSTM-Bidirectional%20%2B%20Attention-purple" />
  <img src="https://img.shields.io/badge/A100-GPU%20Trained-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Public%20API-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Cloudflare%20Pages-Live-F38020?logo=cloudflare&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Himawari--9-Satellite-blueviolet" />
  <img src="https://img.shields.io/badge/Blitzortung-Live%20Lightning-yellow" />
  <img src="https://img.shields.io/badge/Llama--3.3--70b-RAG%20Explainability-orange" />
  <img src="https://img.shields.io/badge/SHAP-Real--Time-red" />
  <img src="https://img.shields.io/badge/Status-Operational-brightgreen" />
</p>

---

### 🌐 [Live Dashboard](https://csir-thunderstorm-bengaluru.pages.dev) &nbsp;|&nbsp; 🔗 [Live API](https://csir-thunderstorm-api.onrender.com) &nbsp;|&nbsp; 📖 [API Docs](https://csir-thunderstorm-api.onrender.com/docs)

</div>

---

## What This System Does

The CSIR Thunderstorm Prediction System is a fully operational AI nowcasting platform that predicts thunderstorm probability over Bengaluru Airport across four 6-hour windows every day. It ingests real-time atmospheric data from multiple sources — GFS NOMADS, Himawari-9 satellite, live METAR, and upper-air soundings — runs a multi-model ensemble trained on an NVIDIA A100 GPU, applies a satellite-driven correction layer, computes real-time SHAP explanations, generates physical explanations using Llama-3.3-70b, and serves everything through a live public dashboard with automatic CI/CD deployment. Twice daily, with zero human intervention.

---

## System Capabilities

### 🎯 6-Hour Nowcasting (4 Slots) — v5 Best-Per-Slot Ensemble

Four models output thunderstorm probabilities for distinct 6-hour windows, each selected as the empirically best-performing model on the 2024–2025 test set:

| Slot | IST Window | Description | Model | AUROC | Threshold | Calibration |
|------|-----------|-------------|-------|:-----:|:---------:|:-----------:|
| 0 | 0001–0600 | Late Night | v4 Ensemble | 0.8484 | 0.15 | Isotonic |
| 1 | 0601–1200 | Morning | **v5 Temporal** | **0.8317** | **0.15** | **Isotonic** |
| **2** | **1201–1800** | **Afternoon (peak)** | **v3 Calibrated** | **0.8710** | **0.226** | **Isotonic** |
| 3 | 1801–2400 | Evening | v3 Calibrated | 0.8710 | 0.163 | Isotonic |

Thresholds are F-beta (β=2) optimized on out-of-fold predictions, penalizing misses 2x more than false alarms — operationally correct for airport safety. Slot 3 POD improved from 47.3% to 65.5% with optimized thresholds.

### 🤖 A100-Trained Multi-Model Ensemble (v4)

All ensemble models were trained on an NVIDIA A100-SXM4-40GB GPU (42.4 GB VRAM) via Google Colab Pro using Optuna hyperparameter optimization:

| Model | CV AUROC | Test AUROC | Training |
|-------|:--------:|:----------:|---------|
| XGBoost (Optuna, 20 seeds) | 0.9243 | 0.8598 | 100 Optuna trials, GPU |
| Random Forest (10 seeds) | — | 0.8653 | CPU, n_jobs=-1 |
| Gradient Boosting | — | 0.8610 | sklearn |
| LSTM (Bidirectional + Attention) | — | 0.8218 | A100 GPU, 100 epochs |
| **MLP Meta-Ensemble** | — | **0.8791** | OOF stacking |

The meta-learner blends all four models using out-of-fold predictions to avoid leakage. LSTM weight (3.165) exceeds XGBoost (2.456) in the meta-learner — the LSTM captures different storm patterns, particularly temporal sequences.

### 🌊 v5 Temporal Models — Atmospheric Lag Features

The v5 models add 30 atmospheric lag and rolling features to the base 54, giving 84 features total. These are leak-free — only past atmospheric states, not past labels:

| Feature Group | Features | Window |
|---|---|---|
| CAPE rolling mean | CAPE_roll3/7/14 | 3, 7, 14 days |
| K-Index rolling mean | KI_roll3/7/14 | 3, 7, 14 days |
| PWAT rolling mean | PWAT_roll3/7/14 | 3, 7, 14 days |
| Wind shear rolling mean | shear_roll3/7/14 | 3, 7, 14 days |
| CAPE lags | CAPE_lag1/2/3 | t−1, t−2, t−3 days |
| K-Index lags | KI_lag1/2/3 | t−1, t−2, t−3 days |
| LI lags | LI_lag1/2/3 | t−1, t−2, t−3 days |
| PWAT lags | PWAT_lag1/2/3 | t−1, t−2, t−3 days |
| Shear lags | shear_lag1/2/3 | t−1, t−2, t−3 days |
| Trend features | CAPE_trend3, KI_trend3, PWAT_trend3 | vs 3-day mean |

Slot 1 (Morning) improved by +3pp with temporal features — morning storms are driven by antecedent moisture accumulation over multiple days, exactly what rolling PWAT and KI lags capture.

### 🛰️ Himawari-9 Real-Time Satellite Override (Trained Correction Model)

When the Himawari-9 satellite detects deep convection near VOBL, a trained correction model (Logistic Regression + Isotonic Calibration, CV AUROC 0.8812) adjusts slot probabilities upward. This corrects for GFS model lag — the GFS is yesterday's forecast while Himawari is real-time.

**Trigger conditions (all must be met):**
- `storm_detected = True`
- `min_bt_50km < −45°C` (deep convective cloud tops)
- `cold_pixels_count ≥ 50` (widespread convection)
- `nearest_pixel_dist_km < 100 km` (storm within range of VOBL)

**Correction model training:** Logistic Regression trained on 2015–2025 ERA5+IMD dataset (3,819 samples, 457 storm events) using atmospheric proxies for satellite cold-top signal. CV AUROC 0.8812 with 4.1x discrimination ratio (storm days vs clear days).

When triggered, affected slots show a **SAT⚡** badge and "HIMAWARI BOOST" label. The dashboard STATUS tile flashes **⚠ ALERT ACTIVE** with a **SAT OVERRIDE** badge. The airport impact score cascades automatically from the boosted probability.

### 🌩️ Live Lightning Strike Feed (Blitzortung WebSocket)

A WebSocket proxy on the Render API (`/ws/lightning`) connects to Blitzortung's global lightning network and forwards strikes within 250 km of VOBL to the dashboard over `wss://`. The Radar Map page shows:

- **Strikes (last hour)** count
- **Rate (strikes/min)** over last 5 minutes
- **Nearest strike distance** (km from VOBL)
- **Activity level:** QUIET → MODERATE → ACTIVE → SEVERE
- Animated strike dots on the SVG radar map: yellow (>100 km), orange (50–100 km), red (<50 km)

### 📈 Trend Arrows & Probability Comparison
Every slot card shows a trend arrow (↑ ↓ →) and percentage change vs the previous forecast run, computed from `forecast_log.csv`. Rising probabilities are flagged in red, falling in green.

### ⚡ Convective Initiation Timer
A real-time instability score (0–100) computed from CAPE, K-Index, Lifted Index, and Totals-Totals. Shows hours to peak convective window (1300–1800 IST), current atmospheric state, and risk level.

### 📅 Multi-Day Extended Outlook
GFS f024 and f048 forecasts provide tomorrow and day-after probabilistic outlooks — CAPE, K-Index, LI, and estimated Slot 2 TS probability.

### 🔴 Real-Time SHAP Explainability
`compute_realtime_shap.py` runs on every pipeline cycle, computing SHAP values from today's actual GFS inputs. The Explainability page shows feature contributions per slot with direction, magnitude, and actual input values.

### 📐 Skew-T Log-P Sounding Diagram
The Explainability page renders a live Skew-T Log-P atmospheric sounding diagram fetched from the University of Wyoming radiosonde archive (Station 43295). When the radiosonde is unavailable, a synthetic sounding is constructed from GFS met parameters. Shows isotherms, isobars, dry adiabats, temperature profile (red), dewpoint profile (green), LCL and LFC markers, and a level table with 850/500 hPa temperatures and lapse rate.

### 🛰️ Himawari-9 Satellite BT Visualization
`fetch_himawari_realtime.py` pulls Band 13 (10.4 μm clean IR window) from NOAA AWS S3. The Radar Map renders real cloud-top brightness temperatures as a color overlay with a 6-frame timeline scrubber (1 hour of history at 10-min intervals).

### 🌪️ Instability Composite Gauge
Dashboard Tile 7 shows a CAPE × K-Index composite (CAPE×KI) as a half-circle arc gauge with four severity levels: MINIMAL (<2,000) → WEAK (2,000–7,500) → MODERATE (7,500–17,500) → SEVERE (>17,500). Also displays CAPE (J/kg), K-Index, and Lifted Index from real GFS/upper-air data.

### 🤖 RAG Explainability Engine (Llama-3.3-70b)
The `/rag/explain` endpoint answers natural language questions grounded in today's real upper-air values. The `/rag/analogs` endpoint retrieves historically similar days from the 2015–2025 training record. The Live API page has functional **▶ Try it** buttons with a live elapsed timer and wake-up UX for Render cold starts.

### 🌤️ Live METAR Integration
`fetch_metar.py` pulls live METAR from `aviationweather.gov` for VOBL every pipeline cycle. No API key required. The dashboard LIVE WEATHER tile and ATC View display real temperature, wind direction/speed, relative humidity (Magnus formula), visibility, flight category (VFR/MVFR/IFR/LIFR), sky cover layers, and thunderstorm present flag. Footer shows "VOBL MVFR · METAR" with actual flight category.

### 🗺️ Synoptic Regime Auto-Detection
KMeans 5-cluster classification of the current atmospheric pattern from GFS fields. The Regimes page highlights today's detected regime with a glowing cyan border and **TODAY ◉** badge, plus a summary banner showing AUROC, TS rate, CAPE, and K-Index for the detected regime.

### ✅ Automated Daily Verification Pipeline
`verify_today.py` runs in the GitHub Action after each forecast cycle. Reads the IMD Table-II TH flag for yesterday's date, determines actual slot outcomes (TH=1 → slots 2/3 actual=1), and writes rolling 30-day POD, FAR, HSS, and Brier Score to `forecast.json` and `data/verification_today.json`. Fails gracefully when IMD data or forecasts are unavailable (exit code 0).

### ⚙️ CI/CD Pipeline (GitHub Actions — ~58 seconds end-to-end)
Two scheduled runs at **15:45 IST** and **21:45 IST**:

```
cron trigger (15:45 IST / 21:45 IST)
→ fetch_himawari_realtime.py    (Himawari-9 Band 13 BT from NOAA S3 — runs FIRST)
→ gfs_fetcher.py                (auto-discovers latest GFS cycle, fetches f012/f024/f048)
→ forecast_action.py            (v5 ensemble inference + Himawari correction + convective timer)
→ fetch_metar.py                (live METAR from aviationweather.gov — VOBL)
→ compute_realtime_shap.py      (per-slot SHAP from today's actual GFS inputs)
→ verify_today.py               (IMD TH-flag verification + rolling POD/FAR/HSS/Brier)
→ forecast.json committed to GitHub
→ Cloudflare Pages auto-deploy (~58s end-to-end)
→ Live website updated
```

Himawari runs before `forecast_action.py` so the correction model can read fresh BT data and apply satellite override if conditions are met.

### 🔌 Public REST API (FastAPI on Render)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + model version |
| POST | `/predict` | Daily thunderstorm prediction |
| GET | `/nowcast/slots/info` | Slot metadata + thresholds |
| POST | `/nowcast/predict/slot/{id}` | Single slot prediction |
| POST | `/nowcast/predict/all` | All 4 slots batch |
| POST | `/rag/explain` | Llama-3.3-70b physical explanation |
| POST | `/rag/analogs` | Historical analog retrieval |
| GET | `/correction/status` | Correction model + Himawari override state |
| GET | `/nowcast/explain/{slot_id}` | Plain-English slot explanation (no LLM) |
| WS | `/ws/lightning` | Real-time Blitzortung lightning proxy (wss://) |

---

## System Architecture

<p align="center">
  <img src="assets/system_architecture_v3.png"
       alt="CSIR Thunderstorm Prediction System Architecture"
       width="100%">
</p>

---

## Model Performance

### Per-Slot Best Model (Production)

| Slot | Model Version | AUROC | POD | FAR | Threshold | Brier |
|------|:-------------:|:-----:|----:|----:|:---------:|------:|
| 0 (Late Night) | v4 Ensemble | 0.8484 | 0.235 | 0.875 | 0.15 | 0.0243 |
| 1 (Morning) | v5 Temporal | 0.8317 | 0.062 | 0.875 | 0.15 | 0.0225 |
| **2 (Afternoon)** | **v3 Calibrated** | **0.8710** | **0.473** | **0.671** | **0.226** | **0.060** |
| 3 (Evening) | v3 Calibrated | 0.8710 | 0.655 | 0.679 | 0.163 | — |

Note: Slot 3 POD improved from 0.473 to 0.655 after threshold optimization (0.30→0.163).

### Daily Meta-Ensemble (Overall System)

| Component | Test AUROC | Brier Score |
|-----------|:----------:|:-----------:|
| XGBoost Ensemble (20 seeds, Optuna) | 0.8598 | 0.0875 |
| Random Forest Ensemble (10 seeds) | 0.8653 | 0.0703 |
| Gradient Boosting | 0.8610 | — |
| LSTM (Bidirectional + Attention) | 0.8218 | 0.1647 |
| **MLP Meta-Ensemble (OOF stacking)** | **0.8791** | **0.0597** |
| Production v3 baseline | 0.8710 | ~0.080 |
| **Improvement over v3** | **+0.81pp** | **−25%** |

### Model Evolution

| Version | Architecture | Key Change | Best AUROC |
|---------|-------------|-----------|:----------:|
| v1 | Single XGBoost | Daily ERA5 features | 0.833 |
| v2 | Single XGBoost | 6-hourly ERA5 (Vidhi) | 0.821 |
| v3 | XGBoost + Isotonic calibration | 10 derived interaction features | **0.871** |
| v4 | XGBoost ensemble (20 seeds) | A100 Optuna, RF+GB+LSTM meta | 0.8791 |
| **v5** | **Temporal XGBoost** | **30 atmospheric lag features (84 total)** | **0.8659** |

### Correction Model (Himawari Override)

| Metric | Value |
|--------|-------|
| Model type | Logistic Regression + Isotonic Calibration |
| Training data | 2015–2025 ERA5+IMD (3,819 samples) |
| CV AUROC (5-fold) | 0.8812 ± 0.0143 |
| Full AUROC | 0.8860 |
| Brier Score | 0.0781 |
| Discrimination ratio | 4.1× (storm days vs clear days) |
| Mean prob (storm days) | 0.3708 |
| Mean prob (clear days) | 0.0904 |

### Key Research Findings
- **ERA5_CAPE rank 42→2** for Slot 3 with 6-hourly ERA5 (v1→v2)
- **`cape_x_kindex`** interaction term entered top-5 SHAP in 3 of 4 slots (v3)
- **October accounts for 32%** of all Slot 2 misses — thermodynamic regime with lower CAPE baseline; specialist model tested but general v5 outperformed it (0.8043 vs 0.7664) — general model handles October correctly via seasonal encodings
- **Slot 3 threshold optimization:** POD 47.3%→65.5% (+18.2pp) with F-beta(2) tuned threshold
- **LSTM meta-weight 3.165 > XGBoost 2.456** — LSTM captures different storm patterns despite lower individual AUROC
- **Leakage detection:** v5 with label-derived temporal features showed AUROC=1.0 (caught and corrected — atmospheric-only lags used)
- **v5 temporal Slot 1 improvement: +3pp** — morning storms driven by antecedent moisture accumulation over multiple days

---

## SHAP Feature Importance (Real-Time, per Slot)

| Rank | Slot 0 (Night) | Slot 1 (Morning) | Slot 2 (Afternoon) | Slot 3 (Evening) |
|------|---------------|-----------------|-------------------|-----------------| 
| 1 | ERA5_u_850hPa | DRNRF | K_INDEX | thetae_850 |
| 2 | LABEL_lag1 | CAPE | cape_x_kindex | ERA5_CAPE |
| 3 | LIFTED_INDEX | ERA5_q_700hPa | LIFTED_INDEX | K_INDEX |

*Updated every pipeline run from today's actual GFS inputs.*

**Physical interpretation:**
- **Slot 0 (Night):** Wind-driven — 850 hPa easterly flow brings moisture; yesterday's storm (LABEL_lag1) strongly predictive
- **Slot 1 (Morning):** DRNRF (dry/rain flag) + mid-level moisture dominate; antecedent conditions matter most
- **Slot 2 (Afternoon):** Purely thermodynamic — K-Index and CAPE×K-Index interaction drive afternoon peak
- **Slot 3 (Evening):** Moist static energy (theta-e) and CAPE dominate — deep moist convection signature

---

## Feature Engineering (54 Base + 30 Temporal = 84 v5 Features)

| Category | Features | Count |
|----------|----------|------:|
| Surface Variables | MAX, MIN, DTR, AW, RF, EVP, DRNRF, SSH | 8 |
| Rolling Statistics | RF_3d, RF_7d, MAX/MIN/DTR_3d_avg | 5 |
| Lag Features | RF_lag1, MAX_lag1, MIN_lag1, LABEL_lag1 | 4 |
| Seasonal Encodings | MONTH_sin/cos, DOY_sin/cos, SEASON | 5 |
| Weather Flags | HA_flag, RF_nonzero | 2 |
| Upper-Air Stability | CAPE, CIN, K_INDEX, LIFTED_INDEX, TOTALS_TOTALS, PRECIP_WATER | 6 |
| ERA5 Surface (6-hrly) | T2M, D2M, U10, V10, CAPE, SP | 6 |
| ERA5 Pressure Levels | T, q, u, v at 500/700/850 hPa | 12 |
| Slot Encodings | slot_sin/cos, slot_month_clim, doy_sin/cos | 5 |
| Derived Interactions ★ | cape_x_kindex, li_x_totals, q_gradient_500_850, thetae_850, wind_shear_500_850, wind_shear_700_850, moisture_flux_850/700, thickness_500_850, mid_level_drying | 10 |
| **Temporal Rolling (v5) ★★** | **CAPE/KI/PWAT/shear rolling means (3/7/14-day)** | **12** |
| **Temporal Lags (v5) ★★** | **CAPE/KI/LI/PWAT/shear lags (t-1/2/3)** | **15** |
| **Trend Features (v5) ★★** | **CAPE_trend3, KI_trend3, PWAT_trend3** | **3** |

★ New in v3 · ★★ New in v5 (atmospheric only, no label leakage)

---

## Data Sources

| Source | Data | Cadence | Script |
|--------|------|---------|--------|
| GFS NOMADS | CAPE, K-Index, LI, TT, ERA5 fields, f012/f024/f048 | 6-hourly | `gfs_fetcher.py` |
| aviationweather.gov | Live METAR — TEMP, WIND, RH, VIS, flight category | 30-min | `fetch_metar.py` |
| Himawari-9 (NOAA S3) | Band 13 IR BT, 50km VOBL box | 10-min | `fetch_himawari_realtime.py` |
| Blitzortung Network | Real-time lightning strikes, 250km radius | Real-time WS | `main.py /ws/lightning` |
| Univ. of Wyoming | Radiosonde sounding Station 43295 | 00Z/12Z | Dashboard (client-side) |
| IMD Table-II (43295) | Surface obs, TH flag, G-codes | Daily | Training + `verify_today.py` |
| ERA5 (CDS API) | 6-hourly T/q/u/v 500/700/850hPa | 6-hourly | Training (Vidhi) |
| IGRA Soundings | Upper-air profiles | Daily | Training only |
| MetPy (GFS-derived) | Stability indices all 4 slots | 6-hourly | `fetch_upperair_realtime.py` |

---

## Repository Structure

```
CSIR_Thunderstorm/
│
├── .github/workflows/
│   └── forecast_update.yml         # CI/CD: Himawari→GFS→forecast→METAR→SHAP→verify→deploy
│
├── data/
│   ├── bengaluru_thunderstorm_features_merged.csv
│   ├── bengaluru_6hr_labels.csv
│   ├── era5_6hrly_bengaluru_2015_2025.csv          (Vidhi)
│   ├── upperair_realtime_43295.csv                 (Atul — GFS MetPy)
│   ├── gfs_realtime_43295.csv                      (gfs_fetcher.py)
│   ├── gfs_multiday_43295.json                     (f024/f048 outlook)
│   ├── himawari_realtime.json                      (current BT + storm signal)
│   ├── himawari_history.json                       (last 6 frames)
│   ├── realtime_shap.json                          (today's SHAP values per slot)
│   ├── forecast_log.csv                            (rolling prediction log)
│   ├── actual_log.csv                              (IMD TH flag ground truth 2015-2025)
│   ├── verification_today.json                     (daily verification output)
│   ├── cape_climatology.json                       (monthly CAPE/KI baselines — Vidhi)
│   └── skill_scores.json                           (rolling 30-day POD/FAR/HSS/Brier)
│
├── models/
│   ├── nowcast_slot0_xgb_v4_ensemble.pkl    ← Slot 0 PRODUCTION (AUROC 0.8484)
│   ├── nowcast_slot1_xgb_v5_temporal.pkl    ← Slot 1 PRODUCTION (AUROC 0.8317, 84 features)
│   ├── nowcast_slot2_xgb_v3_calibrated.pkl  ← Slot 2 PRODUCTION (AUROC 0.8710)
│   ├── nowcast_slot3_xgb_v3_calibrated.pkl  ← Slot 3 PRODUCTION (AUROC 0.8710)
│   ├── himawari_correction_model.pkl         ← Satellite correction (CV AUROC 0.8812)
│   ├── correction_model_meta.json
│   ├── xgb_optuna_best.pkl                  (daily meta-ensemble)
│   ├── rf_best.pkl
│   ├── gb_best.pkl
│   ├── lstm_best.pt
│   ├── lstm_scaler.pkl
│   ├── ensemble_meta_mlp.pkl                (meta-learner, AUROC 0.8791)
│   ├── ensemble_config.json
│   ├── optimal_thresholds.json              (F-beta optimized per slot)
│   ├── slot_top_features.json               (SHAP top-20 per slot)
│   └── synoptic_clusterer.pkl               (KMeans regime detector)
│
├── forecast_action.py              # Main pipeline — v5 best-per-slot + Himawari correction
├── gfs_fetcher.py                  # Smart GFS fetcher (auto cycle discovery + f012/f024/f048)
├── fetch_himawari_realtime.py      # Himawari-9 Band 13 BT from NOAA S3 (Atul)
├── fetch_upperair_realtime.py      # MetPy stability indices from GFS (Atul)
├── fetch_metar.py                  # Live METAR from aviationweather.gov (VOBL)
├── compute_realtime_shap.py        # Real-time SHAP from today's GFS inputs
├── verify_today.py                 # Daily IMD TH-flag verification (Sneha)
├── train_correction_model.py       # Himawari correction model training
├── build_actual_log_historical.py  # Backfill actual_log from IMD (Sneha)
├── compute_skill_scores.py         # Rolling 30-day skill score time series (Sneha)
├── main.py                         # FastAPI — 10 endpoints + WS lightning (Satvik)
├── index.html                      # React dashboard — 8 pages (single file)
├── forecast.json                   # Latest forecast — auto-updated 2x daily
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Aprameya05/CSIR-Thunderstorm-Bengaluru.git
cd CSIR-Thunderstorm-Bengaluru
pip install xgboost scikit-learn==1.6.1 joblib shap pandas numpy torch \
            fastapi uvicorn requests metpy boto3 satpy websockets groq \
            pyresample pytz cfgrib eccodes xarray optuna lightgbm
```

---

## Usage

```bash
# Run 6-hour nowcast (best-per-slot v5 ensemble)
python forecast_action.py

# Smart GFS fetch (auto-discovers latest cycle + f024/f048)
python gfs_fetcher.py

# Fetch Himawari-9 satellite data (run BEFORE forecast_action)
python fetch_himawari_realtime.py

# Fetch live METAR for VOBL
python fetch_metar.py

# Compute real-time SHAP from today's GFS inputs
python compute_realtime_shap.py

# Run daily verification against IMD observations
python verify_today.py

# Train Himawari correction model
python train_correction_model.py

# Retrain v5 temporal ensemble (requires A100 or equivalent)
# See CSIR_Thunderstorm_A100_Training.ipynb

# Start API locally
uvicorn main:app --reload
# → http://127.0.0.1:8000/docs

# Test correction model status
curl https://csir-thunderstorm-api.onrender.com/correction/status

# Test slot explanation (no LLM)
curl https://csir-thunderstorm-api.onrender.com/nowcast/explain/2

# Test RAG explanation
curl -X POST https://csir-thunderstorm-api.onrender.com/rag/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did Slot 2 fire today?", "date": "2026-07-28"}'
```

---

## Development Roadmap

### Phase 1 — Daily Model ✅
- [x] XGBoost with surface + upper-air + ERA5 (AUROC 0.871)
- [x] SHAP explainability, WMO verification metrics, FastAPI

### Phase 2 — 6-Hour Nowcasting ✅
- [x] Per-slot XGBoost v1→v3 with calibration
- [x] ERA5 6-hourly 2015–2025 (Vidhi)
- [x] GFS real-time + MetPy upper-air (Atul)
- [x] RAG endpoints (Groq Llama-3.3-70b)

### Phase 3 — Operations ✅
- [x] Live React dashboard — 8 pages (Cloudflare Pages)
- [x] GitHub Actions CI/CD — 58s end-to-end
- [x] FastAPI public deployment (Render)
- [x] Himawari-9 BT storm proximity fetcher (Atul)
- [x] Rolling verification pipeline (Sneha)

### Phase 3.5 — Intelligence Layer ✅
- [x] Smart GFS cycle auto-discovery
- [x] Trend arrows on slot probability cards
- [x] Convective initiation timer with instability score (0–100)
- [x] Multi-day extended outlook (f024/f048)
- [x] Real-time SHAP per slot from today's GFS inputs
- [x] Himawari-9 BT color visualization on Radar Map (6-frame timeline)
- [x] Live METAR integration (VOBL — TEMP, WIND, RH, VIS, flight category)
- [x] Synoptic regime auto-detection with today's regime highlighted
- [x] Live API "Try it" buttons with elapsed timer + Render wake-up UX
- [x] Automated daily verification pipeline (IMD TH flag → POD/FAR/HSS/Brier)
- [x] Blitzortung lightning WebSocket feed (250km radius, wss:// proxy)
- [x] Skew-T Log-P sounding diagram (Univ. of Wyoming + GFS fallback)
- [x] Instability Composite gauge (CAPE × K-Index, MINIMAL→SEVERE)
- [x] Himawari real-time satellite override (trained correction model, CV AUROC 0.8812)
- [x] Full logical consistency audit — STATUS, CONVECTIVE RISK, alert card all react to live state

### Phase 4 — A100 Ensemble Upgrade ✅
- [x] XGBoost Optuna ensemble (20 seeds, 100 trials, GPU) — CV AUROC 0.9243
- [x] Bidirectional LSTM + Multi-head Attention (100 epochs, A100)
- [x] MLP meta-learner via OOF stacking — Test AUROC 0.8791 (+0.81pp vs v3)
- [x] v5 temporal models — 84 atmospheric lag features, Slot 1 +3pp
- [x] F-beta(2) optimized thresholds per slot — Slot 3 POD 47%→66%
- [x] Per-slot best model selection (v3/v4/v5 based on test AUROC)
- [x] October specialist model tested and correctly rejected (general v5 outperforms)
- [x] Historical Himawari BT archive pull (Atul, 2023 storm days — in progress)

### Phase 5 — Pending 📋
- [ ] Historical Himawari BT archive (Atul — full 2016-2025, replaces ERA5 proxies in correction model)
- [ ] Retrain correction model on real Himawari BT (expected AUROC >0.92)
- [ ] Model drift detection alert (rolling AUROC < 0.75 trigger)
- [ ] Nowcast skill score chart on Models page (pending 2026 IMD data from Dr. Agnihotri)
- [ ] Upper-air fetcher for all 4 slots independently (Atul)
- [ ] GPM IMERG precipitation overlay on Radar Map
- [ ] IMD Doppler radar integration (pending Dr. Agnihotri — VOBL DWR not yet public)
- [ ] IMD Table-II 2026 update (pending Dr. Agnihotri — needed for live verification)
- [ ] Monsoon break/active phase detector (OLR-based)
- [ ] Seasonal automated retraining pipeline

---

## Team

| Member | Role | Key Contributions |
|--------|------|------------------|
| **Aprameya Bharadwaj** | ML Lead & Architect | A1–A5, A100 ensemble training, v4/v5 models, Himawari correction model, threshold optimization, pipeline, RAG, dashboard (8 pages), CI/CD, GFS fetcher, SHAP, convective timer, live METAR, regime detection, Live API UX, lightning WebSocket, Skew-T diagram, logical consistency audit |
| **Atul Denny** | Data & Pipeline Lead | GFS fetcher (all 4 slots, UTC date fix), MetPy upper-air stability indices, u/v wind components, Himawari-9 satellite fetcher (NOAA S3), historical Himawari BT archive (in progress) |
| **Satvik** | Deployment Lead | FastAPI on Render, 10 endpoints including RAG + correction status + slot explain + WS lightning proxy, Render WebSocket deployment |
| **Vidhi** | ERA5 Data | 6-hourly ERA5 2015–2025 (16,072 rows, zero nulls), CAPE climatology by month |
| **Sneha** | Verification Lead | EDA charts, ForecastLogger, `verify_today.py`, `build_actual_log_historical.py`, `compute_skill_scores.py`, actual_log 2015-2025 (15,280 rows, 898 storm slots) |

---

## Acknowledgements

Developed under the guidance of **Dr. Geeta Agnihotri (Scientist F, IMD Bengaluru)** as part of a CSIR-backed research initiative for operational thunderstorm prediction at Kempegowda International Airport.

Data: IMD Table-II observations, ERA5 (Copernicus CDS), GFS (NOAA NOMADS), Himawari-9 (NOAA AWS S3), IGRA soundings, Blitzortung lightning network, University of Wyoming radiosonde archive, aviationweather.gov METAR.

Compute: NVIDIA A100-SXM4-40GB via Google Colab Pro (ensemble training).

---

<div align="center">

*For academic and research purposes only.*
*Copyright © 2026 CSIR Thunderstorm Prediction System. All rights reserved.*

</div>
