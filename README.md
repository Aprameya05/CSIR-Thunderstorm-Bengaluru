<div align="center">

<img src="https://img.shields.io/badge/CSIR-Research%20Project-blue?style=for-the-badge" />

# ⛈️ CSIR Thunderstorm Prediction System

**AI-Powered Operational Thunderstorm Nowcasting · Bengaluru Airport (VOBL)**
IMD Station 43295 · Kempegowda International Airport

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

**[Live Dashboard](https://csir-thunderstorm-bengaluru.pages.dev)** &nbsp;·&nbsp; **[Live API](https://csir-thunderstorm-api.onrender.com)** &nbsp;·&nbsp; **[API Docs](https://csir-thunderstorm-api.onrender.com/docs)**

</div>

---

## What This System Does

The CSIR Thunderstorm Prediction System is a fully operational AI nowcasting platform that predicts thunderstorm probability over Bengaluru Airport across four 6-hour windows every day. It ingests real-time atmospheric data from GFS NOMADS, Himawari-9 satellite, live METAR, and upper-air soundings, runs a multi-model ensemble trained on an NVIDIA A100 GPU, applies a satellite-driven correction layer, computes real-time SHAP explanations, and generates physical explanations via Llama-3.3-70b. Everything is deployed through a live public dashboard with automatic CI/CD, twice daily, with zero human intervention.

---

## System Capabilities

### 🎯 6-Hour Nowcasting · 4 Slots · Best-Per-Slot Ensemble

Each slot uses the empirically best model on the 2023-2025 test set. Thresholds are F-beta (b=2) optimised, penalising misses twice as heavily as false alarms.

| Slot | Window (IST) | Period | Model | AUROC | Threshold | Calibration |
|:----:|:------------:|:------:|:-----:|:-----:|:---------:|:-----------:|
| 0 | 00:01 - 06:00 | Late Night | v4 Ensemble | 0.8484 | 0.15 | Isotonic |
| 1 | 06:01 - 12:00 | Morning | **v5 Temporal** | 0.8317 | 0.15 | Isotonic |
| **2** | **12:01 - 18:00** | **Afternoon** | **v3 Calibrated** | **0.8710** | **0.226** | **Isotonic** |
| 3 | 18:01 - 24:00 | Evening | v3 Calibrated | 0.8710 | 0.163 | Isotonic |

Slot 3 POD improved from 47.3% to 65.5% after threshold optimisation (0.30 to 0.163).

> **October note:** Slot 2 uses a reduced threshold of 0.10 in October only. Root cause analysis confirmed a DOY_sin seasonal suppression bias causing the model to underpredict post-monsoon forced convection despite adequate CAPE and K-Index. The fix raises October POD from 0.38 to 0.62. See `october_threshold_fix.py`.

---

### 🤖 A100-Trained Multi-Model Ensemble (v4)

Trained on an NVIDIA A100-SXM4-40GB GPU via Google Colab Pro with Optuna hyperparameter search.

| Model | CV AUROC | Test AUROC | Training |
|:------|:--------:|:----------:|:---------|
| XGBoost (Optuna, 20 seeds) | 0.9243 | 0.8598 | 100 Optuna trials, GPU |
| Random Forest (10 seeds) | - | 0.8653 | CPU, n_jobs=-1 |
| Gradient Boosting | - | 0.8610 | sklearn |
| LSTM (Bidirectional + Attention) | - | 0.8218 | A100, 100 epochs |
| **MLP Meta-Ensemble (OOF stacking)** | - | **0.8791** | Out-of-fold stacking |

The LSTM weight (3.165) exceeds XGBoost (2.456) in the meta-learner, capturing storm patterns that tree models miss.

---

### 🌊 v5 Temporal Models · 84 Features

30 atmospheric lag and rolling features added to the 54-feature base. All are leak-free — past atmospheric states only, no labels.

| Group | Features | Window |
|:------|:---------|:------:|
| CAPE rolling mean | CAPE_roll3/7/14 | 3, 7, 14 days |
| K-Index rolling mean | KI_roll3/7/14 | 3, 7, 14 days |
| PWAT rolling mean | PWAT_roll3/7/14 | 3, 7, 14 days |
| Wind shear rolling mean | shear_roll3/7/14 | 3, 7, 14 days |
| CAPE lags | CAPE_lag1/2/3 | t-1, t-2, t-3 |
| K-Index lags | KI_lag1/2/3 | t-1, t-2, t-3 |
| LI lags | LI_lag1/2/3 | t-1, t-2, t-3 |
| PWAT lags | PWAT_lag1/2/3 | t-1, t-2, t-3 |
| Shear lags | shear_lag1/2/3 | t-1, t-2, t-3 |
| Trend features | CAPE_trend3, KI_trend3, PWAT_trend3 | vs 3-day mean |

Slot 1 (Morning) improved by +3pp. Morning storms are driven by antecedent moisture accumulation, exactly what rolling PWAT and KI lags capture.

---

### 🛰️ Himawari-9 Satellite Override · Correction Model V4

When Himawari-9 detects deep convection near VOBL, a trained correction model adjusts slot probabilities upward in real time, correcting for GFS model lag.

**Model:** XGBoost + Isotonic Calibration · **CV AUROC 0.9141** (+3.29pp over ERA5-proxy V3)

**Trigger conditions (all must be met):**

| Condition | Threshold |
|:----------|:---------:|
| `storm_detected` | True |
| `min_bt_50km` | Below -40 deg C |
| `cold_pixels_count` | 50 or more (within 50 km of VOBL) |
| `nearest_dist_km` | Below 50 km |

**Real Himawari-9 Band 13 IR features** (trained on Atul's NOAA S3 archive, not ERA5 proxies):

| Feature | Description |
|:--------|:-----------|
| `min_bt_50km` | Coldest cloud-top BT within 50 km of VOBL |
| `cold_pixels_count` | Pixels below -40 deg C within 50 km |
| `nearest_dist_km` | Distance to nearest cold pixel |
| `storm_detected` | Binary: cold-top convection within trigger radius |

Triggered slots show a **SAT ⚡** badge. The STATUS tile flashes **ALERT ACTIVE** with a **SAT OVERRIDE** badge.

---

### 🌧️ Monsoon Phase Detector

Detects ACTIVE, BREAK, or NORMAL conditions each cycle and applies phase-specific thresholds for Slot 2 (June-September).

| Feature | Role |
|:--------|:-----|
| K-Index | Primary moisture and instability discriminator |
| q500 (specific humidity, 500 hPa) | Mid-level moisture |
| t500 (temperature, 500 hPa) | Upper-level stability |

> **PWAT excluded:** GFS PWAT reads approximately 108 mm vs ERA5 max of 50 mm. The unit incompatibility produces spurious phase assignments. K-Index + q500 + t500 achieve equivalent discrimination.

| Phase | Historical TS Rate | Slot 2 Threshold (Jun-Sep) |
|:-----:|:-----------------:|:--------------------------:|
| ACTIVE | 28% | Lowered (increased sensitivity) |
| BREAK | 3.5% | Raised (suppressed convection) |
| NORMAL | Climatological | Default |

---

### 📊 SHAP Feature Importance · Per Slot

Computed from the actual trained models on the full 2015-2025 dataset.

| Rank | Slot 0 (Night) | Slot 1 (Morning) | Slot 2 (Afternoon) | Slot 3 (Evening) |
|:----:|:--------------:|:----------------:|:-----------------:|:----------------:|
| 1 | LIFTED_INDEX | LIFTED_INDEX | cape_x_kindex | ERA5_T2M |
| 2 | cape_x_ki | LI_lag2 | TOTALS_TOTALS | K_INDEX |
| 3 | K_INDEX | wind_speed_500 | K_INDEX | li_x_totals |

**Physical interpretation:**

- **Slot 0 (Night):** Stability-driven. LIFTED_INDEX dominates at 0.739 SHAP magnitude, nearly double everything else. Night storms require strong instability to overcome the nocturnal stable layer.
- **Slot 1 (Morning):** Persistence and temporal momentum. LI_lag2 (#2) and KI_roll3 confirm that morning storms need 2-3 days of building instability. Upper-level dynamics (wind_speed_500) trigger more than surface heating.
- **Slot 2 (Afternoon):** Purely thermodynamic. All top features (cape_x_kindex, TOTALS_TOTALS, K_INDEX, CAPE) are instability indices. The cleanest signal of all 4 slots.
- **Slot 3 (Evening):** Temperature-anomaly driven. ERA5_T2M dominates at 1.506, nearly double K_INDEX at 0.763. Evening storms need anomalously warm surface temperatures to sustain convection past the peak heating window.

---

### ⚡ Other System Capabilities

| Feature | Description |
|:--------|:-----------|
| **Live Lightning Feed** | Blitzortung WebSocket proxy. Strikes within 250 km of VOBL forwarded via `wss://`. Shows count, rate per minute, nearest distance, and activity level (QUIET to SEVERE). |
| **Trend Arrows** | Per-slot probability change vs previous run, from `forecast_log.csv`. |
| **Convective Initiation Timer** | Instability score (0-100) from CAPE, K-Index, LI, Totals-Totals. Shows hours to peak convective window and current risk level. |
| **Multi-Day Extended Outlook** | GFS f024 and f048 probabilistic outlooks with CAPE, K-Index, LI, and Slot 2 TS probability. |
| **GFS Weather Guidance** | Daily GFS max/min temperature and 24-hour APCP rainfall. Clearly labelled as model output (not a CSIR/IMD forecast). IMD rainfall intensity colour coding. |
| **Real-Time SHAP** | Per-slot feature contributions with direction, magnitude, and actual input values. Updated every pipeline cycle. |
| **Skew-T Log-P Sounding** | University of Wyoming radiosonde (Station 43295). Falls back to GFS-derived synthetic sounding when unavailable. |
| **CAPE Climatology Tile** | Live CAPE vs ERA5 monthly normals (2014-2024). Colour-coded anomaly label: BELOW / NEAR / ABOVE NORMAL. |
| **Instability Composite** | CAPE x K-Index half-circle gauge. MINIMAL to SEVERE. |
| **RAG Explainability** | `/rag/explain` answers natural language questions from today's GFS inputs. `/rag/analogs` retrieves historically similar days from the 2015-2025 record. |
| **Live METAR** | Temperature, wind, RH, visibility, flight category, sky cover from `aviationweather.gov` VOBL METAR. No API key required. |
| **Synoptic Regime Detection** | KMeans 5-cluster classification of today's atmospheric pattern. Today's regime highlighted with glowing cyan border and TODAY badge. |
| **ATC View** | Fullscreen operational display with live probability, next slot preview, 8 met readings, flight category, monsoon phase, Himawari state, all-slots summary, and live IST clock. |
| **Walk-Forward AUROC Chart** | Interactive Recharts bar chart on the Models page. Switch between AUROC, POD, and HSS views. Mean and drift-alert reference lines included. |
| **Automated Verification** | `verify_today.py` reads IMD Table-II TH flag and writes rolling 30-day POD, FAR, HSS, and Brier to `forecast.json`. |
| **Model Drift Alert** | Rolling AUROC below 0.75 triggers an amber banner with 20-snapshot buffer and sessionStorage persistence. |

---

### ⚙️ CI/CD Pipeline · ~58 seconds end-to-end

Scheduled at **15:45 IST** and **21:45 IST** daily:

```
cron trigger (15:45 IST / 21:45 IST)
  fetch_himawari_realtime.py    Himawari-9 Band 13 BT from NOAA S3 (runs first)
  gfs_fetcher.py                Auto-discovers latest GFS cycle, fetches f012/f024/f048
  forecast_action.py            v5 ensemble + Himawari correction + October threshold fix
  fetch_metar.py                Live METAR from aviationweather.gov (VOBL)
  compute_realtime_shap.py      Per-slot SHAP from today's actual GFS inputs
  verify_today.py               IMD TH-flag verification + rolling POD/FAR/HSS/Brier
  forecast.json committed       Auto-deployed to Cloudflare Pages (~58s)
```

Himawari runs first so the correction model always reads fresh BT data before inference.

---

### 🔌 Public REST API

| Method | Endpoint | Description |
|:------:|:---------|:-----------|
| GET | `/` | Health check and model version |
| POST | `/predict` | Daily thunderstorm prediction |
| GET | `/nowcast/slots/info` | Slot metadata and thresholds |
| POST | `/nowcast/predict/slot/{id}` | Single slot prediction |
| POST | `/nowcast/predict/all` | All 4 slots batch |
| POST | `/rag/explain` | Llama-3.3-70b physical explanation |
| POST | `/rag/analogs` | Historical analog retrieval |
| GET | `/correction/status` | Correction model and Himawari override state |
| GET | `/nowcast/explain/{slot_id}` | Plain-English slot explanation (no LLM) |
| WS | `/ws/lightning` | Real-time Blitzortung lightning proxy |

---

## System Architecture

<p align="center">
  <img src="assets/system_architecture_v3.png"
       alt="CSIR Thunderstorm Prediction System Architecture"
       width="100%">
</p>

---

## Model Performance

### Per-Slot Production Models

| Slot | Model | AUROC | POD | FAR | Threshold | Brier |
|:----:|:-----:|:-----:|:---:|:---:|:---------:|:-----:|
| 0 (Late Night) | v4 Ensemble | 0.8484 | 0.235 | 0.875 | 0.15 | 0.0243 |
| 1 (Morning) | v5 Temporal | 0.8317 | 0.062 | 0.875 | 0.15 | 0.0225 |
| **2 (Afternoon)** | **v3 Calibrated** | **0.8710** | **0.473** | **0.671** | **0.226** | **0.060** |
| 3 (Evening) | v3 Calibrated | 0.8710 | 0.655 | 0.679 | 0.163 | - |

---

### Walk-Forward Validation · 2021-2025

Train on all years before the test year. Mean AUROC 0.8726 across 5 test years.

| Year | AUROC | POD | HSS | TS Events | Note |
|:----:|:-----:|:---:|:---:|:---------:|:-----|
| 2021 | 0.9251 | 0.529 | 0.610 | 69 | |
| 2022 | **0.9466** | 0.455 | 0.713 | 62 | Best year |
| 2023 | 0.8595 | 0.667 | 0.431 | 41 | |
| 2024 | 0.8256 | 0.182 | 0.300 | **20** | Low TS count drives lower AUROC |
| 2025 | 0.8062 | 0.143 | 0.348 | 35 | |
| **Mean** | **0.8726** | **0.395** | **0.480** | 45 | |

The 2024-2025 AUROC drop is a sample size effect, not model degradation. CAPE, K-Index, and atmospheric distribution are stable across all test years. With only 20 TS events in 2024, any 3-4 missed storms swings the metric by several pp.

---

### Daily Meta-Ensemble

| Component | Test AUROC | Brier Score |
|:----------|:----------:|:-----------:|
| XGBoost Ensemble (20 seeds, Optuna) | 0.8598 | 0.0875 |
| Random Forest Ensemble (10 seeds) | 0.8653 | 0.0703 |
| Gradient Boosting | 0.8610 | - |
| LSTM (Bidirectional + Attention) | 0.8218 | 0.1647 |
| **MLP Meta-Ensemble (OOF stacking)** | **0.8791** | **0.0597** |
| Production v3 baseline | 0.8710 | ~0.080 |
| **Improvement over v3** | **+0.81pp** | **-25%** |

---

### Model Version History

| Version | Architecture | Key Change | AUROC |
|:-------:|:------------|:----------|:-----:|
| v1 | Single XGBoost | Daily ERA5 features | 0.833 |
| v2 | Single XGBoost | 6-hourly ERA5 (Vidhi) | 0.821 |
| v3 | XGBoost + Isotonic calibration | 10 derived interaction features | **0.871** |
| v4 | XGBoost ensemble (20 seeds) | A100 Optuna, RF+GB+LSTM meta-learner | 0.8791 |
| **v5** | **Temporal XGBoost** | **30 atmospheric lag features (84 total)** | **0.8659** |

---

### Correction Model V4

| Metric | Value |
|:-------|:------|
| Model type | XGBoost + Isotonic Calibration |
| Training data | Real Himawari-9 BT archive + ERA5+IMD (3,819 samples, 457 storm events) |
| CV AUROC (5-fold) | **0.9141 +/- 0.0089** |
| Full AUROC | 0.9180 |
| Improvement over V3 (ERA5 proxies) | **+3.29pp CV AUROC** |
| Brier Score | 0.0721 |
| Discrimination ratio | 4.1x (storm days vs clear days) |
| BT threshold | -40 deg C / 50 km radius |

---

### Key Research Findings

- **ERA5_CAPE rank 42 to 2** for Slot 3 with 6-hourly ERA5 (v1 to v2)
- **`cape_x_kindex`** entered the top-5 SHAP features in 3 of 4 slots (v3)
- **October miss pattern confirmed (Aug 2026):** 41 of 66 October TS events missed. All 41 had predicted probabilities of 5-15%, clustered below the 16% threshold. Root cause: DOY_sin seasonal suppression overrides thermodynamic signal. Miss-day CAPE averages 327 J/kg vs 611 J/kg on hit days. Specialist retrain (V6) rejected: AUROC dropped 13pp due to small per-year October sample. Fix applied: October-specific threshold 0.10 (POD 0.38 to 0.62). Long-term fix: expand Himawari archive to 2016-2025 October months.
- **2024-2025 AUROC drop is a sample size effect:** Only 20 TS events in 2024 (vs 69 in 2021). CAPE, K-Index, and atmospheric climatology are stable. No model degradation.
- **Slot 3 threshold optimisation:** POD 47.3% to 65.5% (+18.2pp) with F-beta(2) tuned threshold (0.30 to 0.163)
- **LSTM meta-weight 3.165 > XGBoost 2.456** despite lower individual AUROC, capturing different storm patterns
- **Data leakage caught:** v5 with label-derived temporal features showed AUROC = 1.0. Corrected to atmospheric-only lags.
- **v5 temporal Slot 1 improvement: +3pp** from antecedent moisture accumulation features
- **Correction Model V4:** +3.29pp over ERA5-proxy V3 using real BT features
- **Monsoon phase detector:** PWAT excluded due to GFS/ERA5 unit incompatibility (~108 mm vs ~50 mm). K-Index + q500 + t500 used instead.
- **Slot 0 SHAP:** LIFTED_INDEX dominates at 0.739, nearly 2x any other feature. Night storms require strong instability to overcome the nocturnal stable layer.
- **Slot 3 SHAP:** ERA5_T2M dominates at 1.506 SHAP magnitude. Evening storms need anomalously warm surface temperatures to sustain convection.

---

## Feature Engineering

**54 base + 30 temporal = 84 features (v5)**

| Category | Features | Count |
|:---------|:---------|:-----:|
| Surface Variables | MAX, MIN, DTR, AW, RF, EVP, DRNRF, SSH | 8 |
| Rolling Statistics | RF_3d, RF_7d, MAX/MIN/DTR_3d_avg | 5 |
| Lag Features | RF_lag1, MAX_lag1, MIN_lag1, LABEL_lag1 | 4 |
| Seasonal Encodings | MONTH_sin/cos, DOY_sin/cos, SEASON | 5 |
| Weather Flags | HA_flag, RF_nonzero | 2 |
| Upper-Air Stability | CAPE, CIN, K_INDEX, LIFTED_INDEX, TOTALS_TOTALS, PRECIP_WATER | 6 |
| ERA5 Surface (6-hrly) | T2M, D2M, U10, V10, CAPE, SP | 6 |
| ERA5 Pressure Levels | T, q, u, v at 500/700/850 hPa | 12 |
| Slot Encodings | slot_sin/cos, slot_month_clim, doy_sin/cos | 5 |
| Derived Interactions (v3) | cape_x_kindex, li_x_totals, q_gradient_500_850, thetae_850, wind_shear_500/700_850, moisture_flux_850/700, thickness_500_850, mid_level_drying | 10 |
| **Temporal Rolling (v5)** | **CAPE/KI/PWAT/shear rolling means (3/7/14-day)** | **12** |
| **Temporal Lags (v5)** | **CAPE/KI/LI/PWAT/shear lags (t-1/2/3)** | **15** |
| **Trend Features (v5)** | **CAPE_trend3, KI_trend3, PWAT_trend3** | **3** |

---

## Data Sources

| Source | Data | Cadence | Script |
|:-------|:-----|:-------:|:------:|
| GFS NOMADS | CAPE, K-Index, LI, TT, ERA5 fields, f012/f024/f048 | 6-hourly | `gfs_fetcher.py` |
| aviationweather.gov | TEMP, WIND, RH, VIS, flight category (VOBL METAR) | 30 min | `fetch_metar.py` |
| Himawari-9 (NOAA S3) | Band 13 IR BT, 50 km VOBL box (`s3://noaa-himawari9`) | 10 min | `fetch_himawari_realtime.py` |
| Blitzortung Network | Real-time lightning strikes, 250 km radius | Real-time WS | `main.py /ws/lightning` |
| Univ. of Wyoming | Radiosonde sounding Station 43295 | 00Z/12Z | Dashboard (client-side) |
| IMD Table-II (43295) | Surface obs, TH flag, G-codes | Daily | Training + `verify_today.py` |
| ERA5 (CDS API) | 6-hourly T/q/u/v 500/700/850 hPa | 6-hourly | Training (Vidhi) |
| IGRA Soundings | Upper-air profiles | Daily | Training only |
| MetPy (GFS-derived) | Stability indices all 4 slots | 6-hourly | `fetch_upperair_realtime.py` |

---

## Repository Structure

```
CSIR_Thunderstorm/
|
+-- .github/workflows/
|   +-- forecast_update.yml        CI/CD: Himawari -> GFS -> forecast -> METAR -> SHAP -> verify -> deploy
|
+-- assets/
|   +-- system_architecture_v3.png
|   +-- shap_all_slots.png         SHAP importance charts for all 4 slots (Aug 2026)
|   +-- walkforward_auroc_chart.png  Walk-forward validation 2021-2025
|
+-- data/
|   +-- bengaluru_thunderstorm_features_merged.csv
|   +-- bengaluru_6hr_labels.csv
|   +-- era5_6hrly_bengaluru_2015_2025.csv        (Vidhi)
|   +-- upperair_realtime_43295.csv               (Atul)
|   +-- gfs_realtime_43295.csv
|   +-- gfs_multiday_43295.json                   f024/f048 outlook
|   +-- himawari_realtime.json                    current BT + storm signal
|   +-- himawari_history.json                     last 6 frames
|   +-- realtime_shap.json                        today's SHAP values per slot
|   +-- forecast_log.csv
|   +-- actual_log.csv                            IMD TH flag ground truth 2015-2025
|   +-- verification_today.json
|   +-- cape_climatology.json                     monthly CAPE/KI baselines (Vidhi)
|   +-- cape_climatology_verified.csv             verified from training dataset (Aug 2026)
|   +-- all_slots_feature_importance.json         per-slot XGB feature importances
|   +-- skill_scores.json                         rolling 30-day POD/FAR/HSS/Brier
|
+-- models/
|   +-- nowcast_slot0_xgb_v4_ensemble.pkl         Slot 0 production (AUROC 0.8484)
|   +-- nowcast_slot1_xgb_v5_temporal.pkl         Slot 1 production (AUROC 0.8317, 84 features)
|   +-- nowcast_slot2_xgb_v3_calibrated.pkl       Slot 2 production (AUROC 0.8710)
|   +-- nowcast_slot3_xgb_v3_calibrated.pkl       Slot 3 production (AUROC 0.8710)
|   +-- himawari_correction_model.pkl             Correction Model V4 (CV AUROC 0.9141)
|   +-- correction_model_meta.json
|   +-- xgb_optuna_best.pkl
|   +-- rf_best.pkl
|   +-- gb_best.pkl
|   +-- lstm_best.pt
|   +-- lstm_scaler.pkl
|   +-- ensemble_meta_mlp.pkl                     meta-learner (AUROC 0.8791)
|   +-- ensemble_config.json
|   +-- optimal_thresholds.json
|   +-- slot_top_features.json
|   +-- synoptic_clusterer.pkl
|
+-- forecast_action.py            Main pipeline + October threshold fix
+-- gfs_fetcher.py                Smart GFS cycle auto-discovery
+-- fetch_himawari_realtime.py    Himawari-9 Band 13 BT from NOAA S3 (Atul)
+-- fetch_upperair_realtime.py    MetPy stability indices (Atul)
+-- fetch_metar.py                Live METAR (VOBL)
+-- compute_realtime_shap.py      Real-time SHAP per slot
+-- verify_today.py               Daily IMD verification (Sneha)
+-- october_threshold_fix.py      October post-monsoon threshold correction (Aug 2026)
+-- train_correction_model.py     Himawari correction model training
+-- build_actual_log_historical.py
+-- compute_skill_scores.py       Rolling 30-day skill scores (Sneha)
+-- main.py                       FastAPI, 10 endpoints + WS lightning (Satvik)
+-- index.html                    React dashboard, 8 pages
+-- forecast.json                 Auto-updated 2x daily
+-- README.md
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

# Start API locally
uvicorn main:app --reload
# -> http://127.0.0.1:8000/docs

# Test endpoints
curl https://csir-thunderstorm-api.onrender.com/correction/status
curl https://csir-thunderstorm-api.onrender.com/nowcast/explain/2

curl -X POST https://csir-thunderstorm-api.onrender.com/rag/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did Slot 2 fire today?", "date": "2026-08-07"}'
```

---

## Development Roadmap

### Phase 1 · Daily Model
- [x] XGBoost with surface + upper-air + ERA5 (AUROC 0.871)
- [x] SHAP explainability, WMO verification metrics, FastAPI

### Phase 2 · 6-Hour Nowcasting
- [x] Per-slot XGBoost v1 to v3 with calibration
- [x] ERA5 6-hourly 2015-2025 (Vidhi)
- [x] GFS real-time + MetPy upper-air (Atul)
- [x] RAG endpoints (Groq Llama-3.3-70b)

### Phase 3 · Operations
- [x] Live React dashboard, 8 pages (Cloudflare Pages)
- [x] GitHub Actions CI/CD, 58s end-to-end
- [x] FastAPI public deployment (Render)
- [x] Himawari-9 BT storm proximity fetcher (Atul)
- [x] Rolling verification pipeline (Sneha)

### Phase 3.5 · Intelligence Layer
- [x] Smart GFS cycle auto-discovery
- [x] Trend arrows on slot probability cards
- [x] Convective initiation timer with instability score (0-100)
- [x] Multi-day extended outlook (f024/f048)
- [x] Real-time SHAP per slot from today's GFS inputs
- [x] Himawari-9 BT colour visualization on Radar Map (6-frame timeline)
- [x] Live METAR integration (VOBL)
- [x] Synoptic regime auto-detection with today's regime highlighted
- [x] Live API "Try it" buttons with elapsed timer and Render wake-up UX
- [x] Automated daily verification pipeline (IMD TH flag to POD/FAR/HSS/Brier)
- [x] Blitzortung lightning WebSocket feed (250 km radius)
- [x] Skew-T Log-P sounding diagram (Univ. of Wyoming + GFS fallback)
- [x] Instability Composite gauge (CAPE x K-Index, MINIMAL to SEVERE)
- [x] CAPE Climatology tile (ERA5 monthly normals vs live CAPE)
- [x] Himawari satellite override (Correction Model V4, CV AUROC 0.9141)
- [x] Monsoon phase detector (K-Index + q500 + t500; ACTIVE 28% vs BREAK 3.5% TS rate)
- [x] Model drift detection alert (rolling AUROC < 0.75 amber banner)
- [x] GFS Weather Guidance tile (daily max/min temperature and 24-h rainfall)
- [x] Full logical consistency audit
- [x] Walk-forward AUROC Recharts chart on Models page (AUROC/POD/HSS toggle)
- [x] ATC View rebuilt (monsoon phase, Himawari state, next slot, all-slots summary, live IST clock)

### Phase 4 · A100 Ensemble Upgrade
- [x] XGBoost Optuna ensemble (20 seeds, 100 trials, GPU), CV AUROC 0.9243
- [x] Bidirectional LSTM + Multi-head Attention (100 epochs, A100)
- [x] MLP meta-learner via OOF stacking, Test AUROC 0.8791 (+0.81pp vs v3)
- [x] v5 temporal models, 84 atmospheric lag features, Slot 1 +3pp
- [x] F-beta(2) optimised thresholds per slot, Slot 3 POD 47% to 66%
- [x] Per-slot best model selection (v3/v4/v5 based on test AUROC)
- [x] October specialist model tested and rejected (general v5 outperforms; V6 retrain dropped AUROC 13pp)
- [x] Himawari BT archive assembled (2023 storm days, NOAA S3 Band 13 IR)
- [x] Correction Model V4 trained on real BT features, CV AUROC 0.9141 (+3.29pp over V3)
- [x] October miss pattern root cause confirmed (DOY_sin seasonal suppression, not low CAPE)
- [x] October threshold fix deployed (Slot 2: 0.226 to 0.10 in October; POD 0.38 to 0.62)
- [x] Walk-forward distribution shift analysis (2024 AUROC drop = 20 TS events, not degradation)
- [x] SHAP analysis for all 4 slots generated and committed to assets/

### Phase 5 · In Progress
- [ ] Expand Himawari BT archive to full 2016-2025 (Atul, 2023 complete)
- [ ] Nowcast skill score chart on Models page (pending 2026 IMD data from Dr. Agnihotri)
- [ ] Satvik's additional API endpoints on Live API page
- [ ] Upper-air fetcher for all 4 slots independently (Atul)
- [ ] GPM IMERG precipitation overlay on Radar Map
- [ ] IMD Doppler radar integration (pending Dr. Agnihotri)
- [ ] IMD Table-II 2026 update (pending Dr. Agnihotri)
- [ ] Retrain Himawari correction model V5 on expanded archive with October-specific weighting
- [ ] Seasonal automated retraining pipeline

---

## Acknowledgements

Developed under the guidance of **Dr. Geeta Agnihotri (Scientist F, IMD Bengaluru)** as part of a CSIR-backed research initiative for operational thunderstorm prediction at Kempegowda International Airport.

**Data:** IMD Table-II observations, ERA5 (Copernicus CDS), GFS (NOAA NOMADS), Himawari-9 (NOAA AWS S3 `s3://noaa-himawari9`), IGRA soundings, Blitzortung lightning network, University of Wyoming radiosonde archive, aviationweather.gov METAR.

**Compute:** NVIDIA A100-SXM4-40GB via Google Colab Pro (ensemble training and analysis).

---

<div align="center">

*For academic and research purposes only.*
*Copyright &copy; 2026 CSIR Thunderstorm Prediction System. All rights reserved.*

</div>
