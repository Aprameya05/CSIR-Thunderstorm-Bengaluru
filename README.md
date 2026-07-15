<div align="center">

# ⛈️ CSIR Thunderstorm Prediction System

### AI-Based Operational Thunderstorm Forecasting for Bengaluru Airport (IMD Station 43295)

Machine learning framework for predicting thunderstorm occurrence using **surface meteorological observations**, **upper-air stability indices**, **ERA5 reanalysis**, and **explainable AI** — now with **6-hour nowcasting**.

<p>
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/XGBoost-Primary%20Model-success" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" />
  <img src="https://img.shields.io/badge/Nowcasting-6--Hour%20Slots-orange" />
</p>

</div>

---

## Overview

The **CSIR Thunderstorm Prediction System** is an operational machine learning framework developed to forecast thunderstorm occurrence over **Kempegowda International Airport, Bengaluru (IMD Station 43295)**, in collaboration with **IMD Bengaluru (Dr. Geeta Agnihotri, Scientist F)**.

The system has two operational modes:

| Mode | Description | Status |
|------|-------------|:------:|
| **Daily Model** | Predicts whether a thunderstorm will occur on a given day | ✅ Complete |
| **6-Hour Nowcast** | Predicts thunderstorm probability for each 6-hour window of the day | ✅ Phase 1 Complete |

---

## System Architecture

```text
IMD Surface Observations (Station 43295)
         +  IGRA Upper-Air Soundings
         +  ERA5 Reanalysis (Daily → 6-Hourly in progress)
                          │
                          ▼
              Data Preprocessing & Label Construction
              (Daily labels + 6-hr slot labels from G-codes)
                          │
                          ▼
                  Feature Engineering
         (Surface · Rolling · Lag · Stability Indices · ERA5)
                          │
                          ▼
              ┌───────────────────────────┐
              │     XGBoost Classifier    │
              │  5-Fold Stratified CV     │
              │  Optuna Hyperparameter    │
              │  Tuning (50 trials)       │
              └───────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   Daily Model (Phase 1)   │  ← thunderstorm_model.pkl
              │   6-Hr Slot Models (A3)   │  ← nowcast_slot{0-3}_xgb.pkl
              └───────────────────────────┘
                          │
                          ▼
              SHAP Explainability + WMO Metrics
                          │
                          ▼
                   FastAPI REST API
                          │
                          ▼
              Operational Forecast Output
```

---

## Dataset

| Dataset | Period | Status |
|---------|--------|:------:|
| IMD Surface Observations (Station 43295) | 2015–2025 | ✅ Complete |
| IGRA Upper-Air Soundings | 2015–2025 | ✅ Complete |
| ERA5 Reanalysis (Daily) | 2015–2025 | ✅ Complete |
| ERA5 Reanalysis (6-Hourly) | 2015–2025 | 🔄 Downloading |
| IMD 6-Hr Thunderstorm Labels (G-codes) | 2015–2025 | ✅ Extracted |
| Ceilometer Data | Feb 2026–Present | 📄 Requested |

### Dataset Summary

| Metric | Value |
|--------|------:|
| Total Days | **3,819** |
| Thunderstorm Days | **457** |
| Total 6-Hr Windows | **15,276** |
| Thunderstorm Windows | **584** |
| 6-Hr Positive Rate | **3.8%** |
| Training Period | **2015–2022** |
| Test Period | **2023–2025** |

---

## 6-Hour Slot Definition

Thunderstorm labels are derived from IMD Table-II weather type codes (T=9) and G-codes (time of commencement). Each day is divided into four 6-hour prediction windows:

| Slot | IST Window | ERA5 Snapshot (UTC) | Base Rate | Notes |
|------|-----------|-------------------|-----------|-------|
| 0 | 0001–0600 | 00Z | 3.7% | Nocturnal / late-night systems |
| 1 | 0601–1200 | 06Z | 1.1% | Morning — rare |
| 2 | 1201–1800 | 12Z | **6.3%** | **Peak convective window** |
| 3 | 1801–2400 | 18Z | 5.9% | Evening convection |

---

## Feature Engineering

| Category | Features | Count |
|----------|----------|------:|
| Surface Variables | MAX, MIN, DTR, AW, RF, EVP, DRNRF, SSH | 8 |
| Rolling Statistics | RF_3d, RF_7d, MAX_3d_avg, MIN_3d_avg, DTR_3d_avg | 5 |
| Lag Features | RF_lag1, MAX_lag1, MIN_lag1, LABEL_lag1 | 4 |
| Seasonal Encodings | MONTH_sin, MONTH_cos, DOY_sin, DOY_cos, SEASON | 5 |
| Weather Flags | HA_flag, RF_nonzero | 2 |
| Upper-Air Stability | CAPE, K_INDEX, LIFTED_INDEX, TOTALS_TOTALS, PRECIP_WATER | 5 |
| ERA5 Surface | T2M, D2M, U10, V10, CAPE, SP | 6 |
| ERA5 Pressure Levels | T/q/u/v at 500, 700, 850 hPa | 12 |
| Slot Features | slot_sin, slot_cos, slot_month_clim, doy_sin, doy_cos | 5 |
| Slot Lag Features | ts_label_lag1_slot, ts_any_yesterday | 2 |
| **Total** | | **54** |

---

## Model Performance

### Daily Model (Phase 1)

| Model | AUROC | POD | FAR | CSI | HSS |
|-------|------:|----:|----:|----:|----:|
| Logistic Regression | 0.818 | — | — | — | — |
| Random Forest | 0.842 | — | — | — | — |
| LightGBM | 0.799 | — | — | — | — |
| **XGBoost (surface only)** | 0.809 | 0.510 | 0.742 | 0.207 | 0.247 |
| **XGBoost (surface + upper-air + ERA5)** | **0.871** | **0.500** | **0.586** | **0.293** | **0.389** |

### 6-Hour Nowcast Models (Phase 2 — A3)

| Slot | Window | CV AUROC | Test AUROC | POD | FAR | CSI | HSS | Threshold |
|------|--------|:--------:|:----------:|----:|----:|----:|----:|:---------:|
| 0 | 0001–0600 IST | 0.9447 | 0.9205 | 0.100 | 0.944 | 0.037 | 0.059 | 0.54 |
| 1 | 0601–1200 IST | — | 0.8637 | 0.333 | 0.917 | 0.071 | 0.129 | 0.47 |
| **2** | **1201–1800 IST** | **0.9113** | **0.8333** | **0.356** | **0.649** | **0.215** | **0.303** | **0.33** |
| 3 | 1801–2400 IST | 0.9056 | 0.8166 | 0.286 | 0.930 | 0.059 | 0.081 | 0.15 |
| **Weighted** | | | **0.8390** | **0.318** | **0.739** | **0.164** | **0.232** | |

> **Note:** Slot 2 (1201–1800 IST) is the operationally ready window — it captures the peak afternoon convective period at Bengaluru Airport. Slots 0, 1, and 3 show strong CV AUROC (0.86–0.94) but high FAR due to low positive rates and shared daily ERA5 features. Performance will improve significantly once 6-hourly ERA5 features are integrated (A5).

---

## SHAP Feature Importance

Top 5 features per slot from SHAP analysis on the 2023–2025 test set:

| Slot | #1 | #2 | #3 | #4 | #5 |
|------|----|----|----|----|-----|
| 0 (Night) | LABEL_lag1 | CAPE | LIFTED_INDEX | DRNRF | RF |
| 1 (Morning) | RF | ts_any_yesterday | CAPE | ERA5_u_700hPa | DRNRF |
| **2 (Afternoon)** | **CAPE** | **TOTALS_TOTALS** | **K_INDEX** | **ERA5_T2M** | **MIN** |
| 3 (Evening) | K_INDEX | CAPE | TOTALS_TOTALS | ERA5_u_700hPa | LIFTED_INDEX |

Key finding: Slot 2 is driven by thermodynamic instability (CAPE, Totals-Totals, K-Index) — consistent with afternoon convective development. Nocturnal slot (0) is dominated by persistence from previous day. Sub-daily ERA5 is needed to differentiate Slot 2 from Slot 3.

---

## Repository Structure

```text
CSIR_Thunderstorm/
│
├── data/
│   ├── bengaluru_thunderstorm_features_merged.csv   # daily features (54 cols)
│   ├── bengaluru_6hr_labels.csv                     # 6-hr slot labels (15,276 rows)
│   ├── bengaluru_6hr_training_dataset.csv           # merged training dataset
│   ├── bengaluru_thunderstorm_features.csv
│   ├── bengaluru_thunderstorm_features_v2.csv
│   └── vidhi_stability_indices.csv
│
├── models/
│   ├── thunderstorm_model.pkl                       # daily XGBoost model
│   ├── nowcast_6hr_xgb_v1.pkl                      # combined 6-hr model
│   ├── nowcast_slot0_xgb.pkl                        # slot 0 model (0001-0600)
│   ├── nowcast_slot1_xgb.pkl                        # slot 1 model (0601-1200)
│   ├── nowcast_slot2_xgb.pkl                        # slot 2 model (1201-1800)
│   └── nowcast_slot3_xgb.pkl                        # slot 3 model (1801-2400)
│
├── results/
│   ├── evaluation_results.csv                       # daily model metrics
│   ├── evaluation_results_6hr.csv                   # combined 6-hr metrics
│   ├── evaluation_results_per_slot.csv              # per-slot metrics
│   ├── shap_per_slot_importance.csv                 # SHAP values all slots
│   ├── slot_model_summary.txt
│   └── shap_figures/
│       ├── shap_bar_slot{0-3}.png                   # bar charts per slot
│       ├── shap_dot_slot{0-3}.png                   # dot plots per slot
│       └── shap_heatmap_all_slots.png               # cross-slot heatmap
│
├── A1_feature_engineering.py    # builds 6-hr training dataset
├── A2_train_model.py            # trains combined 6-hr XGBoost
├── A3_slot_models.py            # trains one model per slot
├── A4_shap_analysis.py          # SHAP analysis for all slot models
├── baseline_model.py
├── tune_model.py
├── evaluate.py
├── predict.py
├── shap_analysis.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Aprameya05/CSIR-Thunderstorm-Bengaluru.git
cd CSIR-Thunderstorm-Bengaluru
pip install -r requirements.txt
```

---

## Usage

### Daily prediction
```bash
python predict.py
```

### 6-Hour nowcast (coming with predict_nowcast.py)
```bash
python predict_nowcast.py --date 2025-06-15
```

### API
```bash
uvicorn src.api:app --reload
```
Swagger UI: `http://127.0.0.1:8000/docs`

---

## REST API

### Daily — POST `/predict`

```json
{
  "date": "2026-07-15",
  "MAX": 29.0,
  "MIN": 21.0,
  "AW": 4,
  "RF": 2.1,
  "SSH": 180,
  "RF_lag1": 0,
  "MAX_lag1": 28.5,
  "MIN_lag1": 20.8,
  "LABEL_lag1": 0
}
```

### 6-Hour Nowcast — POST `/nowcast/predict/slot/{slot_id}` *(in development)*

```json
{
  "date": "2026-07-15",
  "slot": 2,
  "MAX": 29.0,
  "MIN": 21.0,
  "ERA5_CAPE": 1240.5,
  "ERA5_T2M": 299.1
}
```

### Response

```json
{
  "date": "2026-07-15",
  "slot": 2,
  "slot_label": "1201-1800 IST",
  "ts_probability": 0.61,
  "ts_predicted": true,
  "threshold_used": 0.33
}
```

---

## Development Roadmap

### Phase 1 — Daily Model ✅
- [x] Surface data preprocessing
- [x] Feature engineering
- [x] Baseline ML models (LR, RF, LightGBM, XGBoost)
- [x] Hyperparameter optimisation (Optuna)
- [x] Upper-air stability indices (IGRA)
- [x] ERA5 daily feature integration
- [x] SHAP explainability
- [x] WMO verification metrics
- [x] FastAPI REST API

### Phase 2 — 6-Hour Nowcasting 🔄
- [x] 6-hr label construction from IMD G-codes
- [x] 6-hr feature engineering pipeline (A1)
- [x] Combined 6-hr XGBoost model (A2)
- [x] Per-slot XGBoost models with Optuna (A3)
- [x] Per-slot SHAP analysis (A4)
- [ ] Prediction script — predict_nowcast.py
- [ ] ERA5 6-hourly feature integration (A5) — awaiting data
- [ ] Retrain slot models with sub-daily ERA5 (A5)
- [ ] Nowcast API endpoints (Satvik)

### Phase 3 — Operations 📋
- [ ] Real-time GFS/NCUM data pipeline (Atul)
- [ ] Operational dashboard (Sneha)
- [ ] Ceilometer integration
- [ ] LSTM temporal model

---

## Team

| Member | Role | Current Task |
|--------|------|-------------|
| **Aprameya Bharadwaj** | ML Lead | A1–A4 complete, writing predict_nowcast.py |
| **Atul** | Upper-Air + Real-Time Pipeline | Scoping GFS/NCUM data latency |
| **Satvik** | FastAPI Deployment | Scaffolding nowcast API endpoints |
| **Vidhi** | ERA5 Data | Downloading 6-hourly ERA5 2015–2025 |
| **Sneha** | Visualization & EDA | Building analysis charts |

---

## Acknowledgements

Developed under the guidance of **Dr. Geeta Agnihotri (Scientist F, IMD Bengaluru)** as part of a CSIR research initiative for operational thunderstorm prediction at Kempegowda International Airport.

---

## License

This repository is intended for academic and research purposes.  
Copyright © 2026 CSIR Thunderstorm Prediction System. All rights reserved.