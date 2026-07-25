<div align="center">

# ⛈️ CSIR Thunderstorm Prediction System

### AI-Based Operational Thunderstorm Forecasting for Bengaluru Airport (IMD Station 43295)

Machine learning framework for predicting thunderstorm occurrence using **surface meteorological observations**, **upper-air stability indices**, **ERA5 reanalysis (6-hourly)**, **Himawari-9 satellite imagery**, and **explainable AI** — with a fully operational **6-hour nowcasting system** and **live public dashboard**.

<p>
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/XGBoost-Primary%20Model-success" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Status-Operational-brightgreen" />
  <img src="https://img.shields.io/badge/Nowcasting-6--Hour%20Slots-orange" />
  <img src="https://img.shields.io/badge/ERA5-6--Hourly-blue" />
  <img src="https://img.shields.io/badge/Dashboard-Live-brightgreen" />
</p>

🌐 **Live Dashboard:** [csir-thunderstorm-bengaluru.pages.dev](https://csir-thunderstorm-bengaluru.pages.dev)
🔗 **Live API:** [csir-thunderstorm-api.onrender.com](https://csir-thunderstorm-api.onrender.com)
📖 **API Docs:** [csir-thunderstorm-api.onrender.com/docs](https://csir-thunderstorm-api.onrender.com/docs)

</div>

---

## Overview

The **CSIR Thunderstorm Prediction System** is an operational machine learning framework developed to forecast thunderstorm occurrence over **Kempegowda International Airport, Bengaluru (IMD Station 43295)**, in collaboration with **IMD Bengaluru (Dr. Geeta Agnihotri, Scientist F)**.

| Mode | Description | Status |
|------|-------------|:------:|
| **Daily Model** | Predicts whether a thunderstorm will occur on a given day | ✅ Complete |
| **6-Hour Nowcast** | Predicts thunderstorm probability for each 6-hour window | ✅ Operational |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│                                                                 │
│  GFS NOMADS (real-time)  │  Himawari-9 S3  │  IMD Upper-Air   │
│  CAPE, K-Index, LI, TT   │  Band 13 IR BT  │  Soundings       │
│  ERA5 Reanalysis          │  50km VOBL box  │  MetPy indices   │
└──────────────┬────────────┴────────┬────────┴──────────┬───────┘
               │                    │                    │
               ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                             │
│                                                                 │
│  forecast_action.py    fetch_himawari_realtime.py              │
│  54-feature vector     Storm proximity signal                  │
│  4 slot inference      himawari_realtime.json                  │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML INFERENCE LAYER                           │
│                                                                 │
│  nowcast_slot{0-3}_xgb_v3_calibrated.pkl                       │
│  Isotonic (Slots 2,3) │ Platt (Slots 0,1)                      │
│  Thresholds: 0.24 │ 0.38 │ 0.16 │ 0.39                        │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                 │
│                                                                 │
│  forecast.json → GitHub → Cloudflare Pages (auto-deploy)      │
│  FastAPI (Render) → /nowcast/predict/all │ /rag/explain        │
│  RAG (Groq Llama-3.3-70b) → Physical explanation engine       │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              LIVE DASHBOARD (React, Cloudflare Pages)           │
│                                                                 │
│  6-Hour Slot Probabilities │ Real Met Parameters               │
│  Himawari Storm Proximity  │ SHAP Explainability               │
│  AI RAG Chatbox            │ Model Verification Metrics        │
│  ATC View                  │ Synoptic Regime Panel             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6-Hour Slot Definition

| Slot | IST Window | ERA5 Snapshot | Base Rate | Status |
|------|-----------|---------------|-----------|--------|
| 0 | 0001–0600 | 00Z UTC | 3.7% | Monitoring |
| 1 | 0601–1200 | 06Z UTC | 1.1% | Monitoring |
| **2** | **1201–1800** | **12Z UTC** | **6.3%** | **Operational ✅** |
| 3 | 1801–2400 | 18Z UTC | 5.9% | Monitoring |

---

## Model Performance — v3 Calibrated (Production)

| Slot | AUROC | POD | FAR | CSI | HSS | Threshold | Calibration |
|------|:-----:|----:|----:|----:|----:|:---------:|:-----------:|
| 0 | 0.901 | 0.200 | 0.950 | 0.042 | 0.065 | 0.24 | Platt |
| 1 | 0.911 | 0.000 | 1.000 | 0.000 | — | 0.38 | Platt |
| **2** | **0.821** | **0.356** | **0.623** | **0.224** | **0.318** | **0.16** | **Isotonic** |
| 3 | 0.842 | 0.286 | 0.926 | 0.063 | 0.087 | 0.39 | Isotonic |

> **Key finding:** ERA5_CAPE jumped from rank 42 → rank 2 for Slot 3 with 6-hourly ERA5. The `cape_x_kindex` interaction term entered top 5 in 3 of 4 slots. Slot 3 Brier score improved 64% after isotonic calibration.

---

## SHAP Feature Importance (v3)

| Slot | #1 | #2 | #3 | #4 | #5 |
|------|----|----|----|----|-----|
| 0 | CAPE | LABEL_lag1 | LIFTED_INDEX | DRNRF | MIN_3d_avg |
| 1 | ts_any_yesterday | CAPE | RF | ERA5_q_700hPa | DRNRF |
| **2** | **cape_x_kindex** | **CAPE** | **TOTALS_TOTALS** | **K_INDEX** | **ERA5_T2M** |
| 3 | cape_x_kindex | ERA5_CAPE ↑ | K_INDEX | LIFTED_INDEX | thetae_850 |

---

## Real-Time Pipeline

### GitHub Actions (2x daily, fully automated)

| UTC | IST | Action |
|-----|-----|--------|
| 10:15 | 15:45 | Fetch → Forecast → Commit → Deploy to Cloudflare |
| 16:15 | 21:45 | Fetch → Forecast → Commit → Deploy to Cloudflare |

### Himawari-9 Satellite (Atul)

`fetch_himawari_realtime.py` — Band 13 (10.4μm IR) from NOAA AWS S3, segments S04+S05+S06, VOBL 50km bounding box:

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| Min BT | < −40°C | Storm cell present |
| Cold pixels | > 0 | Active deep convection |
| VOBL BT | — | Cloud-top temp directly overhead |

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Daily prediction |
| GET | `/nowcast/slots/info` | Slot metadata |
| POST | `/nowcast/predict/slot/{id}` | Single slot |
| POST | `/nowcast/predict/all` | All 4 slots |
| POST | `/rag/explain` | Llama-3.3-70b explanation |
| POST | `/rag/analogs` | Historical analog retrieval |

---

## Installation

```bash
git clone https://github.com/Aprameya05/CSIR-Thunderstorm-Bengaluru.git
cd CSIR-Thunderstorm-Bengaluru
pip install xgboost scikit-learn joblib shap pandas numpy fastapi uvicorn \
            requests metpy boto3 satpy pyresample pytz groq
```

---

## Usage

```bash
# Run nowcast
python predict_nowcast.py
python predict_nowcast.py --date 2023-10-11

# Run full pipeline
python run_daily_forecast_v2.py

# Export forecast.json
python forecast_json_exporter_v2.py

# Fetch satellite data
python fetch_himawari_realtime.py

# Start API
uvicorn main:app --reload
```

---

## Development Roadmap

### Phase 1 — Daily Model ✅
- [x] XGBoost with surface + upper-air + ERA5 (AUROC 0.871)
- [x] SHAP explainability
- [x] FastAPI REST API

### Phase 2 — 6-Hour Nowcasting ✅
- [x] Per-slot XGBoost v1/v2/v3 with calibration
- [x] ERA5 6-hourly 2015–2025 (Vidhi)
- [x] GFS real-time + upper-air MetPy (Atul)
- [x] RAG endpoints (Groq Llama-3.3-70b)

### Phase 3 — Operations ✅
- [x] Live React dashboard (Cloudflare Pages)
- [x] GitHub Actions auto-deploy (2x daily)
- [x] FastAPI public deployment (Render)
- [x] RAG chatbox live
- [x] Himawari-9 storm proximity fetcher (Atul)
- [x] Real met parameters on dashboard
- [x] Rolling verification (Sneha)

### Phase 4 — Enhancements 📋
- [ ] gfs_fetcher.py — all 4 slots real-time
- [ ] Himawari signal in forecast.json + dashboard panel
- [ ] Daily automated verification in GitHub Action
- [ ] LSTM temporal model
- [ ] IMD radar integration (pending Dr. Agnihotri)

---

## Team

| Member | Role | Contribution |
|--------|------|-------------|
| **Aprameya Bharadwaj** | ML Lead | A1–A5, pipeline, RAG, dashboard, architecture |
| **Atul Denny** | Pipeline Lead | GFS fetcher, upper-air MetPy, Himawari-9 fetcher |
| **Satvik** | Deployment | FastAPI, Render, RAG endpoints |
| **Vidhi** | ERA5 Data | 6-hourly ERA5 2015–2025 (16,072 rows) |
| **Sneha** | Verification | EDA, ForecastLogger, verification dashboard |

---

## Acknowledgements

Developed under the guidance of **Dr. Geeta Agnihotri (Scientist F, IMD Bengaluru)** as part of a CSIR research initiative for operational thunderstorm prediction at Kempegowda International Airport.

---

## License

For academic and research purposes only.
Copyright © 2026 CSIR Thunderstorm Prediction System. All rights reserved.
