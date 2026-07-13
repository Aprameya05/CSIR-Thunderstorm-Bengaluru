<div align="center">

# ⛈️ CSIR Thunderstorm Prediction System

### AI-Based Operational Thunderstorm Forecasting for Bengaluru Airport (IMD Station 43295)

Machine learning framework for predicting daily thunderstorm occurrence using **surface meteorological observations**, **feature engineering**, **ensemble learning**, and **explainable AI**.

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/XGBoost-Primary%20Model-success" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" />
</p>

</div>

---

## Overview

The **CSIR Thunderstorm Prediction System** is an operational machine learning framework developed to forecast **daily thunderstorm occurrence** over **Kempegowda International Airport, Bengaluru (IMD Station 43295)**.

The system combines meteorological observations with machine learning to generate **probabilistic thunderstorm forecasts** for operational decision support.

Current development focuses on **surface observations**, with future integration of **upper-air soundings**, **ERA5 reanalysis**, and **ceilometer observations**.

---

## Key Features

- Binary thunderstorm prediction
- Probability-based forecasting
- Operational alert levels
- Meteorological feature engineering
- Explainable AI using SHAP
- WMO verification metrics
- FastAPI REST API
- Modular machine learning pipeline

---

## System Workflow

```text
                    IMD Surface Observations
                               │
                               ▼
                     Data Preprocessing
                               │
                               ▼
                    Feature Engineering
                               │
                               ▼
                 Machine Learning Models
          (XGBoost • Random Forest • LightGBM)
                               │
                               ▼
                     Model Evaluation
                               │
                               ▼
                    SHAP Explainability
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
|----------|--------|:------:|
| IMD Surface Observations | 2015–2025 | ✅ Available |
| University of Wyoming Upper-Air Soundings | 2015–2025 | 🚧 In Progress |
| ERA5 Reanalysis | 2015–2025 | 🚧 In Progress |
| IMD Thunderstorm Records | 2015–2025 | 📄 Requested |
| Ceilometer Data | Feb 2026–Present | 📄 Requested |

### Dataset Summary

| Metric | Value |
|---------|------:|
| Total Days | **3,819** |
| Thunderstorm Days | **457** |
| Non-Thunderstorm Days | **3,362** |
| Positive Class | **12.0%** |

---

## Feature Engineering

| Category | Features |
|----------|----------|
| Surface Variables | MAX, MIN, DTR, AW, RF, EVP, DRNRF, SSH |
| Rolling Statistics | RF_3d, RF_7d, MAX_3d_avg, MIN_3d_avg, DTR_3d_avg |
| Lag Features | RF_lag1, MAX_lag1, MIN_lag1, LABEL_lag1 |
| Seasonal Features | MONTH_sin, MONTH_cos, DOY_sin, DOY_cos, SEASON |
| Weather Flags | HA_flag, RF_nonzero |

---

## Model Performance

| Model | AUROC | POD | FAR | CSI | HSS |
|------|------:|------:|------:|------:|------:|
| Logistic Regression | 0.818 | — | — | — | — |
| Random Forest | 0.842 | — | — | — | — |
| LightGBM | 0.799 | — | — | — | — |
| **XGBoost (Tuned)** | **0.887 (CV)** / **0.809 (Test)** | **0.510** | **0.742** | **0.207** | **0.247** |
| ERA5 Threshold Rule | — | 0.20 | — | — | — |

### Performance Summary

- **Cross Validation AUROC:** 0.887
- **Test AUROC:** 0.809
- **Probability of Detection:** 0.510
- **Critical Success Index:** 0.207
- **Heidke Skill Score:** 0.247

The tuned XGBoost model detects approximately **2.5× more thunderstorms** than the baseline ERA5 threshold-based approach using only surface observations.

---

## Explainability

Model interpretation is performed using **SHAP (SHapley Additive Explanations)**.

Top contributing variables include:

- Diurnal Temperature Range
- Day of Year
- Previous Day Rainfall
- Three-Day Maximum Temperature Average
- Three-Day Rainfall

---

## Repository Structure

```text
CSIR_Thunderstorm/
│
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── src/
│   ├── preprocess.py
│   ├── baseline_model.py
│   ├── tune_model.py
│   ├── shap_analysis.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/CSIR_Thunderstorm.git
cd CSIR_Thunderstorm
pip install -r requirements.txt
```

---

## Usage

```bash
python src/preprocess.py
python src/tune_model.py
python src/evaluate.py
python src/shap_analysis.py
python src/predict.py
uvicorn src.api:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## REST API

### POST `/predict`

```json
{
  "date":"2026-07-13",
  "MAX":29.0,
  "MIN":21.0,
  "AW":4,
  "RF":2.1,
  "SSH":180,
  "RF_lag1":0,
  "MAX_lag1":28.5,
  "MIN_lag1":20.8,
  "LABEL_lag1":0
}
```

### Response

```json
{
  "date":"2026-07-13",
  "thunderstorm_probability":0.403,
  "alert_level":"YELLOW",
  "message":"Thunderstorm probability: 40.3%"
}
```

---

## Development Roadmap

- [x] Surface data preprocessing
- [x] Feature engineering
- [x] Baseline machine learning models
- [x] Hyperparameter optimization
- [x] SHAP explainability
- [x] WMO verification metrics
- [x] FastAPI deployment
- [x] REST API
- [ ] Upper-air stability indices
- [ ] ERA5 feature integration
- [ ] Hourly forecasting pipeline
- [ ] Ceilometer integration
- [ ] Temporal deep learning (LSTM)
- [ ] Operational deployment

---

## Team

| Member | Contribution |
|----------|-------------|
| **Aprameya Bharadwaj** | Machine Learning Pipeline, Feature Engineering, Model Development, Explainability, Evaluation |
| Atul | Upper-Air Data Processing |
| Satvik | Backend Development & FastAPI |
| Vidhi | ERA5 Data Processing |
| Sneha | Exploratory Data Analysis & Visualization |

---

## License

This repository is intended for academic and research purposes.

Copyright © 2026. All rights reserved.
