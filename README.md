````markdown
<div align="center">

# ⛈️ CSIR Thunderstorm Prediction System

### AI-Based Operational Thunderstorm Forecasting for Bengaluru Airport (IMD Station 43295)

Predicting daily thunderstorm occurrence using **Machine Learning**, **Surface Meteorological Observations**, **Upper-Air Stability Indices**, and **ERA5 Reanalysis Data**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Primary%20Model-success)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

</div>

---

# Overview

The **CSIR Thunderstorm Prediction System** is an operational machine learning framework developed to forecast **daily thunderstorm occurrence** over **Kempegowda International Airport, Bengaluru (IMD Station 43295)**.

The system integrates meteorological observations with machine learning to provide probabilistic thunderstorm forecasts, explainable predictions, and operational decision support.

Unlike traditional threshold-based approaches, this framework combines temporal feature engineering, ensemble learning, and explainable AI to improve thunderstorm detection performance.

---

# System Architecture

<p align="center">
<img src="assets/architecture.png" width="900">
</p>

---

# Machine Learning Pipeline

<p align="center">
<img src="assets/pipeline.png" width="900">
</p>

---

# Repository Structure

```text
CSIR_Thunderstorm/
│
├── assets/
│   ├── banner.png
│   ├── architecture.png
│   ├── pipeline.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── feature_importance.png
│   └── swagger.png
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── thunderstorm_model.pkl
│
├── outputs/
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── confusion_matrix.png
│
├── src/
│   ├── preprocess.py
│   ├── baseline_model.py
│   ├── tune_model.py
│   ├── shap_analysis.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
│
├── requirements.txt
└── README.md
```

---

# Dataset

| Dataset | Period | Status |
|----------|--------|--------|
| IMD Surface Observations | 2015–2025 | ✅ Complete |
| University of Wyoming Upper-Air Soundings | 2015–2025 | 🚧 In Progress |
| ERA5 Surface & Pressure Levels | 2015–2025 | 🚧 In Progress |
| IMD Thunderstorm Event Records | 2015–2025 | 📄 Requested |
| Ceilometer Cloud Base Height | Feb 2026–Present | 📄 Requested |

### Dataset Statistics

| Metric | Value |
|---------|------:|
| Total Days | **3819** |
| Thunderstorm Days | **457** |
| Non-Thunderstorm Days | **3362** |
| Positive Class | **12.0%** |

---

# Feature Engineering

A total of **24 engineered predictors** are used.

### Surface Meteorological Features

- Maximum Temperature
- Minimum Temperature
- Diurnal Temperature Range
- Rainfall
- Sunshine Hours
- Evaporation
- Weather Codes

### Rolling Features

- 3-Day Rainfall
- 7-Day Rainfall
- 3-Day Temperature Average
- 3-Day DTR Average

### Lag Features

- Previous Day Rainfall
- Previous Day Maximum Temperature
- Previous Day Minimum Temperature
- Previous Thunderstorm Label

### Seasonal Features

- Month (Sin/Cos)
- Day of Year (Sin/Cos)
- Season Encoding

### Weather Flags

- High Humidity Indicator
- Rainfall Indicator

---

# Model Performance

| Model | AUROC | POD | FAR | CSI | HSS |
|------|------:|------:|------:|------:|------:|
| Logistic Regression | 0.818 | — | — | — | — |
| Random Forest | 0.842 | — | — | — | — |
| LightGBM | 0.799 | — | — | — | — |
| **XGBoost (Tuned)** | **0.887 (CV)** / **0.809 (Test)** | **0.510** | **0.742** | **0.207** | **0.247** |
| ERA5 Threshold Rule | — | 0.20 | — | — | — |

---

# ROC Curve

<p align="center">
<img src="assets/roc_curve.png" width="700">
</p>

---

# Experimental Results

| Metric | Value |
|---------|------:|
| Cross Validation AUROC | **0.887** |
| Test AUROC | **0.809** |
| Probability of Detection | **0.510** |
| Critical Success Index | **0.207** |
| Heidke Skill Score | **0.247** |
| Positive Samples | **457** |
| Negative Samples | **3362** |

The tuned XGBoost model detects approximately **2.5× more thunderstorms** than the baseline ERA5 threshold rule using only surface meteorological observations.

---

# Explainability

Model predictions are interpreted using **SHAP (SHapley Additive Explanations)**.

Top contributing variables:

- Diurnal Temperature Range
- Day of Year
- Previous Day Rainfall
- Three-Day Maximum Temperature Average
- Three-Day Rainfall

<p align="center">
<img src="assets/shap_summary.png" width="900">
</p>

---

# Feature Importance

<p align="center">
<img src="assets/feature_importance.png" width="800">
</p>

---

# Confusion Matrix

<p align="center">
<img src="assets/confusion_matrix.png" width="550">
</p>

---

# Installation

```bash
git clone https://github.com/<username>/CSIR_Thunderstorm.git

cd CSIR_Thunderstorm

pip install -r requirements.txt
```

---

# Usage

## 1. Data Preprocessing

```bash
python src/preprocess.py
```

## 2. Model Training

```bash
python src/tune_model.py
```

## 3. SHAP Analysis

```bash
python src/shap_analysis.py
```

## 4. Evaluation

```bash
python src/evaluate.py
```

## 5. Single Prediction

```bash
python src/predict.py
```

## 6. Launch API

```bash
uvicorn src.api:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

---

# REST API

## POST `/predict`

```json
{
  "date": "2026-07-13",
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

### Response

```json
{
  "date": "2026-07-13",
  "thunderstorm_probability": 0.403,
  "alert_level": "YELLOW",
  "message": "Thunderstorm probability: 40.3%"
}
```

---

## GET `/health`

```json
{
  "status": "ok",
  "model": "thunderstorm_v1"
}
```

---

# API Documentation

<p align="center">
<img src="assets/swagger.png" width="900">
</p>

---

# Project Roadmap

## Completed

- [x] Surface data preprocessing
- [x] Feature engineering
- [x] Baseline machine learning models
- [x] Hyperparameter optimization
- [x] SHAP explainability
- [x] WMO verification metrics
- [x] ERA5 benchmark comparison
- [x] FastAPI deployment
- [x] REST API

## In Progress

- [ ] Upper-air stability indices integration
- [ ] ERA5 pressure-level feature integration
- [ ] Hourly thunderstorm prediction
- [ ] Ceilometer cloud-base integration
- [ ] LSTM temporal forecasting
- [ ] Operational validation
- [ ] CSIR deployment

---

# Team

| Member | Responsibility |
|----------|----------------|
| **Aprameya Bharadwaj** | Machine Learning Pipeline, Feature Engineering, Model Development, SHAP Analysis, Evaluation |
| Atul | Upper-Air Data Processing |
| Satvik | FastAPI Development & Docker |
| Vidhi | ERA5 Data Acquisition |
| Sneha | Exploratory Data Analysis & Visualization |

---

# Future Work

- CAPE
- Lifted Index
- K-Index
- Total Totals Index
- Vertical Wind Shear
- Relative Humidity Profiles
- Temporal Deep Learning (LSTM)
- Transformer-Based Forecasting
- Ensemble Learning
- Real-Time Operational Deployment

---

# License

This repository is intended for academic and research purposes.

```
Copyright (c) 2026

All Rights Reserved.

This project was developed as part of ongoing thunderstorm prediction research.
Unauthorized commercial use or redistribution without permission is prohibited.
```
````
