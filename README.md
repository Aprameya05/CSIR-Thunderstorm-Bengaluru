<div align="center">

<img src="https://img.shields.io/badge/CSIR-Research%20Project-blue?style=for-the-badge" />

# ⛈️ CSIR Thunderstorm Prediction System

**AI-Powered Operational Thunderstorm Nowcasting · Bengaluru Airport (VOBL)**  
IMD Station 43295 · Kempegowda International Airport

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-v5%20Temporal%20%2B%20v4%20Ensemble-success" />
  <img src="https://img.shields.io/badge/Himawari--9-BT%20Correction%20AUROC%200.914-blueviolet" />
  <img src="https://img.shields.io/badge/FastAPI-Public%20API-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Cloudflare%20Pages-Live-F38020?logo=cloudflare&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-4--cron%20CI%2FCD-2088FF?logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Real--Time%20Explainability-red" />
  <img src="https://img.shields.io/badge/Status-Operational%20Slot%202%20%E2%9C%85-brightgreen" />
</p>

**[Live Dashboard](https://csir-thunderstorm-bengaluru.pages.dev)** &nbsp;·&nbsp; **[Live API](https://csir-thunderstorm-api.onrender.com)** &nbsp;·&nbsp; **[API Docs](https://csir-thunderstorm-api.onrender.com/docs)**

</div>

---

## What This System Does

The CSIR Thunderstorm Prediction System is a fully operational AI nowcasting platform that predicts thunderstorm probability over Bengaluru Airport (VOBL/IMD 43295) across four 6-hour windows every day. It ingests real-time atmospheric data from GFS NOMADS, Himawari-9 satellite (Band 13 IR), live METAR from aviationweather.gov, and GFS upper-air soundings — runs a multi-model ensemble trained on 2015–2025 IMD data — applies a satellite-driven brightness temperature correction layer — computes real-time SHAP explanations — and deploys through a live public dashboard via Cloudflare Pages, four times daily, with zero human intervention.

**The mission:** An operational system that IMD and air traffic control can put their name on.

---

## 🎯 Forecast Slots & Model Performance

Each slot uses the empirically best production model from walk-forward validation on the 2024–2025 holdout set. Thresholds are F-beta (β=2) optimised — misses are penalised twice as heavily as false alarms because a missed storm is an aviation safety event.

| Slot | Window (IST) | Period | Production Model | CV AUROC | Threshold | Status |
|:----:|:------------:|:------:|:----------------:|:--------:|:---------:|:------:|
| 0 | 00:01–06:00 | Late Night | v4 Ensemble | 0.8484 | 0.24 | Monitoring |
| 1 | 06:01–12:00 | Morning | **v5 Temporal** | 0.8317 | 0.15 | Monitoring |
| **2** | **12:01–18:00** | **Afternoon** | **v5 Temporal** | **0.8710** | **0.16** | **Operational ✅** |
| 3 | 18:01–24:00 | Evening | **v5 Temporal** | 0.8710 | 0.39 | Monitoring |

**October post-monsoon fix:** Slot 2 threshold is automatically lowered from 0.16 → 0.10 in October only. This recovers POD from 0.379 → 0.621 (DOY_sin suppression artefact, confirmed via SHAP analysis).

**Correction model:** Himawari-9 Band 13 BT correction layer (CV AUROC 0.9141) applied on top of XGBoost base model. Real Himawari BT features from Atul's archive (1056 dates, 565 successful fetches).

---

## Pipeline Architecture

```
GitHub Actions (4 crons/day)
        │
        ▼
gfs_fetcher.py ──────── NOAA NOMADS (GFS 0.25°)
        │                   - Stability indices → upperair_realtime_43295.csv
        │                   - TMP f006/f012/f018/f024 → gfs_realtime_43295.csv
        │                   - f024/f048 outlook → gfs_multiday_43295.json
        ▼
fetch_himawari_realtime.py ── NOAA S3 / JAXA / NICT
        │                   - Himawari-9 Band 13 BT → himawari_realtime.json
        ▼
forecast_action.py ─────── XGBoost v5 Temporal / v4 Ensemble
        │                   - 4-slot probability forecast
        │                   - Instability score + synoptic regime
        │                   - Historical analog search (2015–2025)
        │                   - Airport impact assessment
        │                   - Pipeline health → pipeline_health.json
        ▼
fetch_metar.py ─────────── aviationweather.gov
        ▼
compute_realtime_shap.py ── SHAP TreeExplainer
        ▼
verify_today.py ─────────── IMD VOBL observations
        │                   - POD/FAR/HSS/Brier/CSI (rolling 30-day)
        ▼
forecast.json ──────────── Cloudflare Pages (live deploy)
```

**Pipeline run time:** ~58 seconds end-to-end. Runs at 4 scheduled times per day (see below).

---

## ⏰ 4-Cron Schedule (GitHub Actions)

| UTC Cron | IST Time | Slot | GFS Cycle |
|:--------:|:--------:|:----:|:---------:|
| `45 17 * * *` | 23:15 IST (prev day) | Slot 0 | prev-day 06Z f012 |
| `45 23 * * *` | 05:15 IST | Slot 1 | prev-day 12Z f012 |
| `45  5 * * *` | 11:15 IST | Slot 2 ★ | prev-day 18Z f012 |
| `15 10 * * *` | 15:45 IST | Slot 3 | same-day 00Z f012 |
| `15 16 * * *` | 21:45 IST | Extra | dashboard refresh |

Cycle selection follows the **t+12 rule**: fetch the GFS cycle 12 hours before the slot valid time, giving a ~6.5h posting buffer before the slot window opens.

---

## Data Sources

| Source | Variables | Latency | Cost |
|--------|-----------|---------|------|
| NOAA NOMADS (GFS 0.25°) | CAPE, CIN, K-Index, LI, TT, PW, T2m, winds, TMP f006–f048 | ~4h after cycle | Free |
| Himawari-9 Band 13 via NOAA S3 | BT (10.4µm) at VOBL ± 50km | ~10 min | Free |
| aviationweather.gov METAR | T, Td, wind, visibility, ceiling, TS flag | ~1h | Free |
| NASA GPM IMERG Early | Precipitation corroboration | ~4h | Free (Earthdata) |
| IMD VOBL surface obs | Daily TH flag (thunderstorm) | Daily | Institutional |

---

## Key Files

```
forecast_action.py          — main pipeline (4-slot XGBoost inference + all sections)
gfs_fetcher.py              — GFS NOMADS fetcher (upper-air + TMP multi-hour + multiday)
fetch_himawari_realtime.py  — Himawari-9 Band 13 via satpy (NOAA S3 → JAXA → NICT)
fetch_metar.py              — METAR from aviationweather.gov (VOBL + VOBG fallback)
fetch_imerg_realtime.py     — GPM IMERG precipitation corroboration
compute_realtime_shap.py    — SHAP TreeExplainer (top 12 features per slot)
verify_today.py             — Daily IMD verification (POD/FAR/HSS/Brier/CSI, 30-day rolling)
resave_models.py            — XGBoost version-agnostic model resave (Booster.save_model .ubj)
october_threshold_fix.py    — October Slot 2 threshold override (DOY_sin artefact fix)
radar_router.py             — FastAPI endpoints: /radar/proximity /radar/history /radar/image
run_radar_scheduler.py      — Python scheduler for Himawari + IMERG (Atul's machine)
run_himawari.bat            — Windows Task Scheduler wrapper (Atul's machine, every 10 min)
forecast.json               — Live forecast output (auto-committed by GitHub Actions)
.github/workflows/forecast_update.yml — Actions workflow (4 crons + deploy)
```

---

## forecast.json Schema

```json
{
  "date": "YYYY-MM-DD",
  "generated_at": "YYYY-MM-DD HH:MM IST",
  "alert_active": true,
  "peak_slot": 2,
  "peak_probability": 0.312,
  "model_version": "v5_temporal_v4_ensemble",
  "slots": [
    {
      "slot": 2, "label": "Afternoon", "time": "1201-1800 IST",
      "ts_probability": 0.312, "ts_predicted": true,
      "threshold": 0.16, "primary": true,
      "source": "gfs+upperair", "model_used": "nowcast_slot2_xgb_v5_temporal.pkl",
      "model_version": "v5_temporal", "raw_probability": 0.298,
      "cape": 1240.5, "k_index": 36.2, "lifted_index": -2.4, "totals_totals": 48.1,
      "trend": "up", "trend_diff": 0.045, "prev_probability": 0.267
    }
  ],
  "met_parameters": {
    "ua_cape_jkg": 1240.5, "ua_k_index": 36.2,
    "ua_lifted_index": -2.4, "ua_totals_totals": 48.1,
    "ERA5_u_500hPa": 5.2, "ERA5_v_500hPa": 1.8,
    "ERA5_u_850hPa": -3.1, "ERA5_v_850hPa": 2.4,
    "instability_level": "Moderate"
  },
  "gfs_tmax_c": 31.4,
  "gfs_tmin_c": 21.8,
  "gfs_rainfall_mm": 4.2,
  "gfs_valid_date": "11 Aug 2026",
  "gfs_cycle": "2026-08-10 18Z f012",
  "satellite": {
    "himawari9": {
      "min_bt_50km": -52.3, "storm_detected": true,
      "alert_level": "ORANGE", "cold_pixels_count": 47,
      "data_source": "Himawari-9 Band 13 (10.4µm) via NOAA AWS S3"
    },
    "history": [...]
  },
  "convective_initiation": {
    "instability_score": 58.4, "initiation_risk": "MODERATE",
    "initiation_status": "PRE-CONVECTIVE", "hours_to_peak": 2.3,
    "cape_now": 1240.5, "ki_now": 36.2, "peak_window_ist": "1300-1800 IST"
  },
  "multiday_outlook": [...],
  "analogs": {...},
  "airport_impact": {
    "total_disrupted_est": 12, "overall_risk": "MODERATE",
    "slots": [...]
  },
  "synoptic_regime": {
    "regime_id": "R4", "regime_name": "Strong Solar Heating",
    "ts_rate": 9.8, "auroc": 0.900
  },
  "verification": {
    "pod": 0.714, "far": 0.286, "hss": 0.523,
    "brier": 0.0812, "csi": 0.556, "window": "30-day"
  },
  "pipeline_health": {...},
  "realtime_shap": {...},
  "metar": {...}
}
```

---

## Model Inventory

| File | Type | CV AUROC | Notes |
|------|------|:--------:|-------|
| `nowcast_slot0_xgb_v4_ensemble.pkl` | XGBoost Ensemble | 0.8484 | Slot 0 production |
| `nowcast_slot1_xgb_v5_temporal.pkl` | XGBoost v5 Temporal | 0.8317 | Slot 1 production, 30 lag features |
| `nowcast_slot2_xgb_v5_temporal.pkl` | XGBoost v5 Temporal | 0.8710 | Slot 2 production ★ |
| `nowcast_slot3_xgb_v5_temporal.pkl` | XGBoost v5 Temporal | 0.8710 | Slot 3 production |
| `himawari_correction_model.pkl` | XGBoost BT correction | 0.9141 | Real Himawari BT archive |
| `nowcast_slot*_xgb_v3_calibrated.pkl` | v3 calibrated | ~0.871 | Fallback |
| `nowcast_slot*_xgb_v2_calibrated.pkl` | v2 calibrated | ~0.85 | Final fallback |

**Model resave:** Run `python resave_models.py` on the training machine (or Colab) after any retraining. Uses `Booster.save_model(.ubj)` format — version-stable across XGBoost 2.x.

---

## Key Technical Facts

- `cfgrib` CAPE level name: `lev_entire_atmosphere_(considered_as_a_single_layer)`
- `satpy` reader: `ahi_hsd` for Himawari HSD files
- GFS requests require Chrome User-Agent header (bare requests → 403 from NOMADS)
- All GFS date logic uses **UTC only** — never IST dates in NOMADS paths
- Monsoon phase ACTIVE TS rate: 28.0% vs BREAK: 3.5%
- Walk-forward 2024–2025 AUROC dip is sample size artefact (only 20 TS events in 2024)
- Deep learning (LSTM/CNN ~0.79 AUROC) underperforms XGBoost (0.871) — dataset too small
- Seasonal specialist models rejected — general model outperforms all seasons
- October DOY_sin suppression is structural — threshold fix is the correct approach

---

## Deployment

**Push workflow:**
```bash
git pull origin main                     # ALWAYS first
git add <files>
git commit -m "type: description"
git push origin main
```

**Conflict on forecast.json (non-fast-forward from Actions race):**
```bash
git pull origin main --no-rebase
git checkout --theirs forecast.json      # auto-generated, never edit manually
git add forecast.json
git commit -m "merge: resolve forecast.json conflict"
git push origin main --force-with-lease  # NEVER bare --force
```

**Cloudflare Pages:**
- Account ID: `d1ebec3d837d56d32a077b449a65f0c0`
- Project: `csir-thunderstorm-bengaluru`
- Secret: `CLOUDFLARE_API_TOKEN` (GitHub repo secrets, Pages:Edit permission)
- Deploys automatically after every `forecast.json` commit via `cloudflare/wrangler-action@v3`

**Trigger GitHub Action manually:**
GitHub repo → Actions → "Update Forecast JSON" → Run workflow

---

## WMO-Standard Verification Metrics (Slot 2, 30-day rolling)

| Metric | Formula | Target |
|--------|---------|--------|
| POD | Hits / (Hits + Misses) | ≥ 0.70 |
| FAR | False Alarms / (Hits + FA) | ≤ 0.35 |
| HSS | 2(hits·CN − miss·FA) / ((H+M)(M+CN) + (H+FA)(FA+CN)) | ≥ 0.45 |
| CSI | Hits / (Hits + Misses + FA) | ≥ 0.40 |
| Brier Score | Mean((P − O)²) | ≤ 0.08 |

Metrics computed daily by `verify_today.py` and injected into `forecast.json`.

---

## Free-Tier Stack

Everything runs at zero cost:
- **NOAA NOMADS** — GFS 0.25° GRIB2 (anonymous, no auth)
- **NOAA AWS S3** — Himawari-9 HSD (anonymous)
- **JAXA P-Tree** — Himawari fallback (HTTP, anonymous)
- **NASA Earthdata** — GPM IMERG (free account + token)
- **aviationweather.gov** — METAR API (no auth)
- **Cloudflare Pages** — static hosting (free tier)
- **GitHub Actions** — CI/CD (free for public repos)
- **Groq (Llama-3.3-70b-versatile)** — RAG explainability (free tier)
- **Render** — FastAPI backend (free tier, cold start ~30s)

---

## Team

| Member | Role |
|--------|------|
| **Aprameya** | ML Lead, project architect, final decision-maker |
| **Atul Denny** | Upper-air data, GFS pipeline, Himawari satellite fetching (Windows) |
| **Satvik** | FastAPI backend on Render |
| **Vidhi** | ERA5 historical data downloads |
| **Sneha** | Verification pipeline, visualisation |

**Institutional collaborator:** Dr. Geeta Agnihotri (Scientist F, IMD Bengaluru) — agreed to share raw 2026 VOBL surface observation data; potential joint technical report.

---

## Open Items (Priority Order)

**Immediate:**
1. Confirm Atul can load resaved models — run `python resave_models.py` on Aprameya's machine, push `.pkl` files
2. Set up Earthdata token on Atul's machine for GPM IMERG
3. Confirm 4-cron schedule is live in GitHub Actions

**Short Term:**
4. Retrain v6 models adding Himawari BT features (min_bt_50km, cold_pixels_count) to main XGBoost feature set
5. Historical BT backtest at 09:00 UTC (14:30 IST — peak convective hour) for better AUROC than current 0.622
6. Integrate Dr. Agnihotri's 2026 raw VOBL surface data when received
7. Add Damini lightning feed (Dr. Agnihotri) or WWLLN research agreement

**Medium Term:**
8. Build skill score trend chart (POD/FAR/HSS over time) — dashboard verification page
9. ATC View page — simplified single-screen for air traffic controllers (no ML jargon)
10. Mobile-responsive dashboard audit
11. API rate limiting + authentication for IMD production use
12. Automated model retraining pipeline when new IMD data arrives

---

*This is not a research paper. This is a real, deployable operational system for CSIR/IMD use.*
