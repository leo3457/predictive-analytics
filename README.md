cat << 'EOF' > README.md
<div align="center">
  <h1>⚡ EV Predictive Maintenance & Anomaly Detection</h1>
  <p><i>Forecasting critical vehicle subsystem failures before they happen using Time-Series AI.</i></p>
</div>

---

## 📖 Overview
This repository contains an end-to-end predictive maintenance pipeline for electric vehicle (EV) telemetry. 

Instead of attempting to forecast unpredictable binary software flags, this pipeline leverages **Amazon's Chronos-2** to forecast **continuous physical covariates** (e.g., brake pedal position, steering angle, IMU metrics). An impending fault is flagged when the physical telemetry violently breaks out of the AI's expected safe baseline.

## 🏗️ Architecture Pipeline

* **🔍 1. Data Extraction (TimescaleDB):** Automatically hunts for target faults (e.g., `Brake_System_Failed`) and pulls a deep context window (5–28 days).
* **🔄 2. Data Engineering:** Pivots native "Long Format" telemetry into an ML-ready "Wide Matrix".
* **⏱️ 3. Temporal Locking & Imputation:** Enforces a strict time grid (1m/5m) and applies TS standards (`ffill` / `bfill`) without leaking future data.
* **🧠 4. AI Inference:** Uses `amazon/chronos-t5-small` (via AutoGluon) to forecast the physical baseline.
* **📊 5. Visualization:** Generates Matplotlib charts comparing Actual Reality (Red) vs AI Forecast (Blue).

## 📂 Key Files
| File | Description |
|------|-------------|
| `chronos2_backtest_leo.ipynb` | Primary end-to-end Jupyter Notebook (SQL, Pandas, Chronos-2, Plotting). |
| `get_tcs_snippets.py` | Script for extracting isolated snippet matrices. Path: ev_prognostics/tcs_testing/chronos-2-protypes/get_tcs_snippets.py|
| `data_cleaner.py` | Memory-safe chunked CSV cleaning utility. Path: ev_prognostics/tcs_testing/chronos-2-protypes/data_cleaner.py|
| `faults.go` | Reference map for software hex codes to English fault names. |

## 🚀 Installation & Setup

**1. Python Environment**
Ensure you are using Python 3.10+ (e.g., activate your `keras_env`).

**2. Install Dependencies**
```bash
pip install pandas matplotlib psycopg2-binary torch chronos-forecasting autogluon