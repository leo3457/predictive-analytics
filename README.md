<div align="center">
  <h1>⚡ EV Predictive Maintenance & Anomaly Detection</h1>
  <p><i>Forecasting critical vehicle subsystem failures before they happen using Time-Series AI.</i></p>
</div>

---

## 📖 Overview
This repository contains an end-to-end predictive maintenance pipeline for electric vehicle (EV) telemetry. 

Instead of attempting to forecast unpredictable binary software flags, this pipeline leverages **Amazon's Chronos-2** to forecast **continuous physical covariates** (e.g., brake pedal position, steering angle, IMU metrics). An impending fault is flagged when the physical telemetry violently breaks out of the AI's expected safe baseline, warning us *before* the hardware or software triggers a discrete error code.

## 🏗️ Architecture Pipeline

* **🔍 1. Data Extraction (TimescaleDB):** Automatically hunts for target faults (e.g., `Brake_System_Failed`) and pulls a deep context window (5–28 days).
* **🔄 2. Data Engineering:** Pivots native "Long Format" telemetry into an ML-ready "Wide Matrix".
* **⏱️ 3. Temporal Locking & Imputation:** Enforces a strict time grid (1m/5m) and applies TS standards (`ffill` / `bfill`) without leaking future data.
* **🧠 4. AI Inference:** Uses `amazon/chronos-t5-small` (via AutoGluon) to forecast the physical baseline using multivariate covariates.
* **📊 5. Visualization:** Generates Matplotlib charts comparing Actual Reality (Red) vs AI Forecast (Blue).

---

## 🗺️ Repository Structure

* 📁 **AutogluonModels/** — Saved AutoGluon model artifacts and training metadata.
* 📄 **abs_wide_matrix_training_data.csv** — Cleaned, Wide-Format matrix ready for ML training.
* 📄 **chronos_actual_truth.csv** — Pipeline Output: Ground truth telemetry values.
* 📄 **chronos_univariate_predictions.csv** — Pipeline Output: AI baseline predictions.
* 📓 **chronos2_backtest_leo.ipynb** — 👑 **PRIMARY:** Multivariate Chronos-2 & AutoGluon backtesting.
* 📓 **chronos_backtest.ipynb** — **SECONDARY:** Original univariate Chronos backtest notebook.
* 🐍 **data_cleaner.py** — Utility: Memory-safe chunked CSV data cleaning.
* 🐍 **extract.py** — Legacy: Standalone SQL extraction script.
* 🐹 **faults.go** — Reference: Maps hex codes (`0x3`) to English fault names.
* 🐍 **get_tcs_snippets.py** — Utility: Extracts isolated matrices for TCS anomalies.
* 🐍 **plot.py** — Legacy: Standalone plotting script.
* 🐍 **run_chronos.py** — Legacy: Standalone HuggingFace Chronos inference script.

---

## 🚀 Installation & Setup

**1. Python Environment** Ensure you are using Python 3.10+ (e.g., activate your `keras_env`).

**2. Install Dependencies** ```bash
pip install pandas matplotlib psycopg2-binary torch chronos-forecasting autogluon

## Usage Guide
**To run the main pipeline:**

Open chronos2_backtest_leo.ipynb.

Set Target Fault: In Cell 1, define the failure to investigate (e.g., TARGET_FAULT = 'Brake_System_Failed').

Set Target Signal: In Cell 3, choose a continuous physical sensor (e.g., ACTIVE_TARGET = 'vcu_brake_pedal_pos'). Do not use binary/discrete flags.

Adjust Context Window: Modify CONTEXT_HOURS (Cell 1) and the pandas .resample() frequency (Cell 2).

Note: Chronos maxes out at 8,192 tokens. Use 1min for 5 days of context, or downsample to 5min for up to 28 days.

Run All Cells to generate your anomaly detection plots!
EOF