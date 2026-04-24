Here is the fully improved README.md. I have added a clean repository map based on the files in your screenshot, updated the file descriptions to match your actual directory, and polished the formatting so it reads like a top-tier open-source project.

Just copy this entire block and press Enter in your terminal:

Bash
cat << 'EOF' > README.md
<div align="center">
  <h1>⚡ EV Predictive Maintenance & Anomaly Detection</h1>
  <p><i>Forecasting critical vehicle subsystem failures before they happen using Time-Series AI.</i></p>
</div>

---

## 📖 Overview
This repository contains an end-to-end predictive maintenance pipeline for electric vehicle (EV) telemetry. 

Instead of attempting to forecast unpredictable binary software flags (which time-series models struggle to read), this pipeline leverages **Amazon's Chronos-2** to forecast **continuous physical covariates** (e.g., brake pedal position, steering angle, IMU metrics). An impending fault is flagged when the physical telemetry violently breaks out of the AI's expected safe baseline, warning us *before* the hardware or software triggers a discrete error code.

## 🏗️ Architecture Pipeline

* **🔍 1. Data Extraction (TimescaleDB):** Automatically hunts for target faults (e.g., `Brake_System_Failed`) and pulls a deep context window (5–28 days).
* **🔄 2. Data Engineering:** Pivots native "Long Format" telemetry into an ML-ready "Wide Matrix".
* **⏱️ 3. Temporal Locking & Imputation:** Enforces a strict time grid (1m/5m) and applies TS standards (`ffill` / `bfill`) without leaking future data.
* **🧠 4. AI Inference:** Uses `amazon/chronos-t5-small` (via AutoGluon) to forecast the physical baseline using multivariate covariates.
* **📊 5. Visualization:** Generates Matplotlib charts comparing Actual Reality (Red) vs AI Forecast (Blue).

---

## 🗺️ Repository Structure

```text
.
├── AutogluonModels/                   # Saved AutoGluon model artifacts and training metadata
├── abs_wide_matrix_training_data.csv  # Cleaned, Wide-Format matrix ready for ML training
├── chronos_actual_truth.csv           # Pipeline Output: Ground truth telemetry values
├── chronos_univariate_predictions.csv # Pipeline Output: AI baseline predictions
├── chronos2_backtest_leo.ipynb        # 👑 PRIMARY: Multivariate Chronos-2 & AutoGluon backtesting
├── chronos_backtest.ipynb             # SECONDARY: Original univariate Chronos backtest notebook
├── data_cleaner.py                    # Utility: Memory-safe chunked CSV data cleaning
├── extract.py                         # Legacy: Standalone SQL extraction script
├── faults.go                          # Reference: Maps hex codes (0x3) to English fault names
├── get_tcs_snippets.py                # Utility: Extracts isolated matrices for TCS anomalies
├── plot.py                            # Legacy: Standalone plotting script
├── run_chronos.py                     # Legacy: Standalone HuggingFace Chronos inference script
└── README.md                          # This documentation