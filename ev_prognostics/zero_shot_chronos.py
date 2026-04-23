import os
import pandas as pd
from chronos import Chronos2Pipeline

# 1. Load the Base Foundation Model directly into memory (No training!)
print("Loading Chronos-2 Foundation Model...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda" # Change to "cpu" if you don't have a GPU
)

# 2. Prepare Context Data (Historical Targets + Past Covariates)
# This data triggers the "Time Attention" and "Group Attention" mechanisms
context_df = pd.DataFrame({
    "truck_id": ["EV_001", "EV_001", "EV_002", "EV_002"],
    "timestamp": pd.to_datetime(["2026-03-01 08:00", "2026-03-01 09:00", "2026-03-01 08:00", "2026-03-01 09:00"]),
    "S_breaking_score": [0.12, 0.14, 0.55, 0.56],     # The Target we want to predict
    "steering_angle_deg": [4.5, -2.1, 12.0, 11.5],    # Past Covariate (Sensor)
    "ambient_temp": [72.5, 73.1, 72.5, 73.1]          # Past Covariate (Weather)
})

# 3. Prepare Future Data (Known Future Covariates)
# What do we deterministically know about the future forecast horizon?
future_df = pd.DataFrame({
    "truck_id": ["EV_001", "EV_001", "EV_002", "EV_002"],
    "timestamp": pd.to_datetime(["2026-03-01 10:00", "2026-03-01 11:00", "2026-03-01 10:00", "2026-03-01 11:00"]),
    "ambient_temp": [74.0, 75.2, 74.0, 75.2]          # e.g., We know the weather forecast!
})

# 4. Execute Direct Multi-Step Forecasting
print("Generating Probabilistic Forecasts...")
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=2,               # How many steps into the future to predict
    quantile_levels=[0.05, 0.5, 0.95], # P05 (Worst Case), P50 (Expected), P95 (Best Case)
    id_column="truck_id",
    timestamp_column="timestamp",
    target="S_breaking_score",
    cross_learning=True,               # THE PDF'S SECRET WEAPON
    batch_size=100
)

print(pred_df)