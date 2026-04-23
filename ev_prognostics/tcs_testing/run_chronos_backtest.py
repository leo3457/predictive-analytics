import pandas as pd
from chronos import ChronosPipeline
import torch
import os

# --- PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
clean_data_path = os.path.join(script_dir, "Brake_System_Failed_single_truck_test_cleaned.csv")
predictions_path = os.path.join(script_dir, "chronos_predictions.csv")
truth_path = os.path.join(script_dir, "actual_hidden_truth.csv")

print("1. Loading the 12-hour active driving dataset...")
df = pd.read_csv(clean_data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Ensure all column names are lowercase to avoid Pandas/Postgres mismatch errors
df.columns = df.columns.str.lower()

truck_id = df['truck_id'].iloc[0]
print(f"-> Successfully loaded data for Truck: {truck_id}")

print("2. Preparing the 'Blind' Backtest...")
# We give the AI the first 10 hours, and ask it to predict the final 2 hours
PREDICTION_MINUTES = 120

fault_time = df['timestamp'].max()
cutoff_time = fault_time - pd.Timedelta(minutes=PREDICTION_MINUTES)

# Context: The first 10 hours (Chronos sees this)
context_df = df[df['timestamp'] <= cutoff_time]

# Future: The last 2 hours (Chronos must predict this)
actual_future_df = df[df['timestamp'] > cutoff_time]

print("3. Loading Chronos-2 AI Model...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", 
    device_map="auto",         
    torch_dtype=torch.bfloat16,
)

print(f"4. Forecasting the final {PREDICTION_MINUTES} minutes of driving...")
pred_df = pipeline.predict_df(
    context_df,
    prediction_length=PREDICTION_MINUTES,             
    quantile_levels=[0.1, 0.5, 0.9],   
    id_column="truck_id",
    timestamp_column="timestamp",
    target="brake_pedal_avg"  # Forecasting the physical brake pedal rhythm!
)

print("5. Saving results to disk...")
pred_df.to_csv(predictions_path, index=False)
actual_future_df.to_csv(truth_path, index=False)

print(f"\nSUCCESS! Backtest complete. Files saved to {script_dir}")