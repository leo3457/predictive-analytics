import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import torch
import warnings
warnings.filterwarnings('ignore')

print("1. Connecting to database...")
conn = psycopg2.connect(
    dbname="tsdb_transaction", user="vt_user_readonly",
    password="8lthoYeq5xarlwr3", host="timescale-replica.engr.harbingerplatform.com", port="5432"
)

TARGET_FAULT = 'ABS_Fault'
CONTEXT_HOURS = 132 # 5.5 Days

print(f"2. Searching for the most recent '{TARGET_FAULT}'...")
find_fault_sql = f"""
    SELECT deviceid AS truck_id, CAST(DATE_TRUNC('minute', time) AS TIMESTAMP) AS fault_time
    FROM fault_event 
    WHERE faultname = '{TARGET_FAULT}' 
    ORDER BY time DESC LIMIT 1;
"""
fault_df = pd.read_sql_query(find_fault_sql, conn)
target_truck = fault_df.iloc[0]['truck_id']
fault_time = fault_df.iloc[0]['fault_time']

print(f"-> Target Truck: {target_truck} | Fault Time: {fault_time}")

# Extract EVERY signal as raw "Long Format" data
print("3. Extracting ALL raw telemetry (Long Format)...")
extract_raw_sql = f"""
    SELECT bucket_1min AS timestamp, signal, max_value
    FROM normalized_signal_1min
    WHERE deviceid = '{target_truck}'
      AND bucket_1min >= (CAST('{fault_time}' AS TIMESTAMP) - INTERVAL '{CONTEXT_HOURS} HOURS')
      AND bucket_1min <= CAST('{fault_time}' AS TIMESTAMP);
"""
raw_df = pd.read_sql_query(extract_raw_sql, conn)
conn.close()
print(f"-> Extracted {len(raw_df)} rows of raw signal data.")





print("Pivoting data from Long to Wide format...")

# 1. Pivot the data: timestamps become rows, signals become columns
wide_df = raw_df.pivot_table(
    index='timestamp', 
    columns='signal', 
    values='max_value', 
    aggfunc='max'
)

# 2. Clean up column names
wide_df.columns = [str(col).lower() for col in wide_df.columns] 
# Drop any duplicate columns caused by lowercasing 
wide_df = wide_df.loc[:, ~wide_df.columns.duplicated()]

# --- PRO-LEVEL MISSING DATA HANDLING ---
# ffill(): Sample-and-hold the last known state (Industry standard)
# bfill(): Safely fill the very beginning of the timeline with the first known state
wide_df = wide_df.resample('1min').ffill().bfill().reset_index()

wide_df['truck_id'] = target_truck

print(f"SUCCESS: Created a Wide matrix with {len(wide_df)} rows and {len(wide_df.columns)} columns!")
display(wide_df.tail())
wide_df.to_csv("abs_wide_matrix_training_data.csv", index=False)
print("Saved wide matrix to CSV!")





from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd

ACTIVE_TARGET = 'ebfm_abs_ind'  # Or whatever signal Denis wants to test
PREDICTION_MINUTES = 64 

print(f"1. Preparing AutoGluon Data for {ACTIVE_TARGET} using all covariates...")

# AutoGluon requires a specific TimeSeriesDataFrame structure
ag_data = TimeSeriesDataFrame.from_data_frame(
    wide_df,
    id_column="truck_id",
    timestamp_column="timestamp"
)

# AutoGluon handles the time-split automatically!
train_data, test_data = ag_data.train_test_split(prediction_length=PREDICTION_MINUTES)

print(f"2. Training Chronos-2 Multivariate Model...")
# Set up the predictor. We do NOT specify known_covariates_names.
# AutoGluon will automatically use all other ~89 columns as past-only covariates!
predictor = TimeSeriesPredictor(
    prediction_length=PREDICTION_MINUTES,
    target=ACTIVE_TARGET,
    eval_metric="MASE"
).fit(
    train_data,
    hyperparameters={"Chronos2": {
        "Chronos2": {
            "model_path": "amazon/chronos-t5-small", 
            "batch_size": 4,             # Force tiny chunks into memory
            "cross_learning": False      # Disable the memory-heavy cross learning
        }
    }}, # Forces the use of the Chronos-2 model
    enable_ensemble=False,
    time_limit=120
)

print("3. Generating Multivariate Inference...")
# Generate the prediction using the covariates
ag_predictions = predictor.predict(train_data)

# Convert AutoGluon's output back to standard Pandas DataFrames for your Cell 4 Plotter
pred_df = ag_predictions.reset_index()
pred_df = pred_df.rename(columns={'mean': '0.5', '0.1': '0.1', '0.9': '0.9'})

# Re-create truth_df for plotting
truth_df = wide_df[wide_df['timestamp'] > (wide_df['timestamp'].max() - pd.Timedelta(minutes=PREDICTION_MINUTES))]

print("Inference complete! Saved predictions to DataFrame.")





# 1. Safely force the target to lowercase just for DataFrame lookups
safe_target = ACTIVE_TARGET.lower()

plt.figure(figsize=(14, 6))

# 2. Plot the Actual Signal Value (Red)
plt.plot(truth_df['timestamp'], truth_df[safe_target], 
         label=f'Actual Value ({ACTIVE_TARGET})', color='red', linewidth=2)

# 3. Plot the AI's Forecasted Signal Value (Blue)
plt.plot(pred_df['timestamp'], pred_df['0.5'], 
         label='AI Expected Baseline (P50)', color='blue', linewidth=2, linestyle='--')

# 4. Plot the Confidence Interval
plt.fill_between(pred_df['timestamp'], pred_df['0.1'], pred_df['0.9'], 
                 color='blue', alpha=0.2, label='AI Confidence Interval')

plt.title(f"Signal Forecast vs Reality: {ACTIVE_TARGET}")
plt.xlabel(f"Time (Final {PREDICTION_MINUTES} Minutes)")
plt.ylabel(f"Sensor Value: {ACTIVE_TARGET}")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.show()