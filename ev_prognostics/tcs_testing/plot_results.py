import pandas as pd
import matplotlib.pyplot as plt
import os

print("Loading predictions and truth data...")
script_dir = os.path.dirname(os.path.abspath(__file__))

# Pointing to the new ABS specific files
pred_file = os.path.join(script_dir, "chronos_predictions.csv")
truth_file = os.path.join(script_dir, "actual_hidden_truth.csv")

pred_df = pd.read_csv(pred_file)
truth_df = pd.read_csv(truth_file)

pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
truth_df['timestamp'] = pd.to_datetime(truth_df['timestamp'])

trucks = pred_df['truck_id'].unique()
print(f"Generating anomaly charts for {len(trucks)} truck(s)...")

for i, truck in enumerate(trucks):
    truck_pred = pred_df[pred_df['truck_id'] == truck]
    truck_truth = truth_df[truth_df['truck_id'] == truck]
    
    plt.figure(figsize=(14, 7))
    
    # Safely handle the column name whether it has the _avg suffix or not
    brake_col = 'brake_pedal_avg' if 'brake_pedal_avg' in truck_truth.columns else 'brake_pedal'
    
    # 1. Plot the ACTUAL Brake Pedal Usage (The Reality)
    plt.plot(truck_truth['timestamp'], truck_truth[brake_col], 
             label='Actual Reality (Brake Pedal %)', color='red', linewidth=2)
    
    # 2. Plot the AI's FORECASTED Brake Pedal Usage (The Baseline)
    plt.plot(truck_pred['timestamp'], truck_pred['0.5'], 
             label='AI Safe Forecast (P50)', color='blue', linewidth=2, linestyle='--')
    
    # 3. Plot the Confidence Interval
    plt.fill_between(truck_pred['timestamp'], truck_pred['0.1'], truck_pred['0.9'], 
                     color='blue', alpha=0.2, label='AI Normalcy Zone (P10 - P90)')
    
    # 4. Highlight the exact moment of the ABS Fault
    fault_time = truck_truth['timestamp'].max()
    plt.axvline(x=fault_time, color='black', linestyle=':', linewidth=2, label='ABS Fault Triggers')
    
    plt.title(f"ABS Anomaly Detection: AI Forecast vs Reality\nTruck: {truck}")
    plt.xlabel("Time (Final 120 Minutes)")
    plt.ylabel("Brake Pedal Position (%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    chart_name = os.path.join(script_dir, f"brake_system_failed_anomaly_result_{i+1}.png")
    plt.savefig(chart_name, bbox_inches='tight')
    plt.close()

print(f"\nSUCCESS! Charts have been saved to {script_dir}")
print("Look for 'brake_system_failed_anomaly_result_1.png'. Did the red line break out of the blue zone?")