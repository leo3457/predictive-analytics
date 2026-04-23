import pandas as pd
import psycopg2
import os

print("Connecting to database...")
# Plug in your database credentials here
conn = psycopg2.connect(
    dbname="tsdb_transaction",
    user="vt_user_readonly",
    password="8lthoYeq5xarlwr3",
    host="timescale-replica.engr.harbingerplatform.com",
    port="5432"
)

# --- Configuration ---
TARGET_FAULT = 'Brake_System_Failed'
HOURS_OF_CONTEXT = 12

print(f"Searching for the latest '{TARGET_FAULT}'...")

# --- AUTOMATICALLY FIND A GOOD TEST TRUCK ---
# We grab the most recent time ANY truck threw this specific fault
find_fault_sql = f"""
    SELECT DISTINCT 
        deviceid AS truck_id,
        CAST(DATE_TRUNC('minute', time) AS TIMESTAMP) AS fault_time
    FROM fault_event 
    WHERE faultname = '{TARGET_FAULT}'
    ORDER BY fault_time ASC
    LIMIT 1;
"""
faults_df = pd.read_sql_query(find_fault_sql, conn)

if faults_df.empty:
    print(f"Error: Could not find any '{TARGET_FAULT}' events.")
    conn.close()
    exit()

target_truck = faults_df.iloc[0]['truck_id']
fault_time = faults_df.iloc[0]['fault_time']
snippet_id = f"{target_truck}_FaultAt_{fault_time.strftime('%Y%m%d_%H%M')}"

print(f"Found ideal test case! Truck: {target_truck}")
print(f"Fault triggered at: {fault_time}. Extracting the preceding {HOURS_OF_CONTEXT} hours...")

extraction_sql = f"""
WITH Filtered_Telemetry AS (
    SELECT bucket_1min, signal, min_value, max_value, avg_value, first_value, last_value
    FROM normalized_signal_1min
    WHERE deviceid = '{target_truck}'
        AND bucket_1min >= (CAST('{fault_time}' AS TIMESTAMP) - INTERVAL '5' DAY)
        AND bucket_1min <= CAST('{fault_time}' AS TIMESTAMP)
        AND signal IN ('TPMSF_Tyre_Pressure_FL', 'TPMSF_Tyre_Pressure_FR', 
                        'VCU_AccelControl_Est', 'IMU_Accel_Sensor_Status', 
                        'IMU_Gyro_Sensor_Status', 'VCU_Brake_Pedal_Pos', 'VCU_Brake_Pedal_Pos')
        AND max_value > 0 
)
SELECT 
    '{snippet_id}' AS truck_id, 
    bucket_1min AS timestamp,
    
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FL' THEN min_value END) AS Tyre_Press_FL_min,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FL' THEN max_value END) AS Tyre_Press_FL_max,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FL' THEN avg_value END) AS Tyre_Press_FL_avg,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FL' THEN first_value END) AS Tyre_Press_FL_first,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FL' THEN last_value END) AS Tyre_Press_FL_last,

    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FR' THEN min_value END) AS Tyre_Press_FR_min,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FR' THEN max_value END) AS Tyre_Press_FR_max,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FR' THEN avg_value END) AS Tyre_Press_FR_avg,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FR' THEN first_value END) AS Tyre_Press_FR_first,
    MAX(CASE WHEN signal = 'TPMSF_Tyre_Pressure_FR' THEN last_value END) AS Tyre_Press_FR_last,
    
    MAX(CASE WHEN signal = 'VCU_AccelControl_Est' THEN min_value END) AS VCU_AccelControl_Est_min,
    MAX(CASE WHEN signal = 'VCU_AccelControl_Est' THEN max_value END) AS VCU_AccelControl_Est_max,
    MAX(CASE WHEN signal = 'VCU_AccelControl_Est' THEN avg_value END) AS VCU_AccelControl_Est_avg,
    MAX(CASE WHEN signal = 'VCU_AccelControl_Est' THEN first_value END) AS VCU_AccelControl_Est_first,
    MAX(CASE WHEN signal = 'VCU_AccelControl_Est' THEN last_value END) AS VCU_AccelControl_Est_last,

    MAX(CASE WHEN signal = 'IMU_Accel_Sensor_Status' THEN min_value END) AS IMU_Accel_min,
    MAX(CASE WHEN signal = 'IMU_Accel_Sensor_Status' THEN max_value END) AS IMU_Accel_max,
    MAX(CASE WHEN signal = 'IMU_Accel_Sensor_Status' THEN avg_value END) AS IMU_Accel_avg,
    MAX(CASE WHEN signal = 'IMU_Accel_Sensor_Status' THEN first_value END) AS IMU_Accel_first,
    MAX(CASE WHEN signal = 'IMU_Accel_Sensor_Status' THEN last_value END) AS IMU_Accel_last,

    MAX(CASE WHEN signal = 'IMU_Gyro_Sensor_Status' THEN min_value END) AS IMU_Gyro_min,
    MAX(CASE WHEN signal = 'IMU_Gyro_Sensor_Status' THEN max_value END) AS IMU_Gyro_max,
    MAX(CASE WHEN signal = 'IMU_Gyro_Sensor_Status' THEN avg_value END) AS IMU_Gyro_avg,
    MAX(CASE WHEN signal = 'IMU_Gyro_Sensor_Status' THEN first_value END) AS IMU_Gyro_first,
    MAX(CASE WHEN signal = 'IMU_Gyro_Sensor_Status' THEN last_value END) AS IMU_Gyro_last,

    MAX(CASE WHEN signal = 'VCU_Brake_Pedal_Pos' THEN min_value END) AS Brake_Pedal_min,
    MAX(CASE WHEN signal = 'VCU_Brake_Pedal_Pos' THEN max_value END) AS Brake_Pedal_max,
    MAX(CASE WHEN signal = 'VCU_Brake_Pedal_Pos' THEN avg_value END) AS Brake_Pedal_avg,
    MAX(CASE WHEN signal = 'VCU_Brake_Pedal_Pos' THEN first_value END) AS Brake_Pedal_first,
    MAX(CASE WHEN signal = 'VCU_Brake_Pedal_Pos' THEN last_value END) AS Brake_Pedal_last,

    MAX(CASE WHEN signal = 'WheelBasedVehicleSpeed' THEN min_value END) AS Wheel_Speed_min,
    MAX(CASE WHEN signal = 'WheelBasedVehicleSpeed' THEN max_value END) AS Wheel_Speed_max,
    MAX(CASE WHEN signal = 'WheelBasedVehicleSpeed' THEN avg_value END) AS Wheel_Speed_avg,
    MAX(CASE WHEN signal = 'WheelBasedVehicleSpeed' THEN first_value END) AS Wheel_Speed_first,
    MAX(CASE WHEN signal = 'WheelBasedVehicleSpeed' THEN last_value END) AS Wheel_Speed_last,

    MAX(CASE WHEN bucket_1min = CAST('{fault_time}' AS TIMESTAMP) THEN 1.0 ELSE 0.0 END) AS Target_TCS_ESC_Fault

FROM Filtered_Telemetry
GROUP BY bucket_1min
ORDER BY bucket_1min ASC;
"""

df_snippet = pd.read_sql_query(extraction_sql, conn)
conn.close()
    
print(f"Extracted {len(df_snippet)} raw minutes of data.")

# --- DATA ENGINEERING: CLEAN AND FILL ---
print("Cleaning data and bridging gaps...")
df_snippet['timestamp'] = pd.to_datetime(df_snippet['timestamp'])

# Resample to guarantee a perfect 1-minute grid, bridging any times the truck was parked
df_snippet = df_snippet.set_index('timestamp').resample('1min').ffill().reset_index()
df_snippet['truck_id'] = snippet_id
df_snippet.fillna(0.0, inplace=True) 

# Force the 1.0 target strictly at the exact fault time
df_snippet['target_fault'] = 0.0
df_snippet.loc[df_snippet['timestamp'] == fault_time, 'target_fault'] = 1.0

# --- 5. EXPORT ---
csv_name = f"{TARGET_FAULT}_single_truck_test.csv"
df_snippet.to_csv(csv_name, index=False)
print(f"\nSUCCESS! Saved pure, focused training data to '{csv_name}'.")