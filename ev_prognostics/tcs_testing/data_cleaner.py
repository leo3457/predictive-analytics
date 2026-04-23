import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Now we safely join that folder path to your CSV filenames
script_dir = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(script_dir, "Brake_System_Failed_single_truck_test.csv")
print(f"Looking for file at: {input_file}")

output_file = os.path.join(script_dir, "Brake_System_Failed_single_truck_test_cleaned.csv")

if os.path.exists(output_file):
    os.remove(output_file)

chunksize = 2500
previous_tail = None

print(f"Beginning memory-safe cleaning")

for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):

    # A. The "Bridge": Attach the last row of the previous chunk so ffill() doesn't break
    if previous_tail is not None:
        chunk = pd.concat([previous_tail, chunk], ignore_index=True)

    # B. Memory Optimization (DownCasting)
    float_cols = chunk.select_dtypes(include=['float64']).columns
    chunk[float_cols] = chunk[float_cols].astype('float32')

    # C. Forward Fill (Bridged perfectly across chunks)
    sensor_columns = [col for col in chunk.columns if col not in ['truck_id', 'timestamp', 'ABS_Fault']]
    chunk[float_cols] = chunk[float_cols].astype('float32')

    # D. Fill remaining absolute blanks with 0.0
    chunk.fillna(0.0, inplace=True)

    # E. Save the very last row to act as the bridge for the NEXT loop
    previous_tail = chunk.iloc[[-1]].copy()

    # F. Drop the Bridge row from THIS chunk (if not the first loop) so we don't write duplicates
    if i > 0:
        chunk = chunk.iloc[1:]

    # G. Append cleanly to the new CSV file
    mode = 'w' if i == 0 else 'a'
    header = True if i == 0 else False
    chunk.to_csv(output_file, mode=mode, header=header, index=False)

    print(f"Cleaned and saved chunk {i + 1} ({(i + 1) * 250000} rows processed)...")

print(f"\nSuccess! Your AI-ready dataset is saved as: {output_file}")