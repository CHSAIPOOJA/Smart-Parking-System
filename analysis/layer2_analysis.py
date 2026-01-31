import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load your data
# -------------------------------
# df = pd.read_csv('parking_data.csv')  # replace with your CSV path
# df = pd.read_csv('../parking_data.csv')


# Get the absolute path to the CSV based on this script's location
# Get the absolute path to the CSV inside code/ folder
script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script
csv_path = os.path.join(script_dir, '..', 'code', 'parking_data.csv')  # code/ folder

# Load CSV with error handling
try:
    df = pd.read_csv(csv_path)
    print(f"CSV loaded successfully from: {csv_path}")
except FileNotFoundError:
    print(f"CSV not found! Checked path: {csv_path}")
    raise



# Convert timestamp to datetime
# df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])  # optional, removes any invalid timestamps


# Extract hour and day for analysis
# df['hour'] = df['timestamp'].dt.hour
# df['day'] = df['timestamp'].dt.day_name()
# Round timestamp to nearest 5-minute interval
df['time_5min'] = df['timestamp'].dt.floor('5T')  # '5T' = 5-minute intervals

# For day labels
df['day'] = df['timestamp'].dt.day_name()

# Average occupancy per 5-minute interval
peak_5min = df.groupby('time_5min')['occupancy_percent'].mean()
print("Peak 5-Minute Intervals (Average Occupancy %):")
print(peak_5min)


# -------------------------------
# Step 2: Peak Hours Analysis
# -------------------------------
# Average occupancy by hour
peak_hours = df.groupby('hour')['occupancy_percent'].mean()
print("Peak Hours Analysis (Average Occupancy % per Hour):")
print(peak_hours)

# Plot Peak Hours
plt.figure(figsize=(10,5))
plt.plot(peak_hours.index, peak_hours.values, marker='o')
plt.title("Average Parking Occupancy by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Occupancy (%)")
plt.grid(True)
plt.show()

# -------------------------------
# Step 3: Slot / Zone Utilization
# -------------------------------
# Assuming you have per-slot occupancy column named 'slot_id' and 'occupied' (1=occupied,0=free)
# If your CSV does not have per-slot data, skip this part
if 'slot_id' in df.columns and 'occupied' in df.columns:
    slot_usage = df.groupby('slot_id')['occupied'].mean()  # mean = fraction of time occupied
    print("\nSlot Utilization (fraction of time occupied):")
    print(slot_usage)

    # Identify underutilized slots (<30% occupied)
    underutilized = slot_usage[slot_usage < 0.3]
    print("\nUnderutilized Slots (<30% occupied):")
    print(underutilized)

    # Bar chart of slot utilization
    plt.figure(figsize=(12,5))
    slot_usage.sort_values().plot(kind='bar', color='skyblue')
    plt.title("Slot Utilization (Fraction of Time Occupied)")
    plt.xlabel("Slot ID")
    plt.ylabel("Occupancy Fraction")
    plt.show()
else:
    print("\nSlot-level occupancy data not available. Skipping slot utilization analysis.")

# -------------------------------
# Step 4: Heatmap of Occupancy by Day and Hour
# -------------------------------
heatmap_data = df.pivot_table(index='day', columns='hour', values='occupancy_percent', aggfunc='mean')
# Reorder days for readability
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(days_order)

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt=".1f")
plt.title("Heatmap of Parking Occupancy (%) by Day and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.show()
