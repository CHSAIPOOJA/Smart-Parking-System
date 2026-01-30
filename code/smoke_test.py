import csv
import time
import sys
import os

# Test CSV write functionality
csv_path = 'parking_data.csv'

print(f"Writing test data to {csv_path}...")

# Open CSV and write header + 3 test rows
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['free_slots', 'occupied_slots', 'total_slots', 'occupancy_percent', 'frame_number', 'timestamp'])
    
    # Write 3 test frames
    for frame_num in range(1, 4):
        free = 15 - frame_num
        occupied = frame_num
        total = 15
        occupancy = (occupied / total * 100)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        writer.writerow([free, occupied, total, f"{occupancy:.1f}", frame_num, timestamp])
        time.sleep(0.5)

print(f"✓ CSV file created and test data written.")

# Read back and display
print("\nCSV Content:")
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        content = f.read()
        print(content)
    print("✓ Smoke test passed!")
else:
    print("✗ CSV file not found!")
