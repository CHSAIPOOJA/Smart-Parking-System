print("Program started")

import sys
sys.path.append('.')
import csv
import time
import os
from datetime import datetime
import cv2
import pickle
import numpy as np
from skimage.transform import resize
from collections import defaultdict, deque

from util import get_parking_spots_bboxes

# Create CSV file to store per-frame parking data (append mode so data accumulates across runs)
# Columns: free_slots, occupied_slots, total_slots, occupancy_percent, frame_number, timestamp
csv_path = os.path.join(os.path.dirname(__file__), 'parking_data.csv')
file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
csv_file = open(csv_path, 'a', newline='')
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow(['free_slots', 'occupied_slots', 'total_slots', 'occupancy_percent', 'frame_number', 'timestamp'])

# Load trained model
# with open('../dataset/archive (1)/parking/model/model.p', 'rb') as f:
#     model = pickle.load(f)



# Always correct absolute path based on project folder
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  
    'dataset',
    'archive (1)',
    'parking',
    'model',
    'model.p'
)

MODEL_PATH = os.path.abspath(MODEL_PATH)
print("Model path:", MODEL_PATH)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load video and mask
# video = cv2.VideoCapture(r'../dataset/archive (1)/parking/parking_1920_1080_loop.mp4')
  

# Build absolute path for video
VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # go one folder up
    'dataset',
    'archive (1)',
    'parking',
    'parking_1920_1080_loop.mp4'
)
VIDEO_PATH = os.path.abspath(VIDEO_PATH)
print("Video path:", VIDEO_PATH)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}")
    raise FileNotFoundError(f"Video missing at: {VIDEO_PATH}")





# mask = cv2.imread(r'../dataset/archive (1)/parking/mask_1920_1080.png', 0)




# Build correct path for mask
MASK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'dataset',
    'archive (1)',
    'parking',
    'mask_1920_1080.png'
)

MASK_PATH = os.path.abspath(MASK_PATH)
print("Mask path:", MASK_PATH)

mask = cv2.imread(MASK_PATH, 0)

if mask is None:
    print("ERROR: Mask not found!")
    raise FileNotFoundError(f"Mask missing at: {MASK_PATH}")

print("Mask loaded successfully!")




# Get parking spot boxes
parking_spots = get_parking_spots_bboxes(mask)

print(f"\n{'='*60}")
print(f"Total parking spots detected: {len(parking_spots)}")
print(f"{'='*60}\n")

# Temporal smoothing: store predictions for last 5 frames per slot
HISTORY_SIZE = 5
CONFIDENCE_THRESHOLD = 0.4
slot_history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

frame_count = 0

try:
    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        free_count = 0

        for slot_idx, (x, y, w, h) in enumerate(parking_spots):
            crop = frame[y:y+h, x:x+w]
            
            # Check pixel intensity - ignore very dark/bright regions
            mean_intensity = np.mean(crop)
            if mean_intensity < 20 or mean_intensity > 240:
                # Skip extremely dark or bright regions
                final_prediction = 1  # Mark as occupied if can't determine
            else:
                # Use 15x15 exactly like the model was trained
                crop_resized = resize(crop, (15, 15, 3))
                crop_flat = crop_resized.flatten().reshape(1, -1)
                
                # Get prediction from model
                prediction = model.predict(crop_flat)[0]
                
                # Add to history for temporal smoothing
                slot_history[slot_idx].append(prediction)
                
                # Get majority vote from history (only use if we have enough history)
                if len(slot_history[slot_idx]) >= HISTORY_SIZE:
                    history_list = list(slot_history[slot_idx])
                    # Count 0s and non-0s for temporal smoothing
                    empty_votes = sum(1 for p in history_list if p == 0)
                    occupied_votes = len(history_list) - empty_votes
                    # Use majority vote
                    final_prediction = 0 if empty_votes > occupied_votes else 1
                else:
                    # During warmup phase, use direct prediction
                    final_prediction = prediction

            # prediction: 0 = EMPTY (free/green), 1 = NOT_EMPTY (occupied/red)
            if final_prediction == 0:
                color = (0, 255, 0)  # GREEN for empty
                free_count += 1
            else:
                color = (0, 0, 255)   # RED for occupied

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, f'Free slots: {free_count} | Total: {len(parking_spots)}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        
        occupied_count = len(parking_spots) - free_count
        occupancy_percent = (occupied_count / len(parking_spots) * 100) if len(parking_spots) > 0 else 0
        # Record occupancy data to CSV for this frame (use ISO timestamp with milliseconds)
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        csv_writer.writerow([free_count, occupied_count, len(parking_spots), f"{occupancy_percent:.1f}", frame_count, timestamp])
        csv_file.flush()  # Flush to disk so data is saved immediately
        try:
            os.fsync(csv_file.fileno())
        except Exception:
            pass
        
        cv2.putText(frame, f'Occupied: {occupied_count} | Occupancy: {occupancy_percent:.1f}%',
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Frame: {frame_count}',
                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 1)

        cv2.imshow('Smart Parking System', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break
except KeyboardInterrupt:
    pass
except Exception as e:
    print("Error during processing:", e)
finally:
    video.release()
    cv2.destroyAllWindows()
    print("Program finished")
    print("Model type:", type(model))
    # Close CSV file
    try:
        csv_file.close()
    except Exception:
        pass

