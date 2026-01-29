import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'code'))
import cv2
import pickle
import numpy as np
from skimage.transform import resize
from util import get_parking_spots_bboxes

print("Loading model...")
with open('dataset/archive (1)/parking/model/model.p', 'rb') as f:
    model = pickle.load(f)

print("Loading video and mask...")
video = cv2.VideoCapture(r'dataset/archive (1)/parking/parking_1920_1080_loop.mp4')
mask = cv2.imread(r'dataset/archive (1)/parking/mask_1920_1080.png', 0)

print("Detecting parking spots...")
parking_spots = get_parking_spots_bboxes(mask)
print(f"Total spots detected: {len(parking_spots)}")

# Read first frame
ret, frame = video.read()
if ret:
    print(f"\nAnalyzing first frame...")
    empty_predictions = []
    occupied_predictions = []
    
    for idx, (x, y, w, h) in enumerate(parking_spots):
        crop = frame[y:y+h, x:x+w]
        mean_intensity = np.mean(crop)
        
        if mean_intensity < 20 or mean_intensity > 240:
            prediction = 1
            status = "DARK/BRIGHT (marked occupied)"
        else:
            crop_resized = resize(crop, (15, 15, 3))
            crop_flat = crop_resized.flatten().reshape(1, -1)
            prediction = model.predict(crop_flat)[0]
            status = f"EMPTY" if prediction == 0 else f"OCCUPIED"
        
        if prediction == 0:
            empty_predictions.append(idx)
        else:
            occupied_predictions.append(idx)
    
    print(f"\nFirst Frame Analysis:")
    print(f"  Empty (prediction=0): {len(empty_predictions)}")
    print(f"  Occupied (prediction=1): {len(occupied_predictions)}")
    print(f"  Total: {len(parking_spots)}")
    print(f"  Free percentage: {len(empty_predictions)/len(parking_spots)*100:.1f}%")
    
    # Draw debug visualization
    debug_frame = frame.copy()
    for idx, (x, y, w, h) in enumerate(parking_spots):
        if idx in empty_predictions:
            color = (0, 255, 0)  # GREEN
        else:
            color = (0, 0, 255)  # RED
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
    
    cv2.putText(debug_frame, f'Empty: {len(empty_predictions)} | Total: {len(parking_spots)}', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite('debug_frame.jpg', debug_frame)
    print(f"\nDebug frame saved as 'debug_frame.jpg'")

video.release()
print("Analysis complete!")
