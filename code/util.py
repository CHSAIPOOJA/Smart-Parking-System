import cv2
import numpy as np


def remove_overlapping_boxes(boxes, overlap_threshold=0.3):
    """
    Remove overlapping bounding boxes by keeping the largest ones
    overlap_threshold: if IoU > threshold, consider them overlapping
    """
    if len(boxes) == 0:
        return boxes
    
    # Sort by area (largest first)
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    filtered = []
    
    for i, box1 in enumerate(boxes_sorted):
        x1, y1, w1, h1 = box1
        is_duplicate = False
        
        for box2 in filtered:
            x2, y2, w2, h2 = box2
            
            # Calculate intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            if xi2 > xi1 and yi2 > yi1:
                inter_area = (xi2 - xi1) * (yi2 - yi1)
                box1_area = w1 * h1
                box2_area = w2 * h2
                union_area = box1_area + box2_area - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > overlap_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append(box1)
    
    return filtered


def get_parking_spots_bboxes(mask):
    """
    Extract bounding boxes of parking spots from mask image
    """
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Morphology to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Use connected components to find individual parking spots
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)

    boxes = []
    for i in range(1, numLabels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = w * h

        # More precise thresholds for parking spots
        # Most parking spots are roughly 50-150 pixels in each dimension
        if 25 <= w <= 200 and 20 <= h <= 200 and 300 <= area <= 30000:
            boxes.append((x, y, w, h))

    # Remove overlapping boxes to avoid counting duplicates
    boxes = remove_overlapping_boxes(boxes, overlap_threshold=0.25)
    
    return boxes
