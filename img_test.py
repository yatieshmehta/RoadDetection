import cv2
import time
import argparse
import torch
import numpy as np
from ultralytics import YOLO
from yolo.inference import detect
from RoadDetection.preprocessing.convert_lables import ID_TO_LABEL
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
from depth_estimation import *

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + OC-SORT Tracking for Images")
    
    parser.add_argument('--source', type=str, default="test.jpg",  # Provide image path
                        help="Path to image file (e.g. 'image.jpg')")
    parser.add_argument('--model', type=str, default="output/runs/detect/train5/weights/best.pt",
                        help="Path to YOLOv8 model checkpoint.")
    parser.add_argument('--thresh', type=float, default=0.25,
                        help="Detection threshold for OC-SORT tracker.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load YOLO model
    model = YOLO(args.model)
    tracker = OCSort(det_thresh=args.thresh)
    
    # Load the image
    frame = cv2.imread(args.source)


    if frame is None:
        print("Error: Could not read the image.")
        return

    # Load MiDaS depth estimation model
    midas, transform = load_midas()

    # Image dimensions
    img_height, img_width = frame.shape[:2]
    img_info = (img_width, img_height)
    img_size = (img_width, img_height)

    # Get depth map
    depth_map = get_depth_map(frame, midas, transform)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_colored = cv2.applyColorMap(np.uint8(depth_map_normalized), cv2.COLORMAP_JET)

    # Show the depth map (heatmap)
    cv2.imwrite("depth_map.jpg", depth_map_colored)
    cv2.imshow("Depth Heatmap", depth_map_colored)
    
    # Run object detection (YOLOv8)
    detections, classes = detect(frame, model)

    if len(detections) > 0:
        tracks = tracker.update(detections, img_info, img_size)
        for track, cls in zip(tracks, classes):
            x1, y1, x2, y2, track_id = map(int, track[:5])
            class_name = ID_TO_LABEL.get(cls, "Unknown")

            # Calculate depth based on bounding box
            depth = calculate_depth(x1, x2, y1, y2, depth_map)

            # Draw bounding boxes and depth info on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {class_name}', (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Depth: {depth}', (x1, y1 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the resulting image
    cv2.imshow("YOLOv8 + OC-SORT + MiDaS Depth", frame)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
