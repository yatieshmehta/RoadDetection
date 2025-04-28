import cv2
import time
import argparse
import torch
import numpy as np
from ultralytics import YOLO
from yolo.inference import detect
from preprocessing.convert_bdd100k_lables import ID_TO_LABEL
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
from depth_estimation import *

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + OC-SORT Tracking")
    
    parser.add_argument('--source', default="oslo.mp4", # 0,
                        help="Path to video file or camera index (default webcam 0)")
    parser.add_argument('--model', type=str, default="output/runs/detect/train5/weights/best.pt",
                        help="Path to YOLOv8 model checkpoint.")
    parser.add_argument('--thresh', type=float, default=0.25,
                        help="Detection threshold for OC-SORT tracker.")
    # parser.add_argument('--show-fps', action='store_true',
    #                     help="Display FPS on output frame.")
    # parser.add_argument('--output-dir', type=str, default=None,
    #                     help="Directory to save output video (optional).")
    # parser.add_argument('--display', action='store_true',
    #                     help="Display live output window.")

    return parser.parse_args()

def main():
    args = parse_args()

    model = YOLO(args.model)
    tracker = OCSort(det_thresh=args.thresh)
    cap = cv2.VideoCapture(args.source)
    midas, transform = load_midas()

    if not cap.isOpened():
        print("Error: Could not open source.")
        return
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]
        img_info = (frame.shape[1], frame.shape[2])
        img_size = (img_width, img_height)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        depth_map = get_depth_map(frame, midas, transform)
        detections, classes = detect(frame, model)

        if len(detections) > 0:
            tracks = tracker.update(detections, img_info, img_size)
            for track, cls in zip(tracks, classes):
                x1, y1, x2, y2, track_id = map(int, track[:5])
                class_name = ID_TO_LABEL.get(cls, "Unknown")
                depth = calculate_depth(x1, x2, y1, y2, depth_map)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f'Class: {class_name}', (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f'Depth: {depth}', (x1, y1 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 - Custom Model Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()