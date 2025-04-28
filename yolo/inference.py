from ultralytics import YOLO
import numpy as np


def detect(frame, model):
    results = model(frame)

    detections = []
    classes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, clses):
            if score > 0.4:
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, score])
                classes.append(int(cls))
    return np.array(detections), classes