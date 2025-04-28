import os
import argparse
from pathlib import Path
import cv2

# Final BDD100K-style label map
BDD100K_LABEL_MAP = {
    "car": 0,
    "bus": 1,
    "person": 2,
    "bike": 3,
    "truck": 4,
    "motor": 5,
    "train": 6,
    "rider": 7,
    "traffic sign": 8,
    "traffic light": 9
}

# KITTI → BDD100K mapping
KITTI_TO_BDD_MAP = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Pedestrian": "person",
    "Person_sitting": "person",
    "Cyclist": "rider",
    "Tram": "train",
    "Misc": None,
    "DontCare": None
}

def parse_args():
    parser = argparse.ArgumentParser(description="Convert KITTI annotations to YOLOv8 format with BDD100K-style classes.")
    parser.add_argument('-l', '--label-dir', type=str, required=True, help='Path to directory with KITTI label .txt files')
    parser.add_argument('-i', '--image-dir', type=str, required=True, help='Path to directory with corresponding images')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Output path to store YOLO format label files')
    return parser.parse_args()

def convert_kitti_to_yolo(label_input_dir, image_dir, output_dir):
    label_input_dir = Path(label_input_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file in label_input_dir.iterdir():
        if file.suffix != '.txt':
            continue

        image_path = Path(image_dir, f'{file.stem}.png')
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        image_height, image_width = image.shape[:2]
        yolo_path = output_dir / file.name

        with open(file, 'r') as fin, open(yolo_path, 'w') as fout:
            for line in fin.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                kitti_class = parts[0]
                bdd_class = KITTI_TO_BDD_MAP.get(kitti_class)

                if bdd_class is None:
                    continue

                class_id = BDD100K_LABEL_MAP[bdd_class]

                x1, y1, x2, y2 = map(float, parts[4:8])
                x_center = (x1 + x2) / 2.0 / image_width
                y_center = (y1 + y2) / 2.0 / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height

                fout.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    args = parse_args()
    convert_kitti_to_yolo(args.label_dir, args.image_dir, args.output_dir)
