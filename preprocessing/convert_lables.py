import os
import json
import argparse
from pathlib import Path

# Label map aligned with BDD100K format
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
    "traffic light": 9,
}

ID_TO_LABEL = {v: k for k, v in BDD100K_LABEL_MAP.items()}


IMG_WIDTH = 1280
IMG_HEIGHT = 720

def parse_args():
    parser = argparse.ArgumentParser(description="Convert BDD100K annotations to YOLOv8 format.")
    parser.add_argument('-l', '--label-json', type=str, help='Path to a single label .json file')
    parser.add_argument('-n', '--label-dir', type=str, help='Path to directory of JSON files (multiple BDD-style label files)')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Output path to store YOLO format label files')
    return parser.parse_args()

def box2d_to_yolo(box2d):
    x1 = box2d["x1"] / IMG_WIDTH
    x2 = box2d["x2"] / IMG_WIDTH
    y1 = box2d["y1"] / IMG_HEIGHT
    y2 = box2d["y2"] / IMG_HEIGHT

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return cx, cy, width, height

def convert_bdd100k_to_yolo(label_json=None, label_dir=None, output_dir=None):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if label_json:
        frames = json.load(open(label_json, "r"))

        for frame in frames:
            image_name = Path(frame["name"]).stem
            yolo_file = Path(output_dir, f"{image_name}.txt")

            with open(yolo_file, "w") as f:
                for label in frame["labels"]:
                    if "box2d" not in label:
                        continue

                    box2d = label["box2d"]
                    if box2d["x1"] >= box2d["x2"] or box2d["y1"] >= box2d["y2"]:
                        continue

                    cx, cy, w, h = box2d_to_yolo(box2d)
                    class_id = BDD100K_LABEL_MAP[label["category"]]
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    elif label_dir:
        label_dir = Path(label_dir)

        for json_file in label_dir.iterdir():
            if json_file.suffix != '.json':
                continue
            with open(json_file, "r") as f:
                frame = json.load(f)

            img_name = frame["name"]
            yolo_file = Path(output_dir, f"{img_name}.txt")

            with open(yolo_file, "w+") as f:
                for label in frame["frames"][0]["objects"]:
                    if "box2d" not in label:
                        continue

                    box2d = label["box2d"]
                    if box2d["x1"] >= box2d["x2"] or box2d["y1"] >= box2d["y2"]:
                        continue

                    cx, cy, w, h = box2d_to_yolo(box2d)
                    class_id = BDD100K_LABEL_MAP[label["category"]]

                    color = label.get("attributes", {}).get("trafficLightColor", None)
                    if class_id == 9 and color not in ["red", "green", "yellow"]:
                        print(f"Traffic Light color '{color}' in file {img_name}")
                    elif class_id != 9 and color not in [None, "none"]:
                        print(f"Unexpected traffic light color '{color}' for {label['category']}")

                    f.write(f"{class_id} {cx} {cy} {w} {h}\n")

def main():
    args = parse_args()

    assert bool(args.label_json) ^ bool(args.label_dir), "Specify either --label-json or --label-dir, not both."

    convert_bdd100k_to_yolo(
        label_json=args.label_json,
        label_dir=args.label_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
