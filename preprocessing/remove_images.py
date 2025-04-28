import argparse
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Removes images with no corresponding label files.")
    parser.add_argument("-d", "--image_path", help="directory with images")
    parser.add_argument("-l", "--label_path", help="path to the label dir")
    parser.add_argument("--dry_run", action="store_true", help="Only log deletions, do not delete files")
    return parser.parse_args()

def remove_images_without_labels(image_path: Path, label_path: Path, dry_run=True):
    """
    Removes images from image_path if there is no corresponding label in label_path.
    Assumes label files are in YOLO format with .txt extensions.

    :params
        image_path: Directory containing images (.jpg)
        label_path: Directory containing label files (.txt)
        dry_run: If True, only logs the deletions without performing them
    """
    image_path = image_path.resolve()
    label_path = label_path.resolve()

    assert image_path.is_dir(), "Image directory must exist"
    assert label_path.is_dir(), "Label directory must exist"

    count_total = 0
    count_deleted = 0

    for image in image_path.rglob("*.jpg"):
        label = label_path / (image.stem + ".txt")
        count_total += 1
        if not label.is_file():
            print(f"Label not found for: {image.name}")
            if dry_run:
                print(f"[Dry Run] Would delete image: {image}")
            else:
                print(f"Deleting image: {image}")
                os.remove(image)
            count_deleted += 1

    print(f"Checked {count_total} images. {'Would remove' if dry_run else 'Removed'} {count_deleted} without labels.")


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    label_path = Path(args.label_path)

    remove_images_without_labels(image_path, label_path, dry_run=args.dry_run)
    

if __name__ == "__main__":
    main()
