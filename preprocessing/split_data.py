import argparse
import random
import shutil
from pathlib import Path


def move_kitti_split(in_dir: Path, out_dir: Path, split: list[float]):
    assert sum(split) == 1.0, "Split ratios must sum to 1.0"

    # Prepare paths
    image_dir = Path(in_dir, "image_2")
    label_dir = Path(in_dir, "label_2")
    velo_dir = Path(in_dir, "velodyne")
    calib_dir = Path(in_dir, "calib")

    assert image_dir.exists(), "image_2 directory not found"
    assert label_dir.exists(), "label_2 directory not found"
    assert velo_dir.exists(), "velodyne directory not found"
    assert calib_dir.exists(), "calib directory not found"

    image_files = sorted(image_dir.glob("*.png"))
    file_stems = [f.stem for f in image_files]
    random.shuffle(file_stems)

    train_end = int(split[0] * len(file_stems))
    val_end = train_end + int(split[1] * len(file_stems))

    splits = {
        "train": file_stems[:train_end],
        "val": file_stems[train_end:val_end],
        "test": file_stems[val_end:],
    }

    for split_name, files in splits.items():
        print(f"Moving {len(files)} files to {split_name}...")
        for f in files:
            for subfolder in ["images", "label_2", "velodyne", "calib"]:
                src_base = in_dir / subfolder if subfolder != "images" else image_dir
                dst_base = out_dir / split_name / 'KITTI' / subfolder
                dst_base.mkdir(parents=True, exist_ok=True)
                src_file = src_base / f"{f}.png" if subfolder == "images" else src_base / f"{f}.txt" if subfolder == "label_2" else src_base / f"{f}.bin" if subfolder == "velodyne" else src_base / f"{f}.txt"

                if src_file.exists():
                    shutil.move(str(src_file), str(dst_base / src_file.name))


def parse_args():
    parser = argparse.ArgumentParser(description="Split KITTI dataset into train/val/test")
    parser.add_argument("-i", "--in_dir", type=str, required=True, help="Path to KITTI input directory")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("-s", "--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/Val/Test split ratios")
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir).absolute()
    out_dir = Path(args.out_dir).absolute()
    split = args.split

    move_kitti_split(in_dir, out_dir, split)


if __name__ == "__main__":
    main()
