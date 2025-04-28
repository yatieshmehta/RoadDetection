import torch
import argparse
import os
from ultralytics import YOLO
from multiprocessing import freeze_support

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model.")

    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help="Path to the pre-trained YOLO model (default: yolov8n.pt).")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to the dataset YAML file (required).")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of training epochs (default: 5).")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size (default: 16).")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="Device to train on (default: cuda).")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate (default: 0.01).")
    parser.add_argument('--img-size', type=int, default=640,
                        help="Input image size (default: 640).")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose training output.")
    parser.add_argument('--save-dir', type=str, default='./runs/train',
                        help="Directory to save results (default: ./runs/train).")
    parser.add_argument('--pretrained', action='store_true',
                        help="Use pretrained weights.")

    return parser.parse_args()

def validate_args(args):
    if args.model != 'yolov8n.pt' and not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    if args.epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.lr <= 0:
        raise ValueError("Learning rate must be a positive float.")
    if args.img_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise NotADirectoryError(f"Save path is not a directory: {args.save_dir}")

def main():
    args = parse_args()
    validate_args(args)

    if torch.cuda.is_available() and args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for training.")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        lr=args.lr,
        imgsz=args.img_size,
        verbose=args.verbose,
        project=args.save_dir,
        pretrained=args.pretrained
    )

    print(f"Training complete. Results saved to: {args.save_dir}")

if __name__ == "__main__":
    freeze_support()
    main()
