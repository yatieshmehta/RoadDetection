# Real-Time Object Detection, Tracking, and Depth Estimation with YOLOv8 and MiDaS

This project combines **tracking, and **MiDaS** for depth estimation to createYOLOv8** for object detection, **OC-SORT** for real-time multi-object  a comprehensive pipeline for analyzing dynamic driving scenes. The system not only detects and tracks objects but also estimates their real-world velocities using depth information. The model is optimized for **real-time** performance, making it suitable for real-world applications like autonomous driving and surveillance.

## Key Features
- **Fast Object Detection**: Fine-tuned **YOLOv8n** model on the **BDD100K** dataset for efficient real-time object detection with high accuracy.
- **Multi-Object Tracking**: Real-time tracking of multiple objects using the **OC-SORT** algorithm, ensuring accurate associations between frames.
- **Depth Estimation**: Depth maps generated via **MiDaS** to estimate the 3D positions of objects.
- **Velocity Calculation**: Object velocity is computed in real-time by tracking pixel movements across frames, scaled using depth information to get real-world speed in meters per second.
- **Real-Time Processing**: The entire pipeline, from detection to velocity estimation, is optimized to run on video streams in real-time, enabling fast and responsive analysis.
- **Lightweight for Fast Inference**: Optimized for low-latency inference, suitable for edge devices or applications requiring quick responses.

## Dataset
The project utilizes the **BDD100K** dataset, which contains a diverse set of driving videos annotated with object classes, bounding boxes, and frame-specific labels.
- **Raw Data**: Download the raw data from [here](http://bdd-data.berkeley.edu/).
- **Preprocessed Data**: Stored in the `datasets/` folder for training, validation, and testing.
Once the raw data is downloaded run the `convert_lables.py` and `remove_images.py` to clean and process the data. Further information is available using the `--help` flag with both programs.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yatieshmehta/RoadDetection.git
   cd RoadDetection
   ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Download the BDD100K dataset**

## Usage
1. **Preprocess the data**:
    Preprocess the data outlined above in the [Dataset](#Dataset) section.
2. **Train YOLOv8 Model**:
    Train the YOLOv8 model on the preprocessed BDD100K dataset:
    ```bash
    python yolo/train.py --data datasets/data.yaml --epochs 100 --batch-size 16 --img-size 640
    ```
3. **Real-Time Inference**:
    Run real-time inference on a video file/webcam:
    ```bash
    python main.py --input path_to_video.mp4 # leave blank for default webcam
    ```

## Performance
- **Real-Time**: The system processes video frames at 80 FPS on a high-tier GPU (RTX 4060).
- **Optimized for Inference**: The YOLOv8n model provides a good balance between performance and speed, making it suitable for edge devices or real-time applications.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **YOLOV8**: [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md)
- **OCSORT**: [OCSORT GitHub](https://github.com/noahcao/OC_SORT)
- **MiDaS**: [MiDaS GitHub](https://github.com/isl-org/MiDaS)