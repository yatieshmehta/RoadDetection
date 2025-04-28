import torch
import cv2

def load_midas():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return midas, transform

def get_depth_map(frame, midas, transform):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map  = prediction.cpu().numpy()
    return depth_map
    
def calculate_depth(x1, x2, y1, y2, depth_map):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    depth = depth_map[center_y, center_x]
    return depth