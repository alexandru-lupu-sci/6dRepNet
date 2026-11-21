import os
import sys
import socket
import struct
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

# 1. Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "6DRepNet360-master")
CODE_DIR = os.path.join(REPO_DIR, "sixdrepnet360")
WEIGHTS_PATH = os.path.join(
    REPO_DIR, "6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
)

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import utils
from test import SixDRepNet360


# 2. Override utils.normalize_vector so it does not hardcode 'cuda:0' on Intel iGPUs
def normalize_vector_device_agnostic(v):
    v_mag = torch.sqrt(v.pow(2).sum(1, keepdim=True))
    v_mag = torch.clamp(v_mag, min=1e-8)
    return v / v_mag


utils.normalize_vector = normalize_vector_device_agnostic

# 3. UDP & Webcam Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 4242
WEBCAM_ID = 0


def get_compute_device():
    """
    Specifically searches for an Intel Integrated GPU across DirectML or XPU,
    even if the laptop also has a dedicated Nvidia GPU.
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            count = torch_directml.device_count()
            print(f"[INFO] DirectML detected {count} GPU adapter(s):")
            intel_idx = None

            for i in range(count):
                # Strip trailing null characters from the DirectX device name
                name = torch_directml.device_name(i).replace("\x00", "").strip()
                print(f"       Adapter {i}: {name}")
                if "intel" in name.lower() and intel_idx is None:
                    intel_idx = i

            if intel_idx is not None:
                chosen_name = torch_directml.device_name(intel_idx).replace("\x00", "").strip()
                print(f"[INFO] -> Selected Intel Integrated Graphics (Adapter {intel_idx}: {chosen_name})")
                return torch_directml.device(intel_idx)
            else:
                chosen_name = torch_directml.device_name(0).replace("\x00", "").strip()
                print(f"[WARNING] No Intel GPU found in DirectML list. Using Adapter 0: {chosen_name}")
                return torch_directml.device(0)
    except ImportError:
        pass

    # Check for native Intel XPU support in PyTorch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"[INFO] Using Intel Graphics via PyTorch XPU: {torch.xpu.get_device_name(0)}")
        return torch.device("xpu:0")

    # Fallback to CPU
    num_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(num_threads)
    print(f"[INFO] Using CPU mode ({num_threads} threads).")
    return torch.device("cpu")


def extract_ypr_no_gimbal_lock(R_mat: np.ndarray):
    """
    Extracts raw (yaw, pitch, roll) in degrees from the 3x3 rotation matrix R
    using the Y-X-Z (Azimuth-Elevation) sequence so pitch never flips at 90 deg yaw.
    """
    fx, fy, fz = R_mat[2, 0], R_mat[2, 1], R_mat[2, 2]
    horiz_mag = np.sqrt(fx * fx + fz * fz)

    yaw = np.degrees(np.arctan2(-fx, fz))
    pitch = np.degrees(np.arctan2(fy, horiz_mag))
    roll = np.degrees(np.arctan2(-R_mat[0, 1], R_mat[1, 1]))

    return float(yaw), float(pitch), float(roll)


def draw_axes_from_R(img, R_mat: np.ndarray, tdx: int, tdy: int, size: int = 100):
    """Draws the 3D orientation axes on the webcam preview."""
    x1 = int(size * R_mat[0, 0] + tdx)
    y1 = int(-size * R_mat[0, 1] + tdy)
    x2 = int(-size * R_mat[1, 0] + tdx)
    y2 = int(size * R_mat[1, 1] + tdy)
    x3 = int(size * R_mat[2, 0] + tdx)
    y3 = int(-size * R_mat[2, 1] + tdy)

    cv2.line(img, (tdx, tdy), (x1, y1), (0, 0, 255), 3)  # X - Red
    cv2.line(img, (tdx, tdy), (x2, y2), (0, 255, 0), 3)  # Y - Green
    cv2.line(img, (tdx, tdy), (x3, y3), (255, 0, 0), 3)  # Z (Nose) - Blue


def main():
    device = get_compute_device()

    # Load model architecture from test.py
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found at: {WEIGHTS_PATH}")

    try:
        saved_state_dict = torch.load(
            WEIGHTS_PATH, map_location="cpu", weights_only=False
        )
    except TypeError:
        saved_state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")

    if "model_state_dict" in saved_state_dict:
        model.load_state_dict(saved_state_dict["model_state_dict"])
    else:
        model.load_state_dict(saved_state_dict)

    model.to(device)
    model.eval()

    # Exact transform pipeline from test.py (operates directly on the full frame)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {WEBCAM_ID}")

    print(f"[INFO] Streaming 48-byte OpenTrack UDP packets to {UDP_IP}:{UDP_PORT}")
    print("[INFO] Press 'q' in the video window to stop.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            # Convert full BGR webcam frame to RGB and apply test.py transforms
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Predict 3x3 rotation matrix and extract raw angles
            R_pred = model(img_tensor)[0].cpu().numpy()
            yaw, pitch, roll = extract_ypr_no_gimbal_lock(R_pred)

            # Pack 48 bytes (6 doubles: x, y, z, yaw, pitch, roll) for PsychoPy
            packet = struct.pack("6d", 0.0, 0.0, 0.0, yaw, pitch, roll)
            sock.sendto(packet, (UDP_IP, UDP_PORT))

            # Display preview window
            draw_axes_from_R(frame, R_pred, w // 2, h // 2, size=100)
            cv2.putText(
                frame, f"Yaw:   {yaw:+6.1f} deg", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"Pitch: {pitch:+6.1f} deg", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Roll:  {roll:+6.1f} deg", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )

            cv2.imshow("6DRepNet360 Raw UDP Stream (Intel Graphics)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()