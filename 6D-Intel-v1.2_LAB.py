import os
import sys
import time
import socket
import struct
from collections import deque
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

# Request 640x480 MJPG @ 30 FPS (optimal for Intel iGPU without frame backlog)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
TARGET_FPS = 30
PREVIEW_MAX_HZ = 10.0  # Throttle OpenCV preview redraws to keep inference FPS high


def get_compute_device():
    """
    Specifically searches for an Intel Integrated GPU across DirectML or XPU,
    even if the machine also has a dedicated NVIDIA GPU.
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            count = torch_directml.device_count()
            print(f"[INFO] DirectML detected {count} GPU adapter(s):")
            intel_idx = None

            for i in range(count):
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

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"[INFO] Using Intel Graphics via PyTorch XPU: {torch.xpu.get_device_name(0)}")
        return torch.device("xpu:0")

    num_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(num_threads)
    print(f"[INFO] Using CPU mode ({num_threads} threads).")
    return torch.device("cpu")


def open_webcam(cam_id: int, width: int, height: int, fps: int):
    """Opens webcam with MJPG low-latency settings and reports negotiated properties."""
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {cam_id}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Webcam negotiated: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS (requested {fps} FPS)")
    return cap


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
    y3 = int(size * R_mat[2, 1] + tdy)

    cv2.line(img, (tdx, tdy), (x1, y1), (0, 0, 255), 3)  # X - Red
    cv2.line(img, (tdx, tdy), (x2, y2), (0, 255, 0), 3)  # Y - Green
    cv2.line(img, (tdx, tdy), (x3, y3), (255, 0, 0), 3)  # Z (Nose) - Blue


def draw_crop_guide(img):
    """
    Draws the exact square region seen by Resize(256) -> CenterCrop(224)
    so the experimenter can center the participant's head before calibration.
    """
    h, w, _ = img.shape
    side = int(min(h, w) * (224.0 / 256.0))
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (0, 220, 0), 1)


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

    # Warmup pass before streaming starts
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224)).to(device)
        _ = model(dummy)
    print("[INFO] Model warmup complete.")

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
    cap = open_webcam(WEBCAM_ID, CAPTURE_WIDTH, CAPTURE_HEIGHT, TARGET_FPS)

    window_name = "6DRepNet360 Raw UDP Stream (LAB Intel Graphics)"
    print(f"[INFO] Streaming 48-byte UDP packets (x,y,z,yaw,pitch,roll) to {UDP_IP}:{UDP_PORT}")
    print("[INFO] Keys in preview window: 'p' = toggle preview rendering | 'q' = quit")

    frame_times = deque(maxlen=30)
    show_preview = True
    last_preview_t = 0.0
    preview_interval = 1.0 / PREVIEW_MAX_HZ

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            frame_times.append(now)
            live_fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) >= 2 and (frame_times[-1] - frame_times[0]) > 0
                else 0.0
            )

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

            # Throttled preview rendering (keeps inference/UDP rate maximized)
            if now - last_preview_t >= preview_interval:
                last_preview_t = now
                if show_preview:
                    h, w, _ = frame.shape
                    draw_crop_guide(frame)
                    draw_axes_from_R(frame, R_pred, w // 2, h // 2, size=100)
                    cv2.putText(
                        frame, f"Yaw:   {yaw:+6.1f} deg", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"Pitch: {pitch:+6.1f} deg", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"Roll:  {roll:+6.1f} deg", (20, 101),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2
                    )
                    cv2.putText(
                        frame, f"FPS:   {live_fps:5.1f} Hz", (20, 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                    )
                    cv2.imshow(window_name, frame)
                else:
                    blank = np.zeros((160, 420, 3), dtype=np.uint8)
                    cv2.putText(
                        blank, f"STREAMING (preview paused) | {live_fps:5.1f} Hz", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                    )
                    cv2.putText(
                        blank, "Press 'p' to resume preview | 'q' to quit", (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1
                    )
                    cv2.imshow(window_name, blank)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    show_preview = not show_preview
                    print(f"[INFO] Preview rendering {'ON' if show_preview else 'PAUSED'}")

    cap.release()
    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
