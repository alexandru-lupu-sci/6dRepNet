# 6DRepNet360 Full Rotation

This is a PyTorch implementation of 6DRepNet360 specified in
["Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation"](https://ieeexplore.ieee.org/document/10477888)
(IEEE Transactions on Image Processing, 2024).

Head orientation is tracked in full 360° rotation, displayed on screen, and streamed as 48-byte UDP packets (`6d`: `x, y, z, yaw, pitch, roll`, with `x = y = z = 0.0`) to `127.0.0.1:4242` (e.g., for PsychoPy).

The head tracker uses a single virtual environment (`.venv`) configured with CUDA-enabled PyTorch:

- **`6D-Cuda-v1.2_LAB.py`** — for NVIDIA GPUs via CUDA (with automatic CPU fallback; 640×480 MJPG @ 60 FPS target, 15 Hz preview throttle).

### Yaw, Pitch, and Roll Coordinates

- **Yaw**:
  - `0°` center
  - Positive° rotation to the right (up to `+179.9°`, i.e. with back to the camera)
  - Negative° rotation to the left (up to `-179.9°`)
- **Pitch**:
  - `0°` center
  - Positive° looking up
  - Negative° looking down
- **Roll**:
  - `0°` center
  - Positive° tilt to the left
  - Negative° tilt to the right
  - **Attention**: Positive/negative left/right is opposite to Yaw!

## Installation (with uv)

### Prerequisites

1. **Install [Git](https://git-scm.com/downloads)** (to clone the repository).
2. **Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)** (tested with `uv 0.13.8`; Python `3.12` is pinned in `.python-version` and downloaded automatically by `uv`, while `pyproject.toml` requires `>=3.11`):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Restart your terminal after installing `uv` so it is available on `PATH`.
3. **NVIDIA GPU driver** compatible with CUDA 13.0 (if no NVIDIA GPU is available, the script automatically falls back to CPU).

### 1. Clone the Repository

Clone the repository from GitHub ([alexandru-lupu-sci/6dRepNet](https://github.com/alexandru-lupu-sci/6dRepNet)) and move into the project folder:

```powershell
git clone https://github.com/alexandru-lupu-sci/6dRepNet.git
cd 6dRepNet
```

> [!NOTE]
> The pretrained model weights (`6DRepNet360-master/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth`, ~90 MB) are already included in the repository—no manual download is required. (`.gitignore` only ignores any additional untracked `*.pth` files.)

### 2. Set Up the Virtual Environment (`.venv`)

From the repository root (`6dRepNet`), run:

```powershell
uv sync
```

This creates `.venv` with Python `3.12` and installs CUDA-enabled PyTorch (`cu130`) along with all locked dependencies from `pyproject.toml` and `uv.lock`.

> [!NOTE]
> If a machine has an older NVIDIA driver requiring an earlier CUDA wheel index (e.g., `cu124` or `cu121`), update the `[[tool.uv.index]]` URL in `pyproject.toml` accordingly and re-run `uv sync`.

## Launching the Head Tracker

Open a terminal in the project folder and launch `6D-Cuda-v1.2_LAB.py` directly with `.venv` (this command is also saved in `launch-codes.txt`):

```powershell
cd <path-to-6dRepNet>
.venv\Scripts\python.exe 6D-Cuda-v1.2_LAB.py
```

Alternatively, you can launch it with `uv`:

```powershell
uv run 6D-Cuda-v1.2_LAB.py
```

### Preview Window Controls

- A green square marks the exact `Resize(256)` → `CenterCrop(224)` region seen by the model so you can center the participant's head before calibration.
- Press **`p`** to pause or resume preview rendering (UDP streaming continues uninterrupted).
- Press **`q`** to quit.

## Packages and Versions Used (Info)

- **Python**: `3.12.16` (`.venv`)
- **Package manager**: `uv 0.13.8`

| Package | Version (`.venv`) |
| :--- | :--- |
| `6drepnet` | `0.1.0` (editable) |
| `contourpy` | `1.4.0` |
| `cycler` | `0.12.1` |
| `filelock` | `4.2.10` |
| `fonttools` | `4.68.0` |
| `fsspec` | `2026.9.0` |
| `jinja2` | `3.1.6` |
| `kiwisolver` | `1.5.1` |
| `markupsafe` | `3.0.4` |
| `matplotlib` | `3.11.2` |
| `mpmath` | `1.3.0` |
| `networkx` | `3.7` |
| `numpy` | `2.5.4` |
| `opencv-python` | `5.0.0.93` |
| `packaging` | `26.3` |
| `pandas` | `3.1.1` |
| `pillow` | `13.0.0` |
| `pyparsing` | `3.3.3` |
| `python-dateutil` | `4.0.0` |
| `scipy` | `1.18.1` |
| `setuptools` | `84.2.1` |
| `sympy` | `1.14.0` |
| `torch` | `2.14.1+cu130` |
| `torchvision` | `0.29.1+cu130` |
| `typing-extensions` | `4.16.0` |
| `tzdata` | `2026.6` |

## Project Files

- **`6D-Cuda-v1.2_LAB.py`** — Active NVIDIA CUDA head tracker launcher (uses `.venv`; opens webcam, predicts head pose, and streams UDP packets to `127.0.0.1:4242`, with automatic CPU fallback).
- **`6D-Cuda-v1.1.py`, `6D-Intel-v1.1.py`, `6D-Intel-v1.2_LAB.py`** — Earlier/legacy scripts kept in the repository for reference (not used; Intel DirectML is no longer supported in `.venv`).
- **`launch-codes.txt`** — Terminal launch command and single-environment notes for `.venv`.
- **`pyproject.toml`, `uv.lock`, `.python-version`** — `uv` project configuration (CUDA `cu130` PyTorch index), locked dependencies for `.venv`, and Python `3.12` version pin.
- **`6DRepNet360-master/sixdrepnet360/test.py`** — Contains the `SixDRepNet360` neural network architecture imported by the launcher script.
- **`6DRepNet360-master/sixdrepnet360/utils.py`** — Rotation math utilities.
- **`6DRepNet360-master/sixdrepnet360/datasets.py`** — Upstream dataset/training module; must remain in the folder.
- **`6DRepNet360-master/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth`** — Pretrained 360° head pose deep learning model weights (included in the repository; additional untracked `*.pth` files are ignored by `.gitignore`).

## Citation

```bibtex
@ARTICLE{10477888,
author={Hempel, Thorsten and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
journal={IEEE Transactions on Image Processing},
title={Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation},
year={2024},
volume={33},
number={},
pages={2377-2387},
doi={10.1109/TIP.2024.3378180}}
```

Article link: https://ieeexplore.ieee.org/document/10477888
