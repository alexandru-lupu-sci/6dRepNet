# 6DRepNet360 Full Rotation

This is a PyTorch implementation of 6DRepNet360 specified in
["Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation"](https://ieeexplore.ieee.org/document/10477888)
(IEEE Transactions on Image Processing, 2024).

Head orientation is tracked in full 360° rotation, displayed on screen, and streamed as 48-byte UDP packets (`6d`: `x, y, z, yaw, pitch, roll`, with `x = y = z = 0.0`) to `127.0.0.1:4242` (e.g., for PsychoPy).

Only the two `_LAB` scripts are used to launch the head tracker:

- `6D-Intel-v1.2_LAB.py` — for Intel Integrated Graphics (DirectML / XPU, with CPU fallback; 640×480 MJPG @ 30 FPS, 10 Hz preview throttle).
- `6D-Cuda-v1.2_LAB.py` — for NVIDIA GPUs via CUDA (with CPU fallback; 640×480 MJPG @ 60 FPS target, 15 Hz preview throttle).

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

## Two Separate Environments

There are **two separate virtual environments** in this project. Depending on which GPU you want to use, a specific build of PyTorch is required, and the Intel DirectML and NVIDIA CUDA builds cannot coexist in the same environment:

- **`.venv`** (default environment, managed by `pyproject.toml` and `uv.lock`) — uses `torch 2.4.1` + `torch-directml` for **Intel Integrated Graphics**.
- **`.venv-cuda`** — uses CUDA-enabled PyTorch (`torch 2.14.1+cu130` + `torchvision 0.29.1+cu130`) for **NVIDIA GPUs**.

Running a script without explicitly selecting `.venv-cuda` defaults to `.venv` (Intel graphics), so you must launch the tracker using the environment-specific terminal commands listed in `launch-codes.txt`.

## Installation (with uv)

### Prerequisites

1. **Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)** (tested with `uv 0.13.8`; Python `3.12` is pinned in `.python-version` and downloaded automatically by `uv`, while `pyproject.toml` requires `>=3.11`):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Model weights**: Place `6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` inside `6DRepNet360-master/` (`*.pth` files are excluded from Git in `.gitignore` due to size).

### 1. Intel Integrated Graphics Environment (`.venv`)

From the repository root, run:

```powershell
uv sync
```

This creates `.venv` with Python `3.12` and installs all locked dependencies from `pyproject.toml` and `uv.lock`.

### 2. NVIDIA GPU / CUDA Environment (`.venv-cuda`)

From the repository root, run:

```powershell
uv venv .venv-cuda --python 3.12
uv pip install --python .venv-cuda\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-cuda\Scripts\python.exe opencv-python pillow numpy scipy matplotlib pandas
```

> [!NOTE]
> If a machine has an older NVIDIA driver requiring an earlier CUDA wheel index (e.g., `cu124` or `cu121`), adjust `--index-url` accordingly.

## Launching the Head Tracker

Only the **`_LAB`** scripts are used to launch the head tracker, using the terminal commands stored in `launch-codes.txt`.

Open a terminal in the project folder:

```powershell
cd <path-to-6dRepNet>
```

Then run the command for your target GPU:

- **Intel Integrated Graphics (`.venv`)**:

  ```powershell
  .venv\Scripts\python.exe 6D-Intel-v1.2_LAB.py
  ```

- **NVIDIA GPU / CUDA (`.venv-cuda`)**:

  ```powershell
  .venv-cuda\Scripts\python.exe 6D-Cuda-v1.2_LAB.py
  ```

### Preview Window Controls (`_LAB` Scripts)

- A green square marks the exact `Resize(256)` → `CenterCrop(224)` region seen by the model so you can center the participant's head before calibration.
- Press **`p`** to pause or resume preview rendering (UDP streaming continues uninterrupted).
- Press **`q`** to quit.

## Packages and Versions Used (Info)

- **Python**: `3.12.16` (both `.venv` and `.venv-cuda`)
- **Package manager**: `uv 0.13.8`

| Package | `.venv` (Intel / DirectML) | `.venv-cuda` (NVIDIA CUDA) |
| :--- | :--- | :--- |
| `6drepnet` | `0.1.0` (editable) | `—` |
| `contourpy` | `1.4.0` | `1.4.0` |
| `cycler` | `0.12.1` | `0.12.1` |
| `filelock` | `4.2.10` | `4.2.4` |
| `fonttools` | `4.68.0` | `4.68.0` |
| `fsspec` | `2026.9.0` | `2026.9.0` |
| `jinja2` | `3.1.6` | `3.1.6` |
| `kiwisolver` | `1.5.1` | `1.5.1` |
| `markupsafe` | `3.0.4` | `3.0.3` |
| `matplotlib` | `3.11.2` | `3.11.2` |
| `mpmath` | `1.3.0` | `1.3.0` |
| `networkx` | `3.7` | `3.7` |
| `numpy` | `2.5.4` | `2.5.4` |
| `opencv-python` | `5.0.0.93` | `5.1.0.95` |
| `packaging` | `26.3` | `26.3` |
| `pandas` | `3.1.1` | `3.1.1` |
| `pillow` | `13.0.0` | `13.0.0` |
| `pyparsing` | `3.3.3` | `3.3.3` |
| `python-dateutil` | `4.0.0` | `4.0.0` |
| `scipy` | `1.18.1` | `1.18.1` |
| `setuptools` | `—` | `78.1.0` |
| `sympy` | `1.14.0` | `1.14.0` |
| `torch` | `2.4.1` | `2.14.1+cu130` |
| `torch-directml` | `0.2.5.dev240914` | `—` |
| `torchvision` | `0.19.1` | `0.29.1+cu130` |
| `typing-extensions` | `4.16.0` | `4.16.0` |
| `tzdata` | `2026.6` | `2026.6` |

## Project Files

- **`6D-Intel-v1.2_LAB.py`** — Active Intel integrated graphics launcher (uses `.venv`; opens webcam, predicts head pose, and streams UDP packets to `127.0.0.1:4242`).
- **`6D-Cuda-v1.2_LAB.py`** — Active NVIDIA CUDA launcher (uses `.venv-cuda`; same UDP stream using CUDA acceleration).
- **`6D-Intel-v1.1.py` & `6D-Cuda-v1.1.py`** — Earlier versions kept for reference (not used for lab launches).
- **`launch-codes.txt`** — Terminal launch commands and environment notes.
- **`pyproject.toml`, `uv.lock`, `.python-version`** — `uv` project configuration, locked dependencies for `.venv`, and Python `3.12` version pin.
- **`6DRepNet360-master/sixdrepnet360/test.py`** — Contains the `SixDRepNet360` neural network architecture imported by the launcher scripts.
- **`6DRepNet360-master/sixdrepnet360/utils.py`** — Rotation math utilities.
- **`6DRepNet360-master/sixdrepnet360/datasets.py`** — Upstream dataset/training module; must remain in the folder.
- **`6DRepNet360-master/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth`** — Pretrained 360° head pose deep learning model weights (git-ignored; copy manually).

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
