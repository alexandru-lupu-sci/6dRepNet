# 6DRepNet360 Full Rotation

This is a PyTorch implementation of 6DRepNet360 specified in
["Towards Robust and Unconstrained Full Range of Rotation Head Pose Estimation"](https://ieeexplore.ieee.org/document/10477888)
IEEE Transactions on Image Processing 2024

Head position is tracked and coordinates are displayed on screen and via UDP at IP127.0.0.1:4242

Contains two versions "python 6D-Intel-v1.1.py" and "python 6D-Cuda-v1.1.py". One using Intel Integrated Graphics, and one using the CUDA cores of NVIDIA GPUs.

###### Monitors Yaw, Pitch and Roll coordinates:

* Yaw:

  * 0° center
  * Positive° rotation to the right (up to +179.9°, i.e. with back to the camera)
  * Negative° rotation to the left (up to -179.9°)
* Pitch:

  * 0° center
  * Positive° looking up
  * Negative° looking down
* Roll:

  * 0° center
  * Positive° tilt to the left
  * Negative° tilt to the right
  * Attention: Positive/negative and left/right is not the same as Yaw!!

# Environment setup

conda create -n ENV-NAME python=3.11 -y
conda activate ENV-NAME

# Modules to download

python==3.11

torch==2.4.1
torchvision==0.19.1
torch-directml
opencv-python
Pillow
numpy
scipy

matplotlib

##### Install code :

* pip install torch==2.4.1 torchvision==0.19.1 torch-directml opencv-python Pillow numpy scipy matplotlib pandas

# Open from terminal

cd C:\\Users\\Lab\\Desktop\\6dRepNet

python 6D-Intel-v1.1.py



(Sam's laptop)

cd C:\\Users\\MuellSam\\switchdrive\\VCNL\\Sam - Experiment\\Pilot\_v2\\6DRepNet\_headtracker

# All modules present in the new environment after install

###### Name                    Version      Build                      Channel

bzip2                      1.0.8            h0ad9c76\_10         conda-forge
ca-certificates       2026.7.22   h4c7d964\_0           conda-forge
contourpy             1.3.3            pypi\_0                     pypi
cycler                     0.12.1          pypi\_0                    pypi
filelock                   4.2.0            pypi\_0                    pypi
fonttools               4.67.0          pypi\_0                    pypi
fsspec                    2026.9.0     pypi\_0                    pypi
jinja2                      3.1.6           pypi\_0                    pypi
kiwisolver              1.5.1           pypi\_0                    pypi
libexpat                 2.8.5           hac47afa\_0           conda-forge
libffi                        3.7.0           h3d046cb\_1         conda-forge
liblzma                   5.8.3           hfd05255\_1          conda-forge
libpython               3.11.17      hfb4101e\_2\_cpython  conda-forge
libsqlite                  3.54.0        hf5d6505\_101       conda-forge
libzlib                      1.3.2          hfd05255\_3           conda-forge
markupsafe           3.0.4          pypi\_0                    pypi
matplotlib              3.11.2        pypi\_0                    pypi
mpmath                 1.3.0          pypi\_0                    pypi
networkx               3.6.1           pypi\_0                   pypi
numpy                   2.4.6           pypi\_0                   pypi
opencv-python     5.0.0.93     pypi\_0                   pypi
openssl                  3.6.5           hf411b9b\_0          conda-forge
packaging              26.3            pyhc364b38\_0     conda-forge
pandas                   3.0.6           pypi\_0                   pypi
pillow                     12.3.0         pypi\_0                   pypi
pip                          26.2.1         pyh8b19718\_0    conda-forge
pyparsing              3.3.3           pypi\_0                   pypi
python                   3.11.17       hb12b558\_2\_cpython  conda-forge
python-dateutil    2.9.0.post0    pypi\_0               pypi
scipy                      1.17.1           pypi\_0                 pypi
setuptools            84.0.0           pyh332efcf\_0     conda-forge
six                          1.17.0           pypi\_0                 pypi
sympy                   1.14.0           pypi\_0                 pypi
tk                           8.6.13           h967ab96\_4       conda-forge
torch                     2.4.1             pypi\_0                 pypi
torch-directml     0.2.5.dev240914  pypi\_0        pypi
torchvision           0.19.1           pypi\_0                 pypi
typing-extensions     4.16.0     pypi\_0                 pypi
tzdata                     2026.5        pypi\_0                 pypi
ucrt                       10.0.26100.0     h57928b3\_0         conda-forge
vc                          14.5               ha367084\_41            conda-forge
vc14\_runtime      14.51.36247      habf1de7\_41       conda-forge
vcomp14              14.51.36247      habf1de7\_41       conda-forge
wheel                    0.48.0                pyhd8ed1ab\_0     conda-forge

# Function of the .py files

##### 6D-Intel-v1.1.py

The script that opens the camera and sends head position feed via UDP at IP127.0.0.1:4242. Works with PCs that only have Intel integrated graphics

##### 6D-Cuda-v1.1.py

Same, but uses the CUDA cores for PCs who have an NVIDIA GPU.

Needs a change in modules loaded (recommend setting a new environment due to modules conflicts): pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

##### test.py

Contains the neural network architecture. 6D-Intel/Cuda-v1.1 imports from here to build the model

##### utils.py

Math functions (e.g. converting raw outputs into a rotation matrix)

##### datasets.py

Can be used for training the model or loading a dataset. Even if not used, this file needs to be present in the folder.

##### 6DRepNet360\_Full-Rotation\_300W\_LP+Panoptic.pth

The actual deep learning model trained on 360° images of faces

# Cite headtracker article

@ARTICLE{10477888,
author={Hempel, Thorsten and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
journal={IEEE Transactions on Image Processing},
title={Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation},
year={2024},
volume={33},
number={},
pages={2377-2387},
doi={10.1109/TIP.2024.3378180}}

Article link : https://ieeexplore.ieee.org/document/10477888

