# Novel View Synthesis

3D Gaussian Splatting pipeline for novel view synthesis. The pipeline extracts frames from an input video, reconstructs camera poses and an initial sparse point cloud with COLMAP, trains a 3DGS model with `gsplat`, and renders novel RGB views from interpolated camera trajectories.

This workflow will be especially useful for those without on-device CUDA.

## Project Structure

```bash
Novel_View_Synthesis/
├── nvs.ipynb                         # Google Colab workflow for training and rendering
├── notes/                            # Notes on relevant NVS papers
└── results/
    └── no_mask/
        ├── cfg.yml                   # Training configuration
        ├── ckpts/                    # Saved 3DGS checkpoints
        ├── renders/                  # Validation images
        ├── stats/                    # Validation & Training metrics
        ├── tb/                       # TensorBoard logs
        ├── traj_frames/              # RGB-only trajectory frames for report figures
        └── videos/                   # RGB / depth trajectory videos
```

## Prerequisites

Frame extraction and Structure-from-Motion (SFM) pipeline was performed via local environment, and the training workflow was run in Google Colab with a CUDA GPU (T4).
The notebook (`nvs.ipynb`) installs Python 3.10, PyTorch CUDA 12.4, `gsplat==1.5.3`, and the `gsplat/examples` dependencies.

System tools:

```bash
conda install -c conda-forge ffmpeg colmap
```

Python environment used in `nvs.ipynb`:

```bash
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev

python3.10 -m venv /content/py310
source /content/py310/bin/activate

pip install --upgrade pip setuptools wheel ninja cmake
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install gsplat==1.5.3 -f https://docs.gsplat.studio/whl/pt24cu124
```

> **Note:** `pip install gsplat` will be sufficient if with on-device CUDA and appropriate python version.

Clone the `gsplat` source and install the example dependencies:

```bash
cd /content
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git checkout v1.5.3

cd examples
pip install -r requirements.txt --no-build-isolation
```

## 1. Frame Extraction

Start from the input video. Extract candidate frames with `ffmpeg`, then manually select frames with enough overlap for COLMAP. The goal is to keep a set of images that covers the target scene from multiple nearby viewpoints, rather than keeping every sequential video frames.

Example command:
```bash
mkdir -p data/frames_candidates
ffmpeg -i data/input_video.mp4 \
  -ss 00:00 -to 00:07 \
  -vf "fps=6" \
  data/frames_candidates/frame_%05d.jpg
```

The final selected images should be placed in:
```bash
data/
└── colmap_no_mask/
    └── images/
        ├── frame_00001.jpg
        ├── frame_00002.jpg
        └── ...
```

## 2. Structure-from-Motion (SFM) Pipeline

Run COLMAP to estimate camera intrinsics / extrinsics and sparse 3D points. The original workflow used the COLMAP GUI.

Recommended COLMAP settings:
- Feature extraction camera model: `SIMPLE_RADIAL`
- Enable shared intrinsics: `Shared for all images`
- Feature matching: Exhaustive matching
- Sparse reconstruction: Default settings

Expected output layout (under data directory):
```bash
colmap_no_mask/
├── images/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── database.db
```

## 3. Setup for Training

Install environment and `gsplat==1.5.3`, then clone the matching `gsplat` source to make `examples/simple_trainer.py` and its dependencies available. Take advantage of Google Drive to load necessary data into Google Colab.

Then, downsample the images to be used for training. The images should be stored in `images_4`. Note that downsampling will not be required if command changed to `--data_factor 1`, but this will make the training longer.

Refer to `nvs.ipynb` for the precise code used.

## 4. Train and Save Checkpoints

Training is run from `gsplat/examples` using `simple_trainer.py`.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python -u simple_trainer.py default \
  --data_dir /content/nvs/colmap_no_mask \
  --data_factor 4 \
  --result_dir ./results/no_mask \
  --disable_viewer True
```

Training saves checkpoints and validation outputs at the configured evaluation steps. The training process will eventually produce:

```bash
results/no_mask/
├── cfg.yml
├── ckpts/
│   ├── ckpt_6999_rank0.pt
│   └── ckpt_29999_rank0.pt
├── renders/
├── stats/
├── tb/
└── videos/
```

## 5. Save RGB Trajectory Frames

The default `gsplat` trajectory render writes a video. This repository's `simple_trainer.py` has been modified to also write RGB-only PNG frames during `render_traj()` into `results/no_mask/traj_frames`.

If more trajectory frames are needed, increase the interpolation density in `render_traj()`. Note that changing the interpolation density does not require retraining; it only changes the camera poses used during rendering.

## 6. Novel View Synthesis

Use the checkpoint with better metrics when performing NVS. The `--ckpt` option will render a trajectory, skipping training.

```bash
python -u simple_trainer.py default \
  --data_dir /content/nvs/colmap_no_mask \
  --data_factor 4 \
  --result_dir ./results/no_mask \
  --ckpt ./results/no_mask/ckpts/ckpt_6999_rank0.pt \
  --render_traj_path interp \
  --disable_viewer
```

## 7. Results

`results/no_mask/traj_frames/step_6999/`: RGB-only novel-view renders from the interpolated camera trajectory

`results/no_mask/renders/`: Validation side-by-side images
- Left: Ground-truth validation iamge
- Right: Rendered image from same validation camera pose

`results/no_mask/videos/`: Trajectory videos, with side-by-side frames
- Left: RGB novel-view render
- Right: Depth visualization
