# 3D Human Motion Rendering

Python tools for visualizing 3D human motion from 4D-Humans predictions using the SMPL body model.

## Features

- Convert predicted SMPL parameters into 3D human meshes.
- Render human motion as images and videos.
- Apply temporal smoothing to reduce motion jitter.
- Export per-frame meshes in OBJ format.
- Experiment with camera placement and coordinate transformations.

## Main Files

- `rendering.py` — mesh rendering with camera-translation smoothing.
- `render_4smpl_min_from_pkl.py` — render tracked people from prediction files.
- `smooth_pkl.py` — smooth prediction data.
- `export_objs_any_pkl.py` — export meshes from 4D-Humans outputs.
- `ren.py` / `rendnew.py` — alternative rendering implementations.

## Usage

Prepare your 4D-Humans prediction files and SMPL model files, then update the input paths, output paths, and camera settings in the scripts.

This repository contains experimental rendering utilities. Human pose estimation is performed separately using 4D-Humans.

**Technologies:** Python, PyTorch, SMPL, PyRender, Trimesh, OpenCV, NumPy.
