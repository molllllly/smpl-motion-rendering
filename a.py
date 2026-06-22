import os
import joblib
import numpy as np
import torch
from smplx import SMPL
import trimesh
from scipy.spatial.transform import Rotation as R

# ------------------------------
# Configuration
# ------------------------------
views = ['bilkamera', 'portcamera', 'survcam']  # Correct the list of views
pkl_paths = {
    'bilkamera': 'demo_bilkamera.pkl',
    'portcamera': 'demo_portcamera_compressed3.pkl',
    'survcam': 'demo_surv_cut.pkl'
}
output_root = 'output_synchronized'
smpl_model_path = 'data/smpl/'  # Path to SMPL neutral model


# ------------------------------
# Helper functions
# ------------------------------

def load_pkl(pkl_file):
    """Load .pkl file using joblib."""
    data = joblib.load(pkl_file)
    return data


def organize_by_tid(data):
    """Organize the data by track ID (tid). Now using image paths as keys."""
    tracks = {}


    for img_path in data.keys():  # Iterate over image paths as keys
        # Extract information based on image path



        tids = data[img_path].get('tid')  # Get 'tid' for the current image

        # Check if 'tid' is a list (i.e., multiple persons in one frame)
        if isinstance(tids, list):
            for tid in tids:
                smpl = data[img_path].get('smpl')
                camera = data[img_path].get('camera')

                if tid not in tracks:
                    tracks[tid] = {'smpl': [], 'camera': [], 'img_paths': []}

                tracks[tid]['smpl'].append(smpl)
                tracks[tid]['camera'].append(camera)
                tracks[tid]['img_paths'].append(img_path)  # Store the image path for reference
        else:
            # If 'tid' is not a list (i.e., only one person in the frame)
            smpl = data[img_path].get('smpl')
            camera = data[img_path].get('camera')

            if tids not in tracks:
                tracks[tids] = {'smpl': [], 'camera': [], 'img_paths': []}

            tracks[tids]['smpl'].append(smpl)
            tracks[tids]['camera'].append(camera)
            tracks[tids]['img_paths'].append(img_path)  # Store the image path for reference

    return tracks


import numpy as np


def linear_interpolate(seq):
    """Linear interpolation for missing frames in the sequence."""

    # Ensure the sequence consists of numeric arrays
    try:
        if isinstance(seq, list) and isinstance(seq[0], dict):
            # Extract relevant numeric data from the dictionary
            betas = [s['betas'] if 'betas' in s else None for s in seq]
            body_pose = [s['body_pose'] if 'body_pose' in s else None for s in seq]
            global_orient = [s['global_orient'] if 'global_orient' in s else None for s in seq]

            # Convert each to numpy arrays
            betas = np.array(betas, dtype=np.float32) if None not in betas else betas
            body_pose = np.array(body_pose, dtype=np.float32) if None not in body_pose else body_pose
            global_orient = np.array(global_orient, dtype=np.float32) if None not in global_orient else global_orient

            # Now interpolate each of these components
            betas = interpolate_component(betas)
            body_pose = interpolate_component(body_pose)
            global_orient = interpolate_component(global_orient)

            # Rebuild the dictionaries for each frame
            return [{'betas': betas[i], 'body_pose': body_pose[i], 'global_orient': global_orient[i]} for i in
                    range(len(seq))]

        else:
            # Handle cases where `seq` is already a numeric array or list
            seq = np.array(seq, dtype=np.float32)  # Force float type for interpolation
            return interpolate_component(seq)
    except Exception as e:
        print(f"Error during conversion: {e}")
        return seq


def interpolate_component(seq):
    """Helper function for interpolating missing values in a sequence."""
    isnan = np.isnan(seq)
    notnan_idx = np.where(~isnan)[0]

    for i in range(len(seq)):
        if isnan[i]:
            prev_idx = max([idx for idx in notnan_idx if idx < i], default=None)
            next_idx = min([idx for idx in notnan_idx if idx > i], default=None)
            if prev_idx is not None and next_idx is not None:
                seq[i] = seq[prev_idx] + (seq[next_idx] - seq[prev_idx]) * ((i - prev_idx) / (next_idx - prev_idx))
            elif prev_idx is not None:
                seq[i] = seq[prev_idx]
            elif next_idx is not None:
                seq[i] = seq[next_idx]

    return seq


# ------------------------------
# Main pipeline
# ------------------------------

# 1. Load all views and organize by tid
all_tracks = {}
for view in views:
    data = load_pkl(pkl_paths[view])  # Use joblib to load pkl
    tracks = organize_by_tid(data)
    all_tracks[view] = tracks

# 2. Interpolate and smooth per view
for view in views:
    for tid in all_tracks[view]:
        all_tracks[view][tid]['smpl'] = linear_interpolate(all_tracks[view][tid]['smpl'])

# 3. Load SMPL model (ensure it is on the correct device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smpl_model = SMPL(model_path="data/smpl/SMPL_NEUTRAL.pkl", gender='NEUTRAL').to(device)

# 4. Export per-frame OBJ meshes
for view in views:
    for tid in all_tracks[view]:
        out_dir = os.path.join(output_root, view, f'tid_{tid}')
        os.makedirs(out_dir, exist_ok=True)
        for frame_idx, smpl_data in enumerate(all_tracks[view][tid]['smpl']):
            # Assuming smpl_data is in a structure that can be passed directly to SMPL
            betas = smpl_data['betas']
            body_pose = smpl_data['body_pose']
            global_orient = smpl_data['global_orient']

            # Convert data to tensors and move to the correct device
            betas_tensor = torch.tensor(betas).unsqueeze(0).to(device)
            body_pose_tensor = torch.tensor(body_pose).unsqueeze(0).to(device)
            global_orient_tensor = torch.tensor(global_orient).unsqueeze(0).to(device)

            # Generate SMPL model output
            output = smpl_model(
                betas=betas_tensor,
                body_pose=body_pose_tensor,
                global_orient=global_orient_tensor
            )
            verts = output.vertices[0].detach().cpu().numpy()  # Move vertices to CPU for export
            faces = smpl_model.faces

            # Create mesh and export to .obj file
            img_path = all_tracks[view][tid]['img_paths'][frame_idx]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(os.path.join(out_dir, f'{os.path.basename(img_path)}.obj'))

print("Synchronization and OBJ export complete.")
