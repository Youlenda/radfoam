import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import pycolmap
from pathlib import Path
import json


def get_omni_ray_dirs(img_wh):
    width, height = img_wh
    x = np.arange(width, dtype=np.float32) + 0.5
    y = np.arange(height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)

    u = x / width
    v = y / height
    
    theta = 2 * np.pi * (u - 0.5)
    phi = np.pi * (v - 0.5)

    ray_dirs = np.stack([
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
        np.cos(phi) * np.cos(theta)
    ], axis=-1)
    ray_dirs = ray_dirs.reshape(-1, 3)
    return torch.tensor(ray_dirs, dtype=torch.float32)

def make_c2w(R, C):
    R = np.array(R)
    C = np.array(C).reshape(3, 1)
    c2w = np.zeros((3,4))
    c2w[:3, :3] = R.T
    c2w[:3, 3] = (-R.T @ C).reshape(-1)
    return torch.tensor(c2w, dtype=torch.float32)

def read_ascii_ply(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Separate header and data
    header_end = lines.index("end_header\n")
    header = lines[:header_end]
    data_lines = lines[header_end + 1:]

    # Parse property order
    property_names = [
        line.strip().split()[-1]
        for line in header
        if line.startswith("property")
    ]

    # Load the data
    data = np.loadtxt(data_lines, dtype=np.float32)

    # Get point positions
    xyz = data[:, [property_names.index('x'),
                   property_names.index('y'),
                   property_names.index('z')]]

    # Get RGB colors
    rgb = data[:, [property_names.index('red'),
                   property_names.index('green'),
                   property_names.index('blue')]] / 255.0

    # Convert to torch tensors
    points3D = torch.tensor(xyz, dtype=torch.float32)
    points3D_color = torch.tensor(rgb, dtype=torch.float32)

    return points3D, points3D_color

class OMNIDataset:
    def __init__(self, datadir, split, downsample):
        images_dir = os.path.join(datadir, "images")
        
        file_path = os.path.join(datadir, f"{split}.txt")
        with open(file_path, "r") as f:
            names = f.read().splitlines()

        names = [name + ".jpg" for name in names]
        names = list(str(name) for name in names)

        im = Image.open(os.path.join(images_dir, names[0]))
        self.img_wh = im.size
        im.close()

        cam_ray_dirs = get_omni_ray_dirs(self.img_wh)

        with open(os.path.join(datadir, "data_views.json")) as f:
            view_data = json.load(f)["views"]

        with open(os.path.join(datadir, "data_extrinsics.json")) as f:
            extrinsics_data = json.load(f)["extrinsics"]

        filename_to_pose = {}
        for ext in extrinsics_data:
            pose_id = ext["key"]
            R = ext["value"]["rotation"]
            C = ext["value"]["center"]
            filename = None
            for view in view_data:
                if view["value"]["ptr_wrapper"]["data"]["id_pose"] == pose_id:
                    filename = view["value"]["ptr_wrapper"]["data"]["filename"]
                    break
            if filename:
                filename_to_pose[filename] = (R, C)
                
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        for name in names:
            R, C = filename_to_pose[name]
            c2w = make_c2w(R, C)
            self.poses.append(c2w)
            world_ray_dirs = torch.einsum(
                "ij,kj->ik",
                cam_ray_dirs,
                c2w[:, :3],
            )

            world_ray_origins = c2w[:, 3] + torch.zeros_like(cam_ray_dirs)
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)

            im = Image.open(os.path.join(images_dir, name))
            im = im.convert("RGB")
            rgbs = torch.tensor(np.array(im), dtype=torch.float32) / 255.0
            im.close()

            self.all_rays.append(world_rays)
            self.all_rgbs.append(rgbs)

        self.poses = torch.stack(self.poses)       # torch.Size([133, 3, 4])
        self.all_rays = torch.stack(self.all_rays) # torch.Size([133, 960, 1920, 6])
        self.all_rgbs = torch.stack(self.all_rgbs) # torch.Size([133, 960, 1920, 3])

        self.points3D, self.points3D_color = read_ascii_ply(os.path.join(datadir, "pcd.ply"))
        
        self.fx = self.img_wh[0]/2*np.pi
        self.fy = self.img_wh[1]/np.pi
