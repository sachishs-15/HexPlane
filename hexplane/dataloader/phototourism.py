import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from .intrinsics import Intrinsics
from .ray_utils import average_poses
import logging as log
import glob
import pandas as pd
from pathlib import Path
from enum import Enum
from typing import Optional, List, Tuple
import torchvision.transforms.functional as TF




import math


from .ray_utils import get_ray_directions_blender, get_rays, read_pfm

blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

class PhototourismScenes(Enum):
    TREVI = "trevi"
    BRANDENBURG = "brandenburg"
    SACRE = "sacre"

    @staticmethod
    def get_scene_from_datadir(datadir: str) -> 'PhototourismScenes':
        if "sacre" in datadir:
            return PhototourismScenes.SACRE
        if "trevi" in datadir:
            return PhototourismScenes.TREVI
        if "brandenburg" in datadir:
            return PhototourismScenes.BRANDENBURG
        raise NotImplementedError(datadir)


def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()

def read_png(file_name: str, resize_h: Optional[int] = None, resize_w: Optional[int] = None) -> torch.Tensor:
    """Reads a PNG image from path, potentially resizing it.
    """
    img = Image.open(file_name).convert('RGB')  # PIL outputs BGR by default
    if resize_h is not None and resize_w is not None:
        img.resize((resize_w, resize_h), Image.LANCZOS)
    img = TF.to_tensor(img)  # TF converts to C, H, W
    img = img.permute(1, 2, 0).contiguous()  # H, W, C
    return img

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    return c2w

def load_camera_metadata(datadir: str, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        poses = np.load(str(Path(datadir) / "c2w_mats.npy"))[idx]
        kinvs = np.load(str(Path(datadir) / "kinv_mats.npy"))[idx]
        bounds = np.load(str(Path(datadir) / "bds.npy"))[idx]
        res = np.load(str(Path(datadir) / "res_mats.npy"))[idx]
    except FileNotFoundError as e:
        error_msg = (
            f"One of the needed Phototourism files does not exist ({e.filename}). "
            f"They can be downloaded from "
            f"https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW"
        )
        log.error(error_msg)
        raise e
    return poses, kinvs, bounds, res


def get_ids_for_split(datadir, split):
    # read all files in the tsv first (split to train and test later)
    tsv = glob.glob(os.path.join(datadir, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()]  # remove data without id
    files.reset_index(inplace=True, drop=True)
    files = files[files["split"] == split]

    imagepaths = sorted((Path(datadir) / "dense" / "images").glob("*.jpg"))
    imkey = np.array([os.path.basename(im) for im in imagepaths])
    idx = np.in1d(imkey, files["filename"])
    return idx, imagepaths


def scale_cam_metadata(poses: np.ndarray, kinvs: np.ndarray, bounds: np.ndarray, scale: float = 0.05):
    poses[:, :3, 3:4] *= scale
    bounds = bounds * scale * np.array([0.9, 1.2])

    return poses, kinvs, bounds


def cache_data(datadir: str, split: str, out_fname: str):

    scale = 0.05
    idx, imagepaths = get_ids_for_split(datadir, split)

    imagepaths = np.array(imagepaths)[idx]
    poses, kinvs, bounds, res = load_camera_metadata(datadir, idx)
    poses, kinvs, bounds = scale_cam_metadata(poses, kinvs, bounds, scale=scale)
    img_w = res[:, 0]
    img_h = res[:, 1]
    size = int(np.sum(img_w * img_h))
    log.info(f"Loading dataset from {datadir}. Num images={len(imagepaths)}. Total rays={size}.")

    all_images, all_rays_o, all_rays_d, all_bounds, all_camera_ids = [], [], [], [], []
    for idx, impath in enumerate(imagepaths):
        image = read_png(impath)

        pose = torch.from_numpy(poses[idx]).float()
        kinv = torch.from_numpy(kinvs[idx]).float()
        bound = torch.from_numpy(bounds[idx]).float()

        rays_o, rays_d = get_rays_tourism(image.shape[0], image.shape[1], kinv, pose)

        camera_id = torch.tensor(idx)

        all_images.append(image.mul(255).to(torch.uint8))
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_bounds.append(bound)
        all_camera_ids.append(camera_id)

    meta_data = {
        "images": all_images,
        "rays_o": all_rays_o,
        "rays_d": all_rays_d,
        "bounds": all_bounds,
        "camera_ids": all_camera_ids,
    }

    torch.save({
        "images": all_images,
        "rays_o": all_rays_o,
        "rays_d": all_rays_d,
        "bounds": all_bounds,
        "camera_ids": all_camera_ids,
    }, os.path.join(datadir, out_fname))

    return meta_data

def get_rays_tourism(H, W, kinv, pose):
    """
    phototourism camera intrinsics are defined by H, W and kinv.
    Args:
        H: image height
        W: image width
        kinv (3, 3): inverse of camera intrinsic
        pose (4, 4): camera extrinsic
    Returns:
        rays_o (H, W, 3): ray origins
        rays_d (H, W, 3): ray directions
    """
    yy, xx = torch.meshgrid(torch.arange(0., H, device=kinv.device),
                            torch.arange(0., W, device=kinv.device),
                            indexing='ij')
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

    directions = torch.matmul(pixco, kinv.T)  # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)

    rays_o = pose[:3, -1].expand_as(rays_d)  # (H, W, 3)

    return rays_o, rays_d

class Phototourism(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=2.0,
        is_stack=False,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
    ):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.world_bound_scale = 1.1

        self.near = 2.0
        self.far = 6.0
        self.near_far = [2.0, 6.0]

        self.define_transforms()  # transform to torch.Tensor

        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        if split == 'train' or split == 'test':
            pt_data_file = os.path.join(datadir, f"cache_{split}.pt")
            if not os.path.isfile(pt_data_file):
                # Populate cache
                self.meta_data = cache_data(datadir=datadir, split=split, out_fname=os.path.basename(pt_data_file))
            pt_data = torch.load(pt_data_file)

            intrinsics = [
                Intrinsics(width=img.shape[1], height=img.shape[0],
                           center_x=img.shape[1] / 2, center_y=img.shape[0] / 2,
                           # focals are unused, we reuse intrinsics from Matt's files.
                           focal_y=0, focal_x=0)
                for img in pt_data["images"]
            ]

        if split == 'train':
                near_far = pt_data["bounds"]

                # near_fars = torch.cat([
                #     pt_data["bounds"][i].expand(intrinsics[i].width * intrinsics[i].height, 2)
                #     for i in range(len(intrinsics))
                # ], dim=0)
                camera_ids = pt_data["camera_ids"]

                # camera_ids = torch.cat([
                #     pt_data["camera_ids"][i].expand(intrinsics[i].width * intrinsics[i].height, 1)
                #     for i in range(len(intrinsics))
                # ])

                images = torch.cat([img.view(-1, 3) for img in pt_data["images"]], 0)
                rays_o = torch.cat([ro.view(-1, 3) for ro in pt_data["rays_o"]], 0)
                rays_d = torch.cat([rd.view(-1, 3) for rd in pt_data["rays_d"]], 0)

        elif split == 'test':
                images = pt_data["images"]
                rays_o = pt_data["rays_o"]
                rays_d = pt_data["rays_d"]
                near_far = pt_data["bounds"]
                camera_ids = pt_data["camera_ids"]

        elif split == 'render':
            n_frames, frame_size = 150, 800
            rays_o, rays_d, camera_ids, near_fars = pt_render_poses(
                datadir, n_frames=n_frames, frame_h=frame_size, frame_w=frame_size)
            images = None
            intrinsics = [
                Intrinsics(width=frame_size, height=frame_size, focal_x=0, focal_y=0,
                           center_x=frame_size / 2, center_y=frame_size / 2)
                for _ in range(n_frames)
            ]
        else:
            raise NotImplementedError(split)
        
        # torch.cat each corresponding elements in the form torch.cat([rays_o, rays_d], 1)


        self.all_rays = []  # (h*w, 6)
        for x in len(len(self.all_rays)):
            self.all_rays += torch.cat([rays_o[x], rays_d[x]], 1)

        self.all_rgbs = []

        for img in images: # converted to white background dont know if its correct --------
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (
                    1 - img[:, -1:]
                )  # blend A to RGB, white background
                self.all_rgbs += [img]
        
        #self.poses = []
        self.all_times = camera_ids #### to check
        #self.all_depth = []
        self.near = near_far[:, 0]
        self.far = near_far[:, 1]


        # self.read_meta()  # Read meta data

        # Calculate a more fine bbox based on near and far values of each ray.
        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        self.define_proj_mat()

        self.white_bg = True
        self.ndc_ray = False
        self.depth_data = False

        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def init_random_pose(self):
        # Randomly sample N_random_pose radius, phi, theta and times.
        radius = np.random.randn(self.N_random_pose) * 0.1 + 4
        phi = np.random.rand(self.N_random_pose) * 360 - 180
        theta = np.random.rand(self.N_random_pose) * 360 - 180
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 2.0 - 1.0)
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        self.random_rays = []
        for i in range(self.N_random_pose):
            random_poses = pose_spherical(theta[i], phi[i], radius[i])
            rays_o, rays_d = get_rays(self.directions, random_poses)
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]

        self.random_rays = torch.stack(self.random_rays, 0).reshape(
            -1, *self.img_wh[::-1], 6
        )

    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        rays_o = self.all_rays[:, 0:3]
        viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    

    def read_meta(self):
        # with open(os.path.join(self.root_dir, f"transforms_{self.split}.json")) as f:
        #     self.meta = json.load(f)

        # w, h = self.img_wh
        # self.focal = (
        #     0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        # )  # original focal length

        # self.focal *= (
        #     self.img_wh[0] / 800
        # )  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)

        mdata = self.meta_data

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_times = []
        self.all_rgbs = []
        self.all_depth = []

        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            frame = self.meta["frames"][i]
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (
                1 - img[:, -1:]
            )  # blend A to RGB, white background
            self.all_rgbs += [img]

            rays_o, rays_d = mdata['rays_o']
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            cur_time = torch.tensor(
                frame["time"]
                if "time" in frame
                else float(i) / (len(self.meta["frames"]) - 1)
            ).expand(rays_o.shape[0], 1)
            self.all_times += [cur_time]

        self.poses = torch.stack(self.poses)
        #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
        #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times, 0)

        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times, 0)

        self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "time": self.all_times[idx],
            }
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            sample = {"rays": rays, "rgbs": img, "time": time}
        return sample

    def get_random_pose(self, batch_size, patch_size, batching="all_images"):
        """
        Apply Geometry Regularization from RegNeRF.
        This function randomly samples many patches from random poses.
        """
        n_patches = batch_size // (patch_size**2)

        N_random = self.random_rays.shape[0]
        # Sample images
        if batching == "all_images":
            idx_img = np.random.randint(0, N_random, size=(n_patches, 1))
        elif batching == "single_image":
            idx_img = np.random.randint(0, N_random)
            idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
        else:
            raise ValueError("Not supported batching type!")
        idx_img = torch.Tensor(idx_img).long()
        H, W = self.random_rays[0].shape[0], self.random_rays[0].shape[1]
        # Sample start locations
        x0 = np.random.randint(
            int(W // 4), int(W // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        y0 = np.random.randint(
            int(H // 4), int(H // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        xy0 = np.concatenate([x0, y0], axis=-1)
        patch_idx = xy0 + np.stack(
            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="xy"),
            axis=-1,
        ).reshape(1, -1, 2)

        patch_idx = torch.Tensor(patch_idx).long()
        # Subsample images
        out = self.random_rays[idx_img, patch_idx[..., 1], patch_idx[..., 0]]

        return out, self.random_times[idx_img]
    
def pt_spiral_path(
        scene: PhototourismScenes,
        poses: torch.Tensor,
        n_frames=120,
        n_rots: float = 1.0,
        zrate=.5) -> torch.Tensor:
    if poses.shape[1] > 3:
        poses = poses[:, :3, :]
    c2w = torch.from_numpy(average_poses(poses.numpy()))  # [3, 4]

    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        rotation = c2w[:3, :3]
        # the additive translation vectors have 3 components (x, y, z axes)
        # each with an additive part - which defines a global shift of all poses
        # and a multiplicative part which changes the amplitude of movement
        # of the poses.
        if scene == PhototourismScenes.SACRE:
            translation = c2w[:, 3:4] + torch.tensor([[
                0.01 + 0.03 * np.cos(theta),
                -0.007 * np.sin(theta),
                0.06 + 0.03 * np.sin(theta * zrate)
            ]]).T
        elif scene == PhototourismScenes.BRANDENBURG:
            translation = c2w[:, 3:4] + torch.tensor([[
                0.08 * np.cos(theta),
                -0.07 - 0.01 * np.sin(theta),
                -0.0 + 0.1 * np.sin(theta * zrate)
            ]]).T
        elif scene == PhototourismScenes.TREVI:
            translation = c2w[:, 3:4] + torch.tensor([[
                -0.05 + 0.2 * np.cos(theta),
                -0.07 - 0.07 * np.sin(theta),
                0.02 + 0.05 * np.sin(theta * zrate)
            ]]).T
        else:
            raise NotImplementedError(scene)
        pose = torch.cat([rotation, translation], dim=1)
        render_poses.append(pose)
    return torch.stack(render_poses, dim=0)

def pt_render_poses(datadir: str, n_frames: int, frame_h: int = 800, frame_w: int = 800):
    scene = PhototourismScenes.get_scene_from_datadir(datadir)
    idx, _ = get_ids_for_split(datadir, split='train')
    train_poses, kinvs, bounds, res = load_camera_metadata(datadir, idx)
    train_poses, kinvs, bounds = scale_cam_metadata(train_poses, kinvs, bounds, scale=0.05)

    # build camera intrinsic from appropriate focal distance and cx, cy. Good for trevi
    k = np.array([[780.0, 0, -frame_w / 2], [0, -780, -frame_h / 2], [0, 0, -1]])
    kinv = torch.from_numpy(np.linalg.inv(k)).to(torch.float32)

    bounds = torch.from_numpy(bounds).float()
    train_poses = torch.from_numpy(train_poses).float()

    r_poses = pt_spiral_path(scene, train_poses, n_frames=n_frames, zrate=1, n_rots=1)

    all_rays_o, all_rays_d, camera_ids, near_fars = [], [], [], []
    for pose_id, pose in enumerate(r_poses):
        pose = pose.float()
        rays_o, rays_d = get_rays_tourism(frame_h, frame_w, kinv, pose)
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

        # Find the closest cam
        closest_cam_idx = torch.linalg.norm(
            train_poses[:, :3, :].view(train_poses.shape[0], -1) - pose.view(-1), dim=1
        ).argmin()

        # For brandenburg and trevi
        if scene == PhototourismScenes.BRANDENBURG or scene == PhototourismScenes.TREVI:
            near_fars.append((
                bounds[closest_cam_idx] + torch.tensor([0.01, 0.0])
            ))
        elif scene == PhototourismScenes.SACRE:
            near_fars.append((
                bounds[closest_cam_idx] + torch.tensor([0.05, 0.0])
            ))
    # camera-IDs. They are floats interpolating between 2 appearance embeddings.
    x = torch.linspace(-1, 1, len(r_poses))
    s = 0.3
    camera_ids = 1/(s * math.sqrt(2 * torch.pi)) * torch.exp(- (x)**2 / (2 * s**2))
    camera_ids = (camera_ids - camera_ids.min()) / camera_ids.max()
    all_rays_o = torch.stack(all_rays_o, dim=0)
    all_rays_d = torch.stack(all_rays_d, dim=0)
    near_fars = torch.stack(near_fars, dim=0)
    return all_rays_o, all_rays_d, camera_ids, near_fars