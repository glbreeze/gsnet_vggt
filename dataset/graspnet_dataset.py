""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import numpy as np
import scipy.io as scio
from PIL import Image
from torchvision import transforms as TF
import torchvision.transforms.functional as F

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask


class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True, vggt=False):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = np.float32(voxel_size)
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.vggt = vggt

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, ret_dict, aug=['flip', 'rot']):
        point_clouds, object_poses_list = ret_dict['point_clouds'], ret_dict['object_poses_list']
        
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            if 'image' in ret_dict:
                ret_dict['image'] = F.hflip(ret_dict['image'])
                ret_dict['image_meta']['hflip'] = not ret_dict['image_meta'].get('hflip', False)

        # ---------- rotation: rotation around X (pitch) ----------
        if 'rot' in aug:
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            rot_mat = np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]])
            point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        ret_dict['point_clouds'] = point_clouds
        ret_dict['object_poses_list'] = object_poses_list
        return ret_dict

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud, uv = create_point_cloud_from_depth_image(depth, camera, organized=True, ret_uv=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        uv_masked = uv[mask]

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        uv_sampled = uv_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'uv': uv_sampled.astype(np.int32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        if self.vggt:
            image, image_meta = self.load_and_preprocess_images(self.colorpath[index])
            ret_dict.update({'image': image, 'image_meta': image_meta})
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud, uv = create_point_cloud_from_depth_image(depth, camera, organized=True, ret_uv=True)

        # ---------- get valid points ---------- 
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        uv_masked = uv[mask]

        # ---------- sample points ---------- 
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        uv_sampled = uv_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()

        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx] # (487, 3), (487, 300, 12, 4), (487, 300, 12, 4)
            collision = self.collision_labels[scene][i]  # (Np, V, A, D) / (487, 300, 12, 4)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'uv': uv_sampled.astype(np.int32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'graspness_label': graspness_sampled.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list,
                    # 'depth': depth,
                    # 'color': color, 
                    }
        if self.vggt:
            image, image_meta = self.load_and_preprocess_images(self.colorpath[index])
            ret_dict.update({'image': image, 'image_meta': image_meta})
        
        if self.augment:
            aug = ['flip'] if self.vggt else ['flip', 'rot']
            ret_dict = self.augment_data(ret_dict, aug=aug)
        return ret_dict
    
    def load_and_preprocess_images(self, image_path: str, target_size: int = 518):
        """
        Load an RGB image and preprocess for VGGT.
        Returns:
          img_t: torch.FloatTensor, shape [3, H, W], in [0,1]
          meta: dict with the mapping from original pixel coords -> processed coords
        """
        img = Image.open(image_path)

        # Robust conversion to RGB (handles paletted PNGs etc.)
        if img.mode != "RGB":
            if img.mode == "RGBA": # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img).convert("RGB")
            else:
                img = img.convert("RGB")

        W0, H0 = img.size  # PIL gives (W,H)
        if (W0, H0) != (1280, 720):
            print(f"[warn] unexpected image size: {(W0, H0)} for {image_path}")

        # --- resize keeping aspect ratio ---
        if W0 >= H0:
            new_W = target_size
            new_H = round(H0 * (new_W / W0) / 14) * 14
        else:
            new_H = target_size
            new_W = round(W0 * (new_H / H0) / 14) * 14

        img = img.resize((new_W, new_H), Image.Resampling.BICUBIC)
        img_t = TF.ToTensor()(img)  # [3,new_H,new_W] in [0,1]

        sx = new_W / float(W0)
        sy = new_H / float(H0)

        meta = {
            "orig_size": (H0, W0),
            "resized_size": (new_H, new_W),
            "scale": (sx, sy),
            "hflip": False, 
        }

        return img_t, meta


def load_grasp_labels(root):
    obj_names = list(range(1, 89))
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        
        elif torch.is_tensor(batch[0]):
            shapes = [tuple(x.shape) for x in batch]
            assert len(set(shapes)) == 1, f"tensor shapes mismatch: {shapes}"
            return torch.stack(batch, 0)
        
        elif isinstance(batch[0], (int, float, np.number, bool)):
            return torch.tensor(batch)
        
        elif isinstance(batch[0], (tuple, str)):
            return list(batch)
        
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        
        elif isinstance(batch[0], container_abcs.Mapping):
            out = {}
            for key in batch[0]:
                if key not in ['coors', 'feats']: 
                    out[key] = collate_fn_([d[key] for d in batch])
            return out
        
    res.update(collate_fn_(list_data))

    return res


import numpy as np
import open3d as o3d

def visualize_pointcloud_graspness_open3d(
    sample,
    voxel_downsample=None,
    mode="graspness",          # "graspness" | "rgb" | "blend"
    blend_alpha=0.6,           # only used for mode="blend": alpha*graspness + (1-alpha)*rgb
):
    """
    Visualize point cloud with either:
      - graspness colormap (mode="graspness")
      - raw RGB colors (mode="rgb") requires sample["point_colors"]
      - blended graspness+RGB (mode="blend") requires sample["point_colors"]

    Args:
        sample: dict with keys "point_clouds", "graspness_label", optionally "point_colors"
        voxel_downsample: float in meters, e.g. 0.003; None disables
        mode: "graspness" | "rgb" | "blend"
        blend_alpha: in [0,1], weight on graspness colors for "blend"
    """
    xyz = sample["point_clouds"].astype(np.float64)                 # (N,3)
    g = sample["graspness_label"].reshape(-1).astype(np.float64)    # (N,)

    # ---- graspness -> [0,1] ----
    g_min, g_max = float(np.min(g)), float(np.max(g))
    if g_max > g_min:
        gn = (g - g_min) / (g_max - g_min)
    else:
        gn = np.zeros_like(g)

    # ---- graspness colormap ----
    import matplotlib.cm as cm
    grasp_colors = cm.viridis(gn)[:, :3].astype(np.float64)         # (N,3)

    # ---- optional RGB colors ----
    rgb = None
    if "point_colors" in sample:
        rgb = sample["point_colors"].astype(np.float64)
        # ensure in [0,1]
        rgb = np.clip(rgb, 0.0, 1.0)

    if mode == "graspness":
        colors = grasp_colors
    elif mode == "rgb":
        if rgb is None:
            raise KeyError('mode="rgb" requires sample["point_colors"]')
        colors = rgb
    elif mode == "blend":
        if rgb is None:
            raise KeyError('mode="blend" requires sample["point_colors"]')
        a = float(np.clip(blend_alpha, 0.0, 1.0))
        colors = a * grasp_colors + (1.0 - a) * rgb
        colors = np.clip(colors, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_downsample is not None and voxel_downsample > 0:
        pcd = pcd.voxel_down_sample(float(voxel_downsample))

    title = f"PointCloud ({mode})"
    o3d.visualization.draw_geometries([pcd], window_name=title)
