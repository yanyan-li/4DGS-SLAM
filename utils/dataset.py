import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

import torchvision.transforms as transforms
from scipy.ndimage import label 

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations
        
    # 提取数字部分进行排序
    def extract_number(self, file_path):
        import re
        # 使用正则表达式提取文件路径中的数字部分
        #match = re.search(r'(\d+(\.\d+)?)', file_path)
        match = re.search(r'(\d+)(?=\.png$)', file_path)
        if match:
            return float(match.group(1))  # 返回提取的数字（作为浮动值）
        return 0
        
    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")
        
        self.mask_paths = []
        if os.path.isdir(os.path.join(datapath, "render_mask")):
            self.mask_path = sorted(glob.glob(os.path.join(self.input_folder, "render_mask", "*.png")), key=self.extract_number)
            #print(self.mask_path)
            print("Using mask image")
        else:
            self.mask_path = None
        
        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]
            
            if self.mask_path is not None:
                self.mask_paths += [self.mask_path[i]]
            
            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]
            
            
            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None
        self.yolo_model = None
        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }
        self.dynamic_objects = 0
        self.seg_chair = True if "seg_chair" in config["Dataset"].keys() else False 
        self.start = calibration["start"] if "start" in calibration.keys() else 0
        self.end = calibration["end"] if "end" in calibration.keys() else -1

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        mask_input = torch.zeros((image.shape[1], image.shape[2]), device=self.device, dtype=torch.bool)
        if self.mask_paths:
            mask_img = Image.open(self.mask_paths[idx]).convert("L")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            mask_tensor = transform(mask_img)
            threshold = 0.01
            mask = (mask_tensor > threshold).float().to(torch.bool).squeeze()
            mask_input = mask.to(device=self.device)
            #return image, depth, pose, ~mask
        
        combined_mask = torch.zeros((image.shape[1], image.shape[2]), device=self.device, dtype=torch.bool)
        
        if self.yolo_model is not None:
            results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[0], save=False, stream=False, show=False, verbose=False, device=self.device)
            for result in results:
                masks = result.masks
                if masks is not None:
                    for mask in masks.data:
                        mask = mask.to(torch.bool)
                        combined_mask |= mask 
        if self.yolo_model is not None and self.seg_chair:
            results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[56], save=False, stream=False, show=False, verbose=False, device=self.device)
            for result in results:
                masks = result.masks
                if masks is not None:
                    for mask in masks.data:
                        mask = mask.to(torch.bool)
                        combined_mask |= mask
        
        
        combined_mask = torch.logical_not(combined_mask | mask_input)
        return image, depth, pose, combined_mask


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose

class CoFusion(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        self.input_folder = dataset_path
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'colour', '*.png')))
            
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, 'depth_noise', '*.exr'))+glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        #self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth_original", "*.exr")))
        self.mask_paths = sorted(
            glob.glob(os.path.join(self.input_folder,"mask_colour", "*.png"))
        )
        self.num_imgs = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'trajectories'))
        calibration = config["Dataset"]["Calibration"]
        
        self.start = calibration["start"] if "start" in calibration.keys() else 0
        self.end = calibration["end"] if "end" in calibration.keys() else self.num_imgs
        print(self.start, "    ", self.end)
        if self.end==-1:
            self.end = self.num_imgs
        self.color_paths = self.color_paths[self.start : self.end]
        self.depth_paths = self.depth_paths[self.start : self.end]
        self.poses = self.poses[self.start : self.end]
        if self.mask_paths:
            self.mask_paths = self.mask_paths[self.start : self.end]
        self.num_imgs = len(self.color_paths)
        
        self.dynamic = config["model_params"]["dynamic_model"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None
        self.yolo_model = None
        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }
        self.dynamic_objects = 0
        self.seg_chair = config["Dataset"]["seg_chair"] if "seg_chair" in config["Dataset"].keys() else False 
        self.seg_teddy = config["Dataset"]["seg_teddy"] if "seg_teddy" in config["Dataset"].keys() else False
        self.seg_clock = config["Dataset"]["seg_clock"] if "seg_clock" in config["Dataset"].keys() else False
        if self.seg_teddy or self.seg_clock:
            self.color_paths = sorted(self.color_paths, key=self.extract_number)
            self.depth_paths = sorted(self.depth_paths, key=self.extract_number)
    
    # 提取数字部分进行排序
    def extract_number(self, file_path):
        import re
        # 使用正则表达式提取文件路径中的数字部分
        match = re.search(r'(\d+)(?=\.png$)', file_path)
        #match = re.search(r'(\d+(\.\d+)?)', file_path)
        if match:
            return float(match.group(1))  # 返回提取的数字（作为浮动值）
        return 0
    
    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data
        
    def load_poses(self, path):
        self.poses = []
        self.frames = []
        if os.path.isfile(os.path.join(path, "gt-cam-0.txt")):
            pose_list = os.path.join(path, "gt-cam-0.txt")
            pose_data = self.parse_list(pose_list)
            pose_vecs = pose_data[:, 0:].astype(np.float64)
            for i in range(self.num_imgs):
                quat = pose_vecs[i][4:]
                trans = pose_vecs[i][1:4]
                T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
                T[:3, 3] = trans
                self.poses += [np.linalg.inv(T)]
                frame = {
                    "file_path": self.color_paths[i],
                    "depth_path": self.depth_paths[i],
                    "transform_matrix": np.linalg.inv(T).tolist(),
                }
                self.frames.append(frame)
        else:
            self.poses = [np.eye(4) for _ in range(self.num_imgs)]
            
    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        
        image = np.array(Image.open(color_path))
        depth = None
        
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = self.poses[index]
        pose = torch.from_numpy(pose).to(device=self.device)
        
        if '.png' in depth_path:
            depth_data = np.array(Image.open(depth_path))
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
            
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        H, W = depth_data.shape
        if self.mask_paths:
            mask_img = Image.open(self.mask_paths[index]).convert("L")
            transform = transforms.Compose([
                transforms.ToTensor(), 
            ])
            
            mask_tensor = transform(mask_img)
            threshold = 0.01
            mask = (mask_tensor > threshold).float().to(torch.bool).squeeze()
            mask = mask.to(device=self.device)
        else:
            combined_mask = torch.zeros((image.shape[1], image.shape[2]), device=self.device, dtype=torch.bool)
            if self.yolo_model is not None:
                results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[0], save=False, stream=False, show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            combined_mask |= mask 
            if self.yolo_model is not None and self.seg_clock:
                results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[74], save=False, stream=False, show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            combined_mask |= mask
            if self.yolo_model is not None and self.seg_teddy:
                results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[77], save=False, stream=False, show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            combined_mask |= mask
            mask = combined_mask
        
        #labeled_mask, num_features = label(mask.cpu().numpy())  # 物体数量变化添加新node和高斯时去掉边缘区域，初始化迭代优化高斯和mlp
        #self.dynamic_objects = 0
        #for region in range(1, num_features + 1):
        #    current_region = (labeled_mask == region)
        #    region_size = np.sum(current_region)
        #    if np.any(current_region[0, :]) or np.any(current_region[-1, :]) or \
        #       np.any(current_region[:, 0]) or np.any(current_region[:, -1]):
        #        #self.has_obj=True
        #        continue
        #    elif region_size<200:
        #        continue
        #    else:
        #        self.dynamic_objects += 1
        #print("dynamic objects: ", self.dynamic_objects)
        #if self.crop_size is not None:
        #    # follow the pre-processing step in lietorch, actually is resize
        #    color_data = color_data.permute(2, 0, 1)
        #    color_data = F.interpolate(
        #        color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
        #    depth_data = F.interpolate(
        #        depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
        #    color_data = color_data.permute(1, 2, 0).contiguous()

        #edge = self.crop_edge
        #if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
        #    color_data = color_data[edge:-edge, edge:-edge]
        #    depth_data = depth_data[edge:-edge, edge:-edge]
        
        return image, depth_data, pose, ~mask

class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mask_paths = parser.mask_paths
        self.poses = parser.poses
        if self.end==-1:
            self.end = self.num_imgs
        self.color_paths = self.color_paths[self.start : self.end]
        self.depth_paths = self.depth_paths[self.start : self.end]
        self.poses = self.poses[self.start : self.end]
        self.num_imgs = len(self.color_paths)


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose

class realDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        self.input_folder = dataset_path
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.png')))
            
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        #self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth_original", "*.exr")))
        self.mask_paths = sorted(
            glob.glob(os.path.join(self.input_folder,"mask_colour", "*.png"))
        )
        self.color_paths = sorted(self.color_paths, key=self.extract_number)
        self.depth_paths = sorted(self.depth_paths, key=self.extract_number)
        self.num_imgs = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'trajectories'))
        calibration = config["Dataset"]["Calibration"]
        
        self.start = calibration["start"] if "start" in calibration.keys() else 0
        self.end = calibration["end"] if "end" in calibration.keys() else self.num_imgs
        print(self.start, "    ", self.end)
        if self.end==-1:
            self.end = self.num_imgs
        self.color_paths = self.color_paths[self.start : self.end]
        self.depth_paths = self.depth_paths[self.start : self.end]
        self.poses = self.poses[self.start : self.end]
        if self.mask_paths:
            self.mask_paths = self.mask_paths[self.start : self.end]
        self.num_imgs = len(self.color_paths)
        
        self.dynamic = config["model_params"]["dynamic_model"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None
        self.yolo_model = None
        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }
        self.dynamic_objects = 0
        self.seg_chair = config["Dataset"]["seg_chair"] if "seg_chair" in config["Dataset"].keys() else False 
    
    # 提取数字部分进行排序
    def extract_number(self, file_path):
        import re
        # 使用正则表达式提取文件路径中的数字部分
        match = re.search(r'(\d+)(?=\.png$)', file_path)
        #match = re.search(r'(\d+(\.\d+)?)', file_path)
        if match:
            return float(match.group(1))  # 返回提取的数字（作为浮动值）
        return 0
    
    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data
        
    def load_poses(self, path):
        self.poses = []
        self.frames = []
        if os.path.isfile(os.path.join(path, "gt-cam-0.txt")):
            pose_list = os.path.join(path, "gt-cam-0.txt")
            pose_data = self.parse_list(pose_list)
            pose_vecs = pose_data[:, 0:].astype(np.float64)
            for i in range(self.num_imgs):
                quat = pose_vecs[i][4:]
                trans = pose_vecs[i][1:4]
                T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
                T[:3, 3] = trans
                self.poses += [np.linalg.inv(T)]
                frame = {
                    "file_path": self.color_paths[i],
                    "depth_path": self.depth_paths[i],
                    "transform_matrix": np.linalg.inv(T).tolist(),
                }
                self.frames.append(frame)
        else:
            self.poses = [np.eye(4) for _ in range(self.num_imgs)]
            
    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        
        image = np.array(Image.open(color_path))
        depth = None
        
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = self.poses[index]
        pose = torch.from_numpy(pose).to(device=self.device)
        
        if '.png' in depth_path:
            depth_data = np.array(Image.open(depth_path))
            
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        H, W = depth_data.shape
        if self.mask_paths:
            mask_img = Image.open(self.mask_paths[index]).convert("L")
            transform = transforms.Compose([
                transforms.ToTensor(), 
            ])
            
            mask_tensor = transform(mask_img)
            threshold = 0.01
            mask = (mask_tensor > threshold).float().to(torch.bool).squeeze()
            mask = mask.to(device=self.device)
        else:
            combined_mask = torch.zeros((image.shape[1], image.shape[2]), device=self.device, dtype=torch.bool)
            if self.yolo_model is not None:
                results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[0], save=False, stream=False, show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            combined_mask |= mask 
            if self.yolo_model is not None and False:
                results = self.yolo_model.predict(source=image.unsqueeze(0), classes=[74], save=False, stream=False, show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            combined_mask |= mask
        
        return image, depth_data, pose, ~mask


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "CoFusion":
        return CoFusion(args, path, config) 
    elif config["Dataset"]["type"] == "real":
        return realDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
