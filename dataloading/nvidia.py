import sys
from pathlib import Path
import os

import cv2
import numpy as np
import pandas as pd
import torch
from math import ceil

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

from skimage.util import random_noise


class NvidiaResizeAndCrop(object):
    def __call__(self, data):
        xmin = 186
        ymin = 600

        scale = 6.0
        width = 258
        height = 66
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        cropped = transforms.functional.resized_crop(data["image"], ymin, xmin, scaled_height, scaled_width,
                                                     (height, width))

        data["image"] = cropped
        return data


class NvidiaCropWide(object):
    def __init__(self, x_delta=0):
        self.x_delta = x_delta

    def __call__(self, data):
        xmin = 300
        xmax = 1620

        ymin = 520
        ymax = 864

        scale = 0.2

        height = ymax - ymin
        width = xmax - xmin
        cropped = F.resized_crop(data["image"], ymin, xmin + self.x_delta, height, width,
                                                     (int(scale * height), int(scale * width)))

        data["image"] = cropped
        return data


class CropViT(object):
    def __call__(self, data):
        xmin = 540
        xmax = 1260

        ymin = 244
        ymax = 964

        scale = 0.312

        height = ymax - ymin
        width = xmax - xmin
        cropped = F.resized_crop(data["image"], ymin, xmin, height, width,
                                                     (int(scale * height), int(scale * width)))
        data["image"] = cropped
        return data


class NvidiaSideCameraZoom(object):

    def __init__(self, zoom_ratio):
        self.zoom_ratio = zoom_ratio

    def __call__(self, data):
        width = 1920
        height = 1208

        xmin = int(self.zoom_ratio * width)
        ymin = int(self.zoom_ratio * height)

        scaled_width = width - (2 * xmin)
        scaled_height = height - (2 * ymin)

        cropped = F.resized_crop(data["image"], ymin, xmin, scaled_height, scaled_width,
                                                     (height, width))

        data["image"] = cropped
        return data


class AugmentationConfig:
    def __init__(self, color_prob=0.0, noise_prob=0.0, blur_prob=0.0):
        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob


class AugmentImage:
    def __init__(self, augment_config):
        print(f"augmentation: color_prob={augment_config.color_prob}, "
              f"noise_prob={augment_config.noise_prob}, "
              f"blur_prob={augment_config.blur_prob}")
        self.augment_config = augment_config

    def __call__(self, data):
        if np.random.random() <= self.augment_config.color_prob:
            jitter = transforms.ColorJitter(contrast=0.5, saturation=0.5, brightness=0.5)
            data["image"] = jitter(data["image"])

        if np.random.random() <= self.augment_config.noise_prob:
            if np.random.random() > 0.5:
                data["image"] = torch.tensor(random_noise(data["image"], mode='gaussian', mean=0, var=0.005, clip=True),
                                             dtype=torch.float)
            else:
                data["image"] = torch.tensor(random_noise(data["image"], mode='salt', amount=0.005),
                                             dtype=torch.float)

        if np.random.random() <= self.augment_config.blur_prob:
            blurrer = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 1))
            data["image"] = blurrer(data['image'])

        return data


class Normalize(object):
    def __call__(self, data, transform=None):
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = data["image"]
        image = image / 255
        # data["image"] = normalize(image)
        data["image"] = image
        return data


class NvidiaDataset(Dataset):

    def __init__(self, dataset_paths, transform=None, camera="front_wide", name="Nvidia dataset",
                 filter_turns=False, metadata_file="nvidia_frames.csv", vehicle_cmd_file="vehicle_cmd.csv", color_space="rgb", dataset_proportion=1.0):
        self.name = name
        self.metadata_file = metadata_file
        self.color_space = color_space
        self.dataset_paths = dataset_paths
        if transform:
            self.transform = transform
            print(f'[NvidiaDataset] Using transform from argument:', self.transform)
        else:
            self.transform = transforms.Compose([Normalize()])
            print(f'[NvidiaDataset] Using default transform:', self.transform)

        self.camera_name = camera
        self.target_size = 1

        datasets = [self.read_dataset(dataset_path, camera) for dataset_path in self.dataset_paths]
        self.frames = pd.concat(datasets)
        keep_n_frames = ceil(len(self.frames) * dataset_proportion)
        self.frames = self.frames.head(keep_n_frames)

        self.vehicle_cmd_frames = None
        self.vehicle_cmd_frames = [pd.read_csv(dataset_path / vehicle_cmd_file) for dataset_path in self.dataset_paths if (dataset_path / vehicle_cmd_file) in dataset_path.glob("*.csv")]
        self.vehicle_cmd_frames = pd.concat(self.vehicle_cmd_frames) if len(self.vehicle_cmd_frames) else None

        if filter_turns:
            print("Filtering turns with blinker signal")
            self.frames = self.frames[self.frames.turn_signal == 1]

    def __getitem__(self, idx):
        frame = self.frames.iloc[idx]
        if self.color_space == "rgb":
            image = torchvision.io.read_image(frame["image_path"])
        elif self.color_space == "bgr":
            image = cv2.imread(frame["image_path"])
            image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
        else:
            print(f"Unknown color space: ", self.color_space)
            sys.exit()

        data = {
            'image': image,
            'steering_angle': np.array(frame["steering_angle"]),
            'vehicle_speed': np.array(frame["vehicle_speed"]),
            'autonomous': np.array(frame["autonomous"]),
            'position_x': np.array(frame["position_x"]),
            'position_y': np.array(frame["position_y"]),
            'yaw': np.array(frame["yaw"]),
            'turn_signal': np.array(frame["turn_signal"]),
            'row_id': np.array(frame["row_id"]),
            'timestamp': np.array(frame["index"]).astype('datetime64[ns]').astype(np.float64),
        }

        target_values = frame["steering_angle"]

        if self.transform:
            data = self.transform(data)

        target = np.zeros((1, self.target_size))
        target[0, :] = target_values
        conditional_mask = np.ones((1, self.target_size))

        return data, target.reshape(-1), conditional_mask.reshape(-1)

    def __len__(self):
        return len(self.frames.index)

    def read_dataset(self, dataset_path, camera):
        if type(dataset_path) is dict:
            frames_df = pd.read_csv(dataset_path['path'] / self.metadata_file)
            len_before_filtering = len(frames_df)
            frames_df = frames_df.iloc[dataset_path['start']:dataset_path['end']]
            dataset_path = dataset_path['path']
        else:
            frames_df = pd.read_csv(dataset_path / self.metadata_file)
            len_before_filtering = len(frames_df)

        frames_df["row_id"] = frames_df.index

        # temp hack
        if "autonomous" not in frames_df.columns:
            frames_df["autonomous"] = False
        # frames_df["autonomous"] = False

        frames_df = frames_df[frames_df['steering_angle'].notna()]  # TODO: one steering angle is NaN, why?
        frames_df = frames_df[frames_df['vehicle_speed'].notna()]
        frames_df = frames_df[frames_df[f'{camera}_filename'].notna()]

        frames_df["turn_signal"].fillna(1, inplace=True)
        frames_df["turn_signal"] = frames_df["turn_signal"].astype(int)

        # Removed frames marked as skipped
        frames_df = frames_df[frames_df["turn_signal"] != -1]  # TODO: remove magic values.

        len_after_filtering = len(frames_df)

        camera_images = frames_df[f"{camera}_filename"].to_numpy()
        frames_df["image_path"] = [str(dataset_path / image_path) for image_path in camera_images]

        print(f"{dataset_path}: length={len(frames_df)}, filtered={len_before_filtering-len_after_filtering}")
        frames_df.reset_index(inplace=True)
        return frames_df

    def steering_angles_degrees(self):
        return self.frames.steering_angle.to_numpy() / np.pi * 180


class NvidiaDatasetGrouped(NvidiaDataset):
    def __init__(self, *args, **kwargs):
        self.group_size = kwargs.pop('group_size', 2)
        super().__init__(*args, **kwargs)


    def __len__(self):
        if self.group_size == 1: return super().__len__()

        return len(self.frames.index) // self.group_size

    def collate_fn(self, batch):
        data, targets, _ = default_collate(batch)
        if self.group_size == 1: return data, targets, _
        for k, v in data.items():
            data[k] = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])

        targets = targets.reshape(targets.shape[0] * targets.shape[1], *targets.shape[2:])  
        return data, targets, _

    def __getitem__(self, idx):
        if self.group_size == 1: return super().__getitem__(idx)

        idx = idx * self.group_size

        frames = self.frames.iloc[idx:idx+self.group_size]
        if len(frames) == 1:
            frames = pd.concat(frames * self.group_size)
        if self.color_space == "rgb":
            images = [torchvision.io.read_image(frame["image_path"]) for _, frame in frames.iterrows()]
        elif self.color_space == "bgr":
            images = [cv2.imread(frame["image_path"]) for _, frame in frames.iterrows()]
            images = [torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1) for image in images]
        else:
            print(f"Unknown color space: ", self.color_space)
            sys.exit()

        images = torch.stack(images)

        data = {
            'image': images,
            'steering_angle': np.array(frames["steering_angle"]),
            'vehicle_speed': np.array(frames["vehicle_speed"]),
            'autonomous': np.array(frames["autonomous"]),
            'position_x': np.array(frames["position_x"]),
            'position_y': np.array(frames["position_y"]),
            'yaw': np.array(frames["yaw"]),
            'turn_signal': np.array(frames["turn_signal"]),
            'row_id': np.array(frames["row_id"]),
        }

        target_values = np.array(frames["steering_angle"])

        if self.transform:
            data = self.transform(data)

        dummy = np.ones((self.group_size, self.target_size))

        return data, target_values.reshape(-1, 1), dummy

class NvidiaTrainDataset(NvidiaDatasetGrouped):
    def __init__(self, root_path, camera="front_wide", **kwargs):
        train_paths = [
            root_path / "2021-05-20-12-36-10_e2e_sulaoja_20_30",
            root_path / "2021-05-20-12-43-17_e2e_sulaoja_20_30",
            root_path / "2021-05-20-12-51-29_e2e_sulaoja_20_30",
            root_path / "2021-05-20-13-44-06_e2e_sulaoja_10_10",
            root_path / "2021-05-20-13-51-21_e2e_sulaoja_10_10",
            root_path / "2021-05-20-13-59-00_e2e_sulaoja_10_10",
            root_path / "2021-05-28-15-07-56_e2e_sulaoja_20_30",
            root_path / "2021-05-28-15-17-19_e2e_sulaoja_20_30",
            root_path / "2021-06-09-13-14-51_e2e_rec_ss2",
            root_path / "2021-06-09-13-55-03_e2e_rec_ss2_backwards",
            root_path / "2021-06-09-14-58-11_e2e_rec_ss3",
            root_path / "2021-06-09-15-42-05_e2e_rec_ss3_backwards",
            root_path / "2021-06-09-16-24-59_e2e_rec_ss13",
            root_path / "2021-06-09-16-50-22_e2e_rec_ss13_backwards",
            root_path / "2021-06-10-12-59-59_e2e_ss4",
            root_path / "2021-06-10-13-19-22_e2e_ss4_backwards",
            root_path / "2021-06-10-13-51-34_e2e_ss12",
            root_path / "2021-06-10-14-02-24_e2e_ss12_backwards",
            root_path / "2021-06-10-14-44-24_e2e_ss3_backwards",
            root_path / "2021-06-10-15-03-16_e2e_ss3_backwards",
            root_path / "2021-06-14-11-08-19_e2e_rec_ss14",
            root_path / "2021-06-14-11-22-05_e2e_rec_ss14",
            root_path / "2021-06-14-11-43-48_e2e_rec_ss14_backwards",
            root_path / "2021-09-24-11-19-25_e2e_rec_ss10",
            root_path / "2021-09-24-11-40-24_e2e_rec_ss10_2",
            root_path / "2021-09-24-12-02-32_e2e_rec_ss10_3",
            root_path / "2021-09-24-12-21-20_e2e_rec_ss10_backwards",
            root_path / "2021-09-24-13-39-38_e2e_rec_ss11",
            root_path / "2021-09-30-13-57-00_e2e_rec_ss14",
            root_path / "2021-09-30-15-03-37_e2e_ss14_from_half_way",
            root_path / "2021-09-30-15-20-14_e2e_ss14_backwards",
            root_path / "2021-09-30-15-56-59_e2e_ss14_attempt_2",
            root_path / "2021-10-07-11-05-13_e2e_rec_ss3",
            root_path / "2021-10-07-11-44-52_e2e_rec_ss3_backwards",
            root_path / "2021-10-07-12-54-17_e2e_rec_ss4",
            root_path / "2021-10-07-13-22-35_e2e_rec_ss4_backwards",
            root_path / "2021-10-11-16-06-44_e2e_rec_ss2",
            root_path / "2021-10-11-17-10-23_e2e_rec_last_part",
            root_path / "2021-10-11-17-14-40_e2e_rec_backwards",
            root_path / "2021-10-11-17-20-12_e2e_rec_backwards",
            root_path / "2021-10-20-14-55-47_e2e_rec_vastse_ss13_17",
            root_path / "2021-10-20-13-57-51_e2e_rec_neeruti_ss19_22",
            root_path / "2021-10-20-14-15-07_e2e_rec_neeruti_ss19_22_back",
            root_path / "2021-10-25-17-31-48_e2e_rec_ss2_arula",
            root_path / "2021-10-25-17-06-34_e2e_rec_ss2_arula_back"

            # '2021-11-08-11-24-44_e2e_rec_ss12_raanitsa.bag' \
            # '2021-11-08-12-08-40_e2e_rec_ss12_raanitsa_backward.bag' \
            ]

        super().__init__(train_paths, camera=camera, **kwargs)


class NvidiaValidationDataset(NvidiaDatasetGrouped):
    # todo: remove default parameters
    def __init__(self, root_path, camera="front_wide", **kwargs):
        valid_paths = [
            root_path / "2021-05-28-15-19-48_e2e_sulaoja_20_30",
            root_path / "2021-06-07-14-20-07_e2e_rec_ss6",
            root_path / "2021-06-07-14-06-31_e2e_rec_ss6",
            root_path / "2021-06-07-14-09-18_e2e_rec_ss6",
            root_path / "2021-06-07-14-36-16_e2e_rec_ss6",
            root_path / "2021-09-24-14-03-45_e2e_rec_ss11_backwards",
            root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva",
            root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
            root_path / "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
            root_path / "2021-10-11-14-50-59_e2e_rec_vahi",
            root_path / "2021-10-14-13-08-51_e2e_rec_vahi_backwards"
        ]

        super().__init__(valid_paths, camera=camera, **kwargs)


class NvidiaElvaDataset(NvidiaDatasetGrouped):
    # todo: remove default parameters
    def __init__(self, root_path, camera="front_wide", eval_section=False, direction='both', **kwargs):

        if eval_section:
            # the 4.3km section used for evaluation drives (back and forth)
            valid_paths = []
            if direction == 'both' or direction == 'forward':
                valid_paths.append(root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk")
            if direction == 'both' or direction == 'backward':
                valid_paths.append(root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk")
        else:
            valid_paths = []
            if direction == 'both' or direction == 'forward':
                valid_paths.append(root_path / "2021-10-26-10-49-06_e2e_rec_ss20_elva")
            if direction == 'both' or direction == 'backward':
                valid_paths.append(root_path / "2021-10-26-11-08-59_e2e_rec_ss20_elva_back")

        super().__init__(valid_paths, camera=camera, **kwargs)