import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch

from dataloading.nvidia import NvidiaDataset, Normalize

from torchvision import transforms


def create_steering_angle_error_plot(model, trainer, dataset_name):
    root_path = Path("/home/romet/data2/datasets/rally-estonia/dataset-cropped")
    tr = transforms.Compose([Normalize()])
    dataset = NvidiaDataset([root_path / dataset_name], transform=tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
    pred_steering_angles = trainer.predict(model, dataloader)

    f, (ax) = plt.subplots(2, 1, figsize=(50, 25))
    true_steering_angle = dataset.frames.steering_angle
    ax[0].plot(true_steering_angle, color="green")
    ax[0].plot(pred_steering_angles, color="red")

    ax[0].plot(dataset.frames.turn_signal, linewidth=3, color="gold")
    ax[0].plot(dataset.frames.vehicle_speed, color="darkorange")
    ax[0].legend(['true_steering_angle', 'pred_steering_angle', 'turn_signal', 'vehicle_speed'])
    ax[0].set_title(dataset_name + " | steering angle")

    steering_error = np.abs(pred_steering_angles - true_steering_angle)
    ax[1].plot(steering_error, color="darkred")
    ax[1].set_title(dataset_name + " | steering error")
    ax[1].legend(['steering error'])


