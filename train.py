import argparse
import os
import sys
from pathlib import Path
import logging
import socket

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import WeightedRandomSampler

from dataloading.nvidia import NvidiaTrainDataset, NvidiaValidationDataset
import trainer as trainers


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model used for saving model and logging in W&B.'
    )

    argparser.add_argument(
        '--model-type',
        required=True,
        choices=['pilotnet', 'pilotnet-ebm', 'pilotnet-classifier', 'pilotnet-mdn'],
        help='Defines which model will be trained.'
    )

    argparser.add_argument(
        '--camera-name',
        required=False,
        default="front_wide",
        choices=['front_wide', 'left', 'right', 'all'],
        help="Camera to use for training. Only applies to 'nvidia-camera' modality."
    )

    argparser.add_argument(
        '--dataset-folder',
        default="/home/romet/data2/datasets/rally-estonia/dataset-cropped",
        help='Root path to the dataset.'
    )

    argparser.add_argument(
        '--wandb-project',
        required=False,
        help='W&B project name to use for metrics. Wandb logging is disabled when no project name is provided.'
    )

    argparser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help="Maximium number of epochs to train"
    )

    argparser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Number of epochs to train without improvement in validation loss. Used for early stopping."
    )

    argparser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help="Learning rate used in training."
    )

    argparser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-02,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--filter-blinker-turns',
        default=False,
        action='store_true',
        help='When true, turns with blinker (left or right) on will be removed from training and validation data.'
    )

    argparser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--batch-sampler',
        required=False,
        choices=['weighted', 'random'],
        default='random',
        help='Sampler used for creating batches for training.'
    )

    argparser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of workers used for data loading.'
    )

    argparser.add_argument(
        '--aug-color-prob',
        type=float,
        default=0.0,
        help='Probability of augmenting input image color by changing brightness, saturation and contrast.'
    )

    argparser.add_argument(
        '--aug-noise-prob',
        type=float,
        default=0.0,
        help='Probability of augmenting input image with noise.'
    )

    argparser.add_argument(
        '--aug-blur-prob',
        type=float,
        default=0.0,
        help='Probability of augmenting input image by blurring it.'
    )

    argparser.add_argument(
        '--loss',
        required=False,
        choices=['mse', 'mae', 'mse-weighted', 'mae-weighted', 'ce'],
        default='mae',
        help='Loss function used for training.'
    )

    argparser.add_argument(
        '--pretrained-model',
        required=False,
        help='Pretrained model used to initialize weights.'
    )

    argparser.add_argument(
        '--ebm-train-samples',
        type=int,
        default=128,
        help='Number of counterexamples used for EBM training.'
    )

    argparser.add_argument(
        '--ebm-inference-samples',
        type=int,
        default=128,
        help='Number of samples used for test-time EBM inference.'
    )

    argparser.add_argument(
        '--ebm-dfo-iters',
        type=int,
        default=0, 
        help='Number of DFO iterations used for test-time EBM inference.'
    )

    argparser.add_argument(
        '--steering-bound',
        type=float,
        default=4.5,
        help='Steering angle bound norm.'
    )
    
    argparser.add_argument(
        '--ebm-constant-samples',
        default=True,
        action='store_true',
        help='Use a constant action grid instead of random sample each time as negatives for EBM training & inference.'
    )

    argparser.add_argument(
        '--temporal-regularization',
        type=float,
        default=0.0,
        help='Temporal regularization weight for EBM training.'
    )

    argparser.add_argument(
        '--temporal-group-size',
        type=int,
        default=2,
        help='Sample items into a mini-batch in sequences of this size. Group size = 2 is necessary for temporal regularization.'
    )

    argparser.add_argument(
        '--temporal-regularization-type',
        required=False,
        choices=['crossentropy', 'l1', 'l2', 'emd', 'emd-squared', 'kldiv'],
        default='l2',
        help='Type of temporal regularization used for EBM training.'
    )

    argparser.add_argument(
        '--loss-variant',
        required=False,
        choices=['default', 'ce-proximity-aware'],
        default='default',
        help='Variant of loss used during training.'
    )

    argparser.add_argument(
        '--ce-proximity-aware-temperature',
        required=False,
        type=float,
        default=0.05, # pretty low, we want sharp peaks
        help='Temperature for gaussian distribution around truth as CE target for EBM training.'
    )

    argparser.add_argument(
        '--mdn-n-components',
        required=False,
        type=int,
        default=3,
        help='Number of components for an MDN model.'
    )

    argparser.add_argument(
        '--debug',
        action='store_true',
        help='When true, debug mode is enabled.'
    )

    argparser.add_argument(
        '--dataset-proportion',
        required=False,
        type=float,
        default=1.0,
        help='Proportion of training set recordings to use for training.'
    )

    return argparser.parse_args()


class TrainingConfig:
    def __init__(self, args):
        self.batch_sampler = args.batch_sampler
        self.batch_size = args.batch_size
        self.camera_name = args.camera_name
        self.ce_proximity_aware_temperature = args.ce_proximity_aware_temperature
        self.dataset_folder = args.dataset_folder
        self.dataset_proportion = args.dataset_proportion
        self.debug = args.debug
        self.ebm_constant_samples = args.ebm_constant_samples
        self.ebm_dfo_iters = args.ebm_dfo_iters
        self.ebm_inference_samples = args.ebm_inference_samples
        self.ebm_train_samples = args.ebm_train_samples
        self.learning_rate = args.learning_rate
        self.loss = args.loss
        self.loss_variant = args.loss_variant
        self.max_epochs = args.max_epochs
        self.mdn_n_components = args.mdn_n_components
        self.model_type = args.model_type
        self.num_workers = args.num_workers
        self.patience = args.patience
        self.steering_bound = args.steering_bound
        self.temporal_group_size = args.temporal_group_size
        self.temporal_regularization = args.temporal_regularization
        self.temporal_regularization_type = args.temporal_regularization_type
        self.wandb_project = args.wandb_project
        self.weight_decay = args.weight_decay

        log_format = "%(message)s"
        if self.debug:
            logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format)
        else:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

        self.n_input_channels = 3
        self.n_outputs = 1

        self.fps = 30
        self.pretrained_model = args.pretrained_model


def train_model(model_name, train_conf):

    print(f"Training model {model_name}, wandb_project={train_conf.wandb_project}")
    if train_conf.wandb_project:
        args = {}
        if "falcon" in socket.gethostname():
            args['settings'] = wandb.Settings(start_method='fork')

        wandb.init(project=train_conf.wandb_project, config=train_conf, **args)
    print('train_conf: ', train_conf.__dict__)

    train_loader, valid_loader = load_data(train_conf)
    if train_conf.wandb_project:
        wandb.config.update({"train_frames": len(train_loader.dataset.frames)})

    if train_conf.model_type == 'pilotnet':
        trainer = trainers.PilotNetTrainer(model_name=model_name, train_conf=train_conf)
    elif train_conf.model_type == "pilotnet-ebm":
        trainer = trainers.EBMTrainer(model_name=model_name, train_conf=train_conf)
    elif train_conf.model_type == "pilotnet-classifier":
        trainer = trainers.ClassificationTrainer(model_name=model_name, train_conf=train_conf)
    elif train_conf.model_type == "pilotnet-mdn":
        trainer = trainers.MDNTrainer(model_name=model_name, train_conf=train_conf)
    else:
        print(f"Unknown model type {train_conf.model_type}")
        sys.exit()

    trainer.train(train_loader, valid_loader, train_conf.max_epochs,
                  train_conf.patience, train_conf.fps)


def load_data(train_conf):
    print(f"Reading data from {train_conf.dataset_folder}, "
          f"camera name={train_conf.camera_name}")

    group_size = train_conf.temporal_group_size

    dataset_path = Path(train_conf.dataset_folder)
    trainset = NvidiaTrainDataset(dataset_path, camera=train_conf.camera_name, group_size=group_size, dataset_proportion=train_conf.dataset_proportion)
    validset = NvidiaValidationDataset(dataset_path, group_size=group_size)

    print(f"Training data has {len(trainset.frames)} frames")
    print(f"Validation data has {len(validset.frames)} frames")
    print(f"Creating {train_conf.num_workers} workers with batch size {train_conf.batch_size} using {train_conf.batch_sampler} sampler.")

    train_conf.batch_size = train_conf.batch_size // group_size

    if train_conf.batch_sampler == 'weighted':
        weights = calculate_weights(trainset.frames)
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=False,
                                                   sampler=sampler, num_workers=train_conf.num_workers,
                                                   pin_memory=True, persistent_workers=True, collate_fn=trainset.collate_fn)
    elif train_conf.batch_sampler == 'random':
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_conf.batch_size, shuffle=True,
                                                   num_workers=train_conf.num_workers, pin_memory=True,
                                                   persistent_workers=True, collate_fn=trainset.collate_fn)
    else:
        print(f"Unknown batch sampler {train_conf.batch_sampler}")
        sys.exit()

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=train_conf.batch_size, shuffle=False,
                                               num_workers=train_conf.num_workers, pin_memory=True,
                                               persistent_workers=True, collate_fn=trainset.collate_fn)

    return train_loader, valid_loader


def calculate_weights(df):
    # optimized_bins = np.array([df["steering_angle"].min() - 0.00001, -2.78245811e+00, -1.02905812e+00, -4.43559368e-01,
    #                            -1.64549582e-01, 6.90239861e-03, 1.69872354e-01, 4.35963640e-01,
    #                            9.63913148e-01, 2.70831896e+00, df["steering_angle"].max() + 0.00001])

    optimized_bins = np.array([df["steering_angle"].min() - 0.00001,
                               -1.121, -0.176, 0.206, 1.088,
                               df["steering_angle"].max() + 0.00001])

    bin_ranges = pd.cut(df["steering_angle"], optimized_bins, labels=np.arange(1, 6))
    df["bins"] = bin_ranges
    counts = bin_ranges.value_counts(sort=False)
    widths = np.diff(optimized_bins)
    weights = (widths / counts) * sum(counts) / sum(widths)

    weights_df = pd.DataFrame(data=weights)
    weights_df.reset_index(inplace=True)
    weights_df.columns = ["bins", "weight"]
    weights_df.set_index('bins', inplace=True)
    df = df.join(weights_df, on="bins")
    return df["weight"].to_numpy()


if __name__ == "__main__":
    print('train.py PID:', os.getpid(), 'hostname:', socket.gethostname())
    args = parse_arguments()
    train_config = TrainingConfig(args)
    train_model(args.model_name, train_config)
