# /home/nikita/e2e-driving/_models/ibc_pure_last.pt

import logging
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import onnx
import numpy as np


from ibc import optimizers
from dataloading.nvidia import NvidiaValidationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_to_onnx(model_path, data_loader, output_path, with_optimization, n_samples, iters):
    action_bounds = data_loader.dataset.get_target_bounds()
    batch_size = data_loader.batch_size

    model = PilotnetEBMWithDFO(n_samples, iters) if with_optimization else PilotnetEBMPure()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    inputs, _, _ = iter(data_loader).next()

    frames = inputs['image'].to(device)
    sample_inputs = [frames]

    input_names = ['x']
    
    dynamic_axes = None

    if not with_optimization:
        random_actions = torch.tensor(np.random.uniform(action_bounds[0, 0], action_bounds[1, 0], size=(batch_size, n_samples, 1)), device=device, dtype=torch.float32)
        sample_inputs.append(random_actions)
        input_names.append('y')
    else: 
        dynamic_axes = {'x': {0: 'batch'}}

    output_path = Path(model_path).with_suffix('.onnx') if output_path is None else output_path

    sample_inputs.append({}) # onnx export magic

    torch.onnx.export(model, 
                      sample_inputs, 
                      output_path, 
                      input_names=input_names, 
                      dynamic_axes=dynamic_axes, 
                      do_constant_folding=False,
                      opset_version=9,
                      )
    onnx.checker.check_model(str(output_path))
    return str(output_path)


def get_loader(batch_size=1, dataset_path='/data/Bolt/dataset-new-small/summer2021', output_modality='steering_angle'):
    n_branches = 1
    num_workers = 2

    validset = NvidiaValidationDataset(Path(dataset_path), output_modality, n_branches, n_waypoints=1)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True,
                                            persistent_workers=False)
    return valid_loader


class PilotnetEBMWithDFO(nn.Module):
    """
    PilotNet with action candidates (Implicit Behavior Cloning)
    https://implicitbc.github.io/
    """

    def __init__(self, samples=1024, iters=3, output_modality='steering_angle', n_input_channels=3):
        super().__init__()

        self.output_modality = output_modality
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(1664+1, 100), # plus one for target candidate
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

        stochastic_optim_config = optimizers.DerivativeFreeConfig(
            bounds=self.get_target_bounds(),
            train_samples=0,
            inference_samples=samples,
            iters=iters,
        )
        
        self.stochastic_optimizer = optimizers.DerivativeFreeOptimizer.initialize(stochastic_optim_config)

    def get_target_bounds(self):
        return {
            "steering_angle":  torch.tensor([[-8.0], [8.0]]), # radians for ±450 degrees steering wheel rotation
            "waypoints":       NotImplemented,
        }[self.output_modality]

    def _forward(self, x, y):
        logging.debug(f'x: {x.shape} {x.dtype}')
        logging.debug(f'y: {y.shape} {y.dtype}')
        out = self.features(x)
        logging.debug(f'after features(): {out.shape} {out.dtype}')
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        logging.debug(f'fused: {fused.shape} {fused.dtype}')
        B, N, D = fused.size()
        logging.debug(f'B, N, D: {B} {N} {D}')
        fused = fused.reshape(B * N, D)
        logging.debug(f'fused (reshaped): {fused.shape} {fused.dtype}')
        out = self.regressor(fused)
        logging.debug(f'regressor output: {out.shape} {out.dtype}')
        out = out.view(B, N)
        logging.debug(f'output: {out.shape} {out.dtype}')
        return out

    def forward(self, inputs):
        return self.stochastic_optimizer.infer(inputs, self._forward)


class PilotnetEBMPure(nn.Module):
    """
    PilotNet with action candidates (Implicit Behavior Cloning)
    https://implicitbc.github.io/
    """

    def __init__(self, output_modality='steering_angle', n_input_channels=3):
        super().__init__()

        self.output_modality = output_modality
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(1664+1, 100), # plus one for target candidate
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, y):
        logging.debug(f'x: {x.shape} {x.dtype}')
        logging.debug(f'y: {y.shape} {y.dtype}')
        out = self.features(x)
        logging.debug(f'after features(): {out.shape} {out.dtype}')
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        logging.debug(f'fused: {fused.shape} {fused.dtype}')
        B, N, D = fused.size()
        logging.debug(f'B, N, D: {B} {N} {D}')
        fused = fused.reshape(B * N, D)
        logging.debug(f'fused (reshaped): {fused.shape} {fused.dtype}')
        out = self.regressor(fused)
        logging.debug(f'regressor output: {out.shape} {out.dtype}')
        out = out.view(B, N)
        logging.debug(f'output: {out.shape} {out.dtype}')
        return out


if __name__ == '__main__':
    # 1. parse arguments
    # 2. get a data loader
    # 3. convert the input model to the desired ONNX variant

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to the PyTorch model')
    parser.add_argument('--output', type=str, help='Path to the output ONNX model')
    parser.add_argument('--with-dfo', default=False, action='store_true', help='Run sampling and DFO inside the ONNX graph. If false, simply output energies for input actions.')
    parser.add_argument('--samples', default=1024, type=int, help='Number of action samples.')
    parser.add_argument('--iters', default=3, type=int, help='Number of DFO iterations. Ignored if --with_dfo is not set.')
    parser.add_argument('--bs', default=1, type=int, help='Batch size. Necessary when --with_dfo is NOT set, inference will only be available with this batch size.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print debug messages')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    dataloader = get_loader(batch_size=args.bs)
    output_path = save_model_to_onnx(args.file, dataloader, args.output, args.with_dfo, args.samples, args.iters)

    print(f'Successfuly converted to ONNX: {output_path}')