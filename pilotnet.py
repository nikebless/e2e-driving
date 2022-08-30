import torch
import torch.nn as nn
import logging
from mdn.mdn import MDN

class PilotNet(nn.Module):
    """
    Network from 'End to End Learning for Self-Driving Cars' paper:
    https://arxiv.org/abs/1604.07316

    Conditonal control is concatenated with input features to each policy branchy
    """

    def __init__(self):
        super(PilotNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
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
            nn.Linear(1664, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.features(x)
        outs = self.regressor(x)
        return outs


class PilotnetEBM(nn.Module):
    """
    PilotNet with action candidates (Implicit Behavior Cloning)
    https://implicitbc.github.io/
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
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

        logging.debug(f'model features: {self.features}')

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

        logging.debug(f'model regressor: {self.regressor}')

    @property
    def device(self):
        return next(self.parameters()).device

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

class PilotnetClassifier(nn.Module):
    """
    Convolutional layer from 'End to End Learning for Self-Driving Cars' paper:
    https://arxiv.org/abs/1604.07316

    The rest transformed from a regressor into a discretized classification model.
    """

    def __init__(self, n_target_bins=256):
        super().__init__()

        self.n_target_bins = n_target_bins

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
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

        self.classifier = nn.Sequential(
            nn.Linear(1664, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, n_target_bins),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PilotnetMDN(nn.Module):
    """
    Convolutional layer from 'End to End Learning for Self-Driving Cars' paper:
    https://arxiv.org/abs/1604.07316

    The rest transformed from a regressor into a discretized classification model.
    """

    def __init__(self, n_gaussians=3):
        super().__init__()

        self.n_gaussians = n_gaussians

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
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
            nn.Linear(1664, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.Tanh(),
            MDN(10, 1, n_gaussians),
        )

    def forward(self, x):
        x = self.features(x)
        outs = self.regressor(x)
        return outs
