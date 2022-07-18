from __future__ import annotations

import dataclasses
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    lr_scheduler_step: int = 100
    lr_scheduler_gamma: float = 0.99


@dataclasses.dataclass
class DerivativeFreeConfig:
    noise_scale: float = 0.33
    noise_shrink: float = 0.5
    iters: int = 3
    train_samples: int = 256
    inference_samples: int = 2 ** 14
    bounds: torch.Tensor = torch.tensor([[-8.], [8.]])


class DerivativeFreeOptimizer(nn.Module):
    """An iterative derivative-free optimizer from the IBC paper. Could be overkill for our purposes."""

    def __init__(self, ebm: nn.Module, config: DerivativeFreeConfig):
        super().__init__()

        self.ebm = ebm
        self.noise_scale = config.noise_scale
        self.noise_shrink = config.noise_shrink
        self.iters = config.iters
        self.train_samples = config.train_samples
        self.inference_samples = config.inference_samples
        self.bounds = config.bounds

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = {'ebm.' + k: v for k,v in state_dict.items()}
        super().load_state_dict(state_dict)

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Drawing samples from the uniform random distribution."""
        bounds = self.bounds
        lower = bounds[0, :]
        upper = bounds[1, :]
        size = (num_samples, bounds.shape[1])
        samples = (lower - upper) * torch.rand(size, device=bounds.device) + upper
        return samples

    def sample(self, batch_size: int) -> torch.Tensor:
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    def to(self, device: torch.device) -> DerivativeFreeOptimizer:
        self.ebm.to(device)
        self.bounds = self.bounds.to(device)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        ebm = self.ebm
        noise_scale = self.noise_scale
        bounds = self.bounds

        logging.debug(f'x: {x.shape}')
        samples = self._sample(x.size(0) * self.inference_samples)
        logging.debug(f'samples: {samples.shape}')
        samples = samples.reshape(x.size(0), self.inference_samples, -1)
        logging.debug(f'samples reshaped: {samples.shape}')

        for i in range(self.iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0))[..., None], idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]
