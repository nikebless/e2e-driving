import torch
import torch.nn as nn
from mdn.mdn import sample

class MDNInferrer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def to(self, device):
        self.model.to(device)
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = {'model.' + k: v for k,v in state_dict.items()}
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, x, return_variances=False):
        pi, sigma, mu = self.model(x)
        return sample(pi, sigma, mu, return_variances=return_variances)
