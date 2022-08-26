import torch.nn as nn
import torch

class ClassifierInferrer(nn.Module):

    def __init__(self, model, bound, n_samples):
        super().__init__()
        self.model = model
        self.model.eval()
        self.target_bins = torch.linspace(-bound, bound, n_samples, dtype=torch.float32)

    def to(self, device):
        self.model.to(device)
        self.target_bins = self.target_bins.to(device)
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = {'model.' + k: v for k,v in state_dict.items()}
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, x):
        logits = self.model(x)
        preds_indices = torch.argmax(logits, dim=1)
        preds = self.target_bins[preds_indices]
        return preds
