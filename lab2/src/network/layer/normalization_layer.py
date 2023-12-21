from typing import Dict

import torch

from network.layer.base_layer import ModuleLayer


class NormalizationLayer(ModuleLayer):
    def __init__(self):
        super().__init__()
        print("normalization layer init...")

    def init_param(self) -> None:
        pass

    def get_params(self) -> Dict:
        pass

    def load_param(self, params_dict) -> None:
        pass

    def update_param(self, lr) -> None:
        pass

    def forward(self, x):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / std
        return x

    def backward(self, *args):
        pass
