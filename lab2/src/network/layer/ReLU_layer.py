from typing import Dict

import torch

from network.layer.base_layer import ModuleLayer


class ReLULayer(ModuleLayer):
    def __init__(self):
        super().__init__()
        print("relu layer init...")

    def init_param(self):
        pass

    def load_param(self, params_dict) -> None:
        pass

    def get_params(self) -> Dict:
        pass

    def update_param(self, lr):
        pass

    def forward(self, x):
        self.x = x
        x = torch.maximum(x, torch.tensor([0.0]))
        return x

    def backward(self, top_grad):
        bottom_grad = top_grad
        bottom_grad[self.x < 0] = 0
        return bottom_grad
