from typing import Dict

import torch

from network.layer.base_layer import ModuleLayer


class FullyConnectLayer(ModuleLayer):

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensor_one = None
        self.w = None
        self.b = None
        self.dw = None
        self.db = None
        super().__init__()
        self.init_param()
        print("fully connect layer init...")

    def init_param(self) -> None:
        self.w = torch.normal(0, 0.01, [self.input_dim, self.output_dim], dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros([1, self.output_dim], dtype=torch.float64, requires_grad=True)

    def load_param(self, params_dict) -> None:
        self.params_dict = params_dict
        self.w = params_dict["w"]
        self.b = params_dict["b"]

    def get_params(self) -> Dict:
        self.params_dict = {
            "w": self.w,
            "b": self.b
        }
        return self.params_dict

    def update_param(self, lr) -> None:
        self.w = self.w - lr * self.dw
        self.b = self.b - lr * self.db

    def forward(self, x):
        self.x = x
        sample_dim = x.shape[0]
        self.tensor_one = torch.ones([sample_dim, 1], dtype=torch.float64)
        x = torch.mm(x, self.w) + torch.mm(self.tensor_one, self.b)
        return x

    def backward(self, top_gard):
        self.dw = torch.mm(self.x.transpose(0, 1), top_gard)
        self.db = torch.mm(self.tensor_one.transpose(0, 1), top_gard)
        bottom_gard = torch.mm(top_gard, self.w.transpose(0, 1))
        return bottom_gard
