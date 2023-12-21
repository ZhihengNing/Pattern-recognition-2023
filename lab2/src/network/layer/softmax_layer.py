from typing import Dict

import torch

from network.layer.base_layer import ModuleLayer


def indicator(target: any, need_target: any) -> int:
    if not type(target) == type(need_target):
        raise Exception("类型错误")
    if target == need_target:
        return 1
    return 0


def softmax(x):
    x = torch.exp(x)
    x = x / torch.sum(x, dim=1, keepdim=True)
    return x


#
class SoftmaxLayer(ModuleLayer):
    def __init__(self, target_set):
        self.outputs = None
        self.true_targets_encode = None
        self.target_set = target_set
        super().__init__()
        print("softmax layer init...")

    def init_param(self) -> None:
        pass

    def load_param(self, params_dict) -> None:
        pass

    def get_params(self) -> Dict:
        pass

    def update_param(self, lr):
        pass

    def forward(self, x):
        self.x = x
        self.outputs = softmax(x)
        return self.outputs

    def get_loss(self, true_targets):
        assert true_targets.shape[0] == self.x.shape[0]
        self.true_targets_encode = self.encode_targets(true_targets)
        total_loss = 0.0
        for i, true_target in enumerate(true_targets):
            # loss = self.sample_loss(i, true_target)
            loss = self.sample_loss(i)
            total_loss += loss
        total_loss = total_loss / true_targets.shape[0]
        return total_loss

    def sample_loss(self, index):
        return -torch.dot(torch.log(self.outputs[index]), self.true_targets_encode[index])

    def sample_loss2(self, index, true_target):
        loss = 0.0
        for i, target in enumerate(self.target_set):
            loss -= indicator(target, true_target) * torch.log(self.outputs[index, i])
        return loss

    def encode_targets(self, true_targets):
        result = torch.zeros_like(self.outputs)
        index_dict = {value: index for index, value in enumerate(self.target_set)}
        # 使用字典来获取每个元素在targets_set中的索引
        indices = [index_dict[element] for element in true_targets]
        sample_count = self.outputs.shape[0]
        result[torch.arange(sample_count), indices] = 1.0
        return result

    def backward(self):
        sample_count = self.outputs.shape[0]
        bottom_grad = (self.outputs - self.true_targets_encode) / sample_count
        return bottom_grad
