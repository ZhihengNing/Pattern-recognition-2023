import pickle
from abc import abstractmethod
from typing import Dict

from network.layer.ReLU_layer import ReLULayer
from network.layer.base_layer import ModuleLayer
from network.layer.fully_connet_layer import FullyConnectLayer
from network.layer.softmax_layer import SoftmaxLayer


class MLP(ModuleLayer):
    def __init__(self):
        super().__init__()
        self.layer_name_list: [ModuleLayer] = []
        attributes = dir(self)
        for attribute_name in attributes:
            attribute = getattr(self, attribute_name)
            if issubclass(type(attribute), ModuleLayer):
                self.layer_name_list.append(attribute_name)
        self.init_param()

    def init_param(self) -> None:
        for layer_name in self.layer_name_list:
            layer = getattr(self, layer_name)
            layer.init_param()

    def get_params(self) -> Dict:
        for layer_name in self.layer_name_list:
            layer = getattr(self, layer_name)
            self.params_dict[layer_name] = layer.get_params()
        return self.params_dict

    def save_params(self, params_file) -> None:
        with open(params_file, 'wb') as file:
            pickle.dump(self.get_params(), file)

    def load_param(self, params_file) -> None:
        with open(params_file, "rb") as file:
            self.params_dict = pickle.load(file)
            for layer_name in self.layer_name_list:
                layer = getattr(self, layer_name)
                param_dict = self.params_dict[layer_name]
                layer.load_param(param_dict)

    def update_param(self, lr) -> None:
        for layer_name in self.layer_name_list:
            layer = getattr(self, layer_name)
            layer.update_param(lr)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_loss(self, true_targets):
        pass

    @abstractmethod
    def backward(self):
        pass


class MnistMLP(MLP):
    def __init__(self, input_dim: int, target_set: []):
        self.input_dim = input_dim
        self.target_set = target_set
        self.output_dim = len(self.target_set)
        dim1, dim2, dim3 = 256, 128, 64
        self.fc1 = FullyConnectLayer(self.input_dim, dim1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectLayer(dim1, dim2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectLayer(dim2, dim3)
        self.relu3 = ReLULayer()
        self.fc4 = FullyConnectLayer(dim3, self.output_dim)
        self.softmax = SoftmaxLayer(target_set)
        super().__init__()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        x = self.relu3.forward(x)
        x = self.fc4.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self):
        bt_grad = self.softmax.backward()
        bt_grad = self.fc4.backward(bt_grad)
        bt_grad = self.relu3.backward(bt_grad)
        bt_grad = self.fc3.backward(bt_grad)
        bt_grad = self.relu2.backward(bt_grad)
        bt_grad = self.fc2.backward(bt_grad)
        bt_grad = self.relu1.backward(bt_grad)
        bt_grad = self.fc1.backward(bt_grad)

    def get_loss(self, true_targets):
        return self.softmax.get_loss(true_targets)


class Uci53MLP(MLP):
    def __init__(self, input_dim: int, target_set: []):
        self.input_dim = input_dim
        self.target_set = target_set
        self.output_dim = len(self.target_set)
        dim1 = 6
        self.fc1 = FullyConnectLayer(self.input_dim, dim1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectLayer(dim1, self.output_dim)
        self.softmax = SoftmaxLayer(target_set)
        super().__init__()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.softmax.forward(x)
        return x

    def get_loss(self, true_targets):
        return self.softmax.get_loss(true_targets)

    def backward(self):
        bt_grad = self.softmax.backward()
        bt_grad = self.fc2.backward(bt_grad)
        bt_grad = self.relu1.backward(bt_grad)
        bt_grad = self.fc1.backward(bt_grad)


class Uci602MLP(MLP):
    def __init__(self, input_dim: int, target_set: []):
        self.input_dim = input_dim
        self.target_set = target_set
        self.output_dim = len(self.target_set)
        dim1, dim2 = 64, 32
        self.fc1 = FullyConnectLayer(self.input_dim, self.output_dim)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectLayer(dim1, dim2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectLayer(dim2, self.output_dim)
        self.softmax = SoftmaxLayer(target_set)
        super().__init__()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        x = self.softmax.forward(x)
        return x

    def get_loss(self, true_targets):
        return self.softmax.get_loss(true_targets)

    def backward(self):
        bt_grad = self.softmax.backward()
        bt_grad = self.fc3.backward(bt_grad)
        bt_grad = self.relu2.backward(bt_grad)
        bt_grad = self.fc2.backward(bt_grad)
        bt_grad = self.relu1.backward(bt_grad)
        bt_grad = self.fc1.backward(bt_grad)
