from abc import abstractmethod
from typing import Dict


class ModuleLayer:
    def __init__(self, *args, **kwargs) -> None:
        self.x = None
        self.params_dict = {}

    @abstractmethod
    def init_param(self) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass

    @abstractmethod
    def load_param(self, params_dict) -> None:
        pass

    @abstractmethod
    def update_param(self, lr) -> None:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, *args):
        pass
