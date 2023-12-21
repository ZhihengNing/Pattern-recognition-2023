import os.path

import numpy as np
from torchvision.datasets import mnist
from torchvision.transforms import transforms

from network.common import BASE_PATH


class MnistDataOriginal:
    def __init__(self):
        self.data_id = "mnist"
        self.category_dim = None
        self.target_set = None
        self.feature_dim = None

        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None

        self.feature_train_path = None
        self.target_train_path = None
        self.feature_test_path = None
        self.target_test_path = None
        self.process_data()

    def get_data(self):
        train_data_path = os.path.join(BASE_PATH, "data", "mnist", "train")
        train_data = mnist.MNIST(train_data_path,
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)
        self.train_features = np.array([item[0] for item in train_data])
        self.train_targets = np.array([item[1] for item in train_data])

        test_data_path = os.path.join(BASE_PATH, "data", "mnist", "test")
        test_data = mnist.MNIST(root=test_data_path,
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
        self.test_features = np.array([item[0] for item in test_data])
        self.test_targets = np.array([item[1] for item in test_data])

    def process_data(self):
        self.get_data()
        train_sample_dim = self.train_features.shape[0]
        self.train_features = self.train_features.reshape(train_sample_dim, -1)
        test_sample_dim = self.test_features.shape[0]
        self.test_features = self.test_features.reshape(test_sample_dim, -1)

        self.feature_dim = self.train_features.shape[1]
        self.target_set = np.unique(self.train_targets)
        self.category_dim = len(self.target_set)
