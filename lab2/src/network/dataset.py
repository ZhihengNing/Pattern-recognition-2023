import numpy as np
import torch


class CategoricalDataset:
    def __init__(self, feature_path: str = None,
                 target_path: str = None,
                 features=None,
                 targets=None,
                 seed=0):
        if feature_path is not None:
            self.feature_path = feature_path
            self.target_path = target_path
            self.features = np.loadtxt(self.feature_path).astype("float64")
            self.targets = np.loadtxt(self.target_path, dtype=str)
        else:
            self.features = features
            self.targets = targets
        self.sample_dim, self.feature_dim = self.features.shape
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def shuffle(self):
        data_indices = np.arange(self.features.shape[0])
        np.random.shuffle(data_indices)
        self.features = self.features[data_indices]
        self.targets = self.targets[data_indices]

    # random choose train sample
    def samples(self) -> (torch.Tensor, torch.Tensor):
        data_size = self.features.shape[0]
        data_indices = self.rng.choice(data_size, size=np.random.randint(1, data_size), replace=False)
        return self.features[data_indices], self.targets[data_indices]

    def batch_samples(self, index: int, batch_size: int):
        begin = index * batch_size
        end = min((index + 1) * batch_size, self.sample_dim)
        return self.features[begin:end], self.targets[begin: end]


