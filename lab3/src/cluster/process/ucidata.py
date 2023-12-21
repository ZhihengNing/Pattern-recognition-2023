import os.path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

from cluster.common import BASE_PATH


class UciDataOriginal:
    def __init__(self,
                 data_id: int,
                 use_binary_classify=False,
                 use_stand=False,
                 use_pca=False,
                 pca_dim=0,
                 seed=0,
                 ratio=0.9):

        self.data_id = data_id
        self.use_binary_classify = use_binary_classify
        self.use_stand = use_stand
        if self.use_stand:
            self.mean = None
            self.std = None
        self.use_pca = use_pca
        if self.use_pca:
            self.pca = None
            self.pca_dim = pca_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.ratio = ratio
        self.data_path = os.path.join(BASE_PATH, "data", f"uci{self.data_id}")

        self.original_path = os.path.join(self.data_path, "original")
        self.train_path = os.path.join(self.data_path, "train")
        self.test_path = os.path.join(self.data_path, "test")
        self.feature_original_path = os.path.join(self.original_path, "features.txt")
        self.target_original_path = os.path.join(self.original_path, "targets.txt")
        self.feature_train_path = os.path.join(self.train_path, "features.txt")
        self.target_train_path = os.path.join(self.train_path, "targets.txt")
        self.feature_test_path = os.path.join(self.test_path, "features.txt")
        self.target_test_path = os.path.join(self.test_path, "targets.txt")

        self.target_set_path = os.path.join(self.data_path, "target_set_path.txt")
        self.feature_names_path = os.path.join(self.data_path, "feature_names_path.txt")

        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None

        if not os.path.exists(self.data_path):
            os.makedirs(self.original_path)
            os.makedirs(self.train_path)
            os.makedirs(self.test_path)
            self.process_data()
        self.target_set = np.loadtxt(self.target_set_path, delimiter=",", dtype=str)
        self.feature_names = np.loadtxt(self.feature_names_path, delimiter=",", dtype=str)
        self.category_dim = len(self.target_set)

    def get_data(self):
        if self.data_id is not None:
            categorical = fetch_ucirepo(id=self.data_id)
            features = categorical.data.features
            targets = categorical.data.targets
            variables = categorical.variables
            return features, targets, variables

    def get_train_data(self):
        features = np.loadtxt(self.feature_train_path).astype("float64")
        targets = np.loadtxt(self.target_train_path, dtype=str)
        if type(features) is not torch.Tensor:
            features = torch.from_numpy(features).type(torch.float64)
        return features, targets

    def get_test_data(self):
        features = np.loadtxt(self.feature_test_path).astype("float64")
        targets = np.loadtxt(self.target_test_path, dtype=str)
        if type(features) is not torch.Tensor:
            features = torch.from_numpy(features).type(torch.float64)
        return features, targets

    def binary_classification(self, features, targets, target_set, is_specify=True):
        # choose the first one as the positive sample
        admitted_target = target_set[0]
        admitted_indices = np.where(targets == admitted_target)[0]
        if is_specify:
            another_admitted_target = target_set[1]
            another_admitted_indices = np.where(targets == another_admitted_target)[0]
            all_admitted_indices = np.concatenate((admitted_indices, another_admitted_indices))
            targets = targets[all_admitted_indices]
            features = features[all_admitted_indices]
            target_set = [admitted_target, another_admitted_target]
        else:
            unadmitted_indices = np.setdiff1d(np.arange(targets.shape[0]), admitted_indices)
            none_admitted_target = "no-" + admitted_target
            targets[unadmitted_indices] = none_admitted_target
            target_set = [admitted_target, none_admitted_target]
        return features, targets, target_set

    def process_data(self):
        features, targets, variables = self.get_data()
        # one hot encode
        features = pd.get_dummies(features, dtype=float).values
        targets = targets.values.reshape(-1)

        if variables is not None:
            feature_names = variables.values[0:-1, 0]
            np.savetxt(self.feature_names_path, feature_names, fmt="%s")
        # Target category set
        target_set = np.unique(targets)
        if self.use_binary_classify:
            features, targets, target_set = self.binary_classification(features, targets, target_set, is_specify=True)
        if self.use_stand:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0)
            features = (features - self.mean) / self.std
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_dim)
            features = self.pca.fit_transform(features)
        np.savetxt(self.target_set_path, target_set, fmt="%s")
        np.savetxt(self.feature_original_path, features, delimiter=" ")
        np.savetxt(self.target_original_path, targets, delimiter=" ", fmt="%s")

        self.divide_data(features, targets, target_set)

    def divide_data(self, features, targets, target_set):
        if self.ratio < 0 or self.ratio > 1:
            raise Exception("ratio arguments error!")
        data_size = features.shape[0]
        # sample train balance
        if self.use_binary_classify:
            indices = np.arange(data_size)
            admitted_indices = indices[targets == target_set[0]]
            unadmitted_indices = indices[targets == target_set[1]]
            count = min(len(admitted_indices), len(unadmitted_indices))
            selected_admitted_indices = self.rng.choice(admitted_indices, size=count, replace=False)
            selected_unadmitted_indices = self.rng.choice(unadmitted_indices, size=count, replace=False)
            selected_indices = np.concatenate((selected_admitted_indices, selected_unadmitted_indices))
            features = features[selected_indices]
            targets = targets[selected_indices]
            data_size = len(selected_indices)

        data_indices = np.arange(data_size)
        threshold = int(data_size * self.ratio)
        train_indices = self.rng.choice(data_size, size=threshold, replace=False)

        train_features = features[train_indices]
        train_targets = targets[train_indices]
        np.savetxt(self.feature_train_path, train_features, delimiter=" ")
        np.savetxt(self.target_train_path, train_targets, delimiter=" ", fmt="%s")

        test_indices = np.setdiff1d(data_indices, train_indices)
        test_features = features[test_indices]
        test_targets = targets[test_indices]
        np.savetxt(self.feature_test_path, test_features, delimiter=" ")
        np.savetxt(self.target_test_path, test_targets, delimiter=" ", fmt="%s")
