import random

import numpy as np
from matplotlib import pyplot as plt

from src.common import DATA, W, B, Choose_Indices
from src.data import CategoricalDataset


def draw_plot(choose_indices, features, targets, W, B, feature_names):
    features = features[:, choose_indices]
    unique_values = np.unique(targets)
    split_indices = [targets == unique_value for unique_value in unique_values]
    split_features = [features[split_index, :] for split_index in split_indices]
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', "k"]
    color_list = random.sample(color_list, len(unique_values))
    for index, unique_value in enumerate(unique_values):
        plt.scatter(split_features[index][:, 0], split_features[index][:, 1],
                    label=unique_values[index], color=color_list[index], marker='o', s=25)
    plt.xlabel(feature_names[choose_indices[0]])
    plt.ylabel(feature_names[choose_indices[1]])
    # only dim=2 =>Binary classification can draw the hyperplane
    if W.shape[0] == 2 and B.shape[0] == 2:
        W = (W[0] - W[1]).detach().numpy()
        W = W[choose_indices]
        x = features[:, 0]
        c = (B[0] - B[1]).detach().numpy()
        y = (-W[0] * x - c) / W[1]
        plt.plot(x, y, label=f'{W[0]:.2f}x+{W[1]:.2f}y+{c:.2f}=0', color='pink')
    plt.title("result graph")
    plt.legend()
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    feature_names = DATA.feature_names
    original_dataset = CategoricalDataset(DATA.feature_original_path, DATA.target_original_path)
    features, targets = original_dataset.features, original_dataset.targets
    if DATA.use_pca:
        features = DATA.pca.inverse_transform(features)

    p = (W[0] - W[1]).detach().numpy()
    print(p)
    draw_plot(Choose_Indices, features, targets, W, B, feature_names)
