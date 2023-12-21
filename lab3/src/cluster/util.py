import json

import torch
from sklearn.mixture import GaussianMixture
from torch.distributions import MultivariateNormal


def generate_positive_definite_matrix(size):
    A = torch.randn(size, size, dtype=torch.float32)
    # 计算 A*A^T 以确保 A 是正定的
    positive_definite_matrix = torch.mm(A, A.t())
    return positive_definite_matrix


def generate_gauss_distribution(feature_dim):
    mean = torch.rand([feature_dim])
    cov = generate_positive_definite_matrix(feature_dim)
    return MultivariateNormal(mean, cov)


def generate_data(category_dim: int, sample_dim: int, feature_dim: int):
    res = []
    for i in range(category_dim):
        distribution = generate_gauss_distribution(feature_dim)
        separate_sample_dim = int(sample_dim / category_dim)
        res.append(distribution.sample((separate_sample_dim,)))
    final_data = torch.cat(res, dim=0)
    return final_data


def sklearn_gaussian_mixture(data, category_dim: int):
    g = GaussianMixture(n_components=category_dim)
    g.fit(data)
    return g.weights_, g.means_, g.covariances_


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def sort(matrix):
    # 获取按照第一行元素排序的索引
    sorted_indices = torch.argsort(matrix[:, 0])
    return sorted_indices


if __name__ == '__main__':
    A = torch.tensor([[4.0, 2.0, 7.0],
                      [1.0, 5.0, 3.0],
                      [9.0, 6.0, 8.0]])
    print(sort(A))
