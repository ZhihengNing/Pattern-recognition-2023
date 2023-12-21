import pickle

import torch
from sklearn.cluster import KMeans

from cluster.common import *
from cluster.process.data import DATA
from cluster.util import *
from gauss import Gauss


class GaussianMixtureModel:
    def __init__(self, category_dim: int):
        self.category_dim = category_dim

        self.gauss_list: [Gauss] = []
        self.a = None
        self.mean = None
        self.cov = None
        self.params_dict = None

    def init_params(self, data, use_k_means: False):
        print("model init begin ... ")
        _, feature_dim = data.shape
        self.a = torch.ones([self.category_dim]) / self.category_dim
        if use_k_means:
            g = KMeans(self.category_dim)
            g.fit(data)
            self.mean = torch.from_numpy(g.cluster_centers_)
        else:
            self.mean = torch.randn([self.category_dim, feature_dim])
        self.cov = (generate_positive_definite_matrix(feature_dim)
                    .expand(self.category_dim, feature_dim, feature_dim))

        for i in range(self.category_dim):
            self.gauss_list.append(Gauss(self.mean[i], self.cov[i]))
        print("init mean:", self.mean)
        print("model init end ...")

    def get_hidden_value(self, y):
        gauss_values = torch.zeros([self.category_dim])
        for k, gauss in enumerate(self.gauss_list):
            gauss_values[k] = self.a[k] * gauss.get_value(y)
            # 防止出现0的情况
            gauss_values[k] += 1e-20

        gauss_values = gauss_values / torch.sum(gauss_values)
        return gauss_values

    def fit(self, data):
        self.init_params(data, True)
        sample_dim, feature_dim = data.shape
        for epoch in range(Epoch_Times):
            a_numerator = torch.zeros([self.category_dim, sample_dim])
            for i, y in enumerate(data):
                a_numerator[:, i] = self.get_hidden_value(y)

            denominator = torch.sum(a_numerator, dim=1)
            self.a = denominator / sample_dim

            mean_numerator = torch.zeros([self.category_dim, sample_dim, feature_dim])


            for k in range(self.category_dim):
                for i, y in enumerate(data):
                    mean_numerator[k][i] = a_numerator[k][i] * y
            self.mean = torch.sum(mean_numerator, dim=1) / denominator.view(-1, 1)

            cov_numerator = torch.zeros([self.category_dim, sample_dim, feature_dim, feature_dim])
            for k in range(self.category_dim):
                for i, y in enumerate(data):
                    vector = y - self.mean[k]
                    cov_numerator[k][i] = a_numerator[k][i] * torch.mul(vector.view(-1, 1), vector)
            self.cov = torch.sum(cov_numerator, dim=1) / denominator.view(-1, 1, 1)

            for k in range(self.category_dim):
                self.gauss_list[k].set_attr(self.mean[k], self.cov[k])

            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1}...")
                print("mean", self.mean)
                print("\n")

        print("guassian mixture finish")

    def sort_params(self):
        sorted_indices = sort(self.mean)
        self.a = self.a[sorted_indices]
        self.mean = self.mean[sorted_indices, :]
        self.cov = self.cov[sorted_indices, :]
        self.gauss_list.clear()
        for i in range(self.category_dim):
            self.gauss_list.append(Gauss(self.mean[i], self.cov[i]))
        return sorted_indices

    def save_params(self, params_path):
        self.params_dict = {
            "a": self.a,
            "mean": self.mean,
            "cov": self.cov
        }

        with open(params_path, 'wb') as file:
            pickle.dump(self.params_dict, file)

    def load_params_by_file(self, params_path):
        with open(params_path, "rb") as file:
            self.params_dict = pickle.load(file)

        self.a = self.params_dict["a"]
        self.mean = self.params_dict["mean"]
        self.cov = self.params_dict["cov"]
        for i in range(self.category_dim):
            self.gauss_list.append(Gauss(self.mean[i], self.cov[i]))

    def load_params_by_data(self, a, mean, cov):
        self.a = a
        self.mean = mean
        self.cov = cov
        for i in range(self.category_dim):
            self.gauss_list.append(Gauss(self.mean[i], self.cov[i]))


if __name__ == '__main__':
    category_dim = Category_Dim
    feature, target = DATA.get_train_data()
    # feature = generate_data(category_dim, 1000, 4)

    g_weights, g_means, g_covs = sklearn_gaussian_mixture(feature, category_dim)
    print(g_weights)
    print(g_means)
    print(g_covs)

    model = GaussianMixtureModel(category_dim)
    model.fit(feature)
    path = os.path.join(BASE_PATH, "data", f"uci{DATA_ID}", "params.pkl")
    model.save_params(path)
    print(model.mean)
    # print(model.cov)
