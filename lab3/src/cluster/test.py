import torch
from torch.distributions import MultivariateNormal

from cluster.common import *
from cluster.process.data import DATA
from cluster.train import GaussianMixtureModel
from cluster.util import sklearn_gaussian_mixture, sort


def eval(features, targets, a, mean, cov):
    for i, feature in enumerate(features):
        posterior_probs = a * MultivariateNormal(mean[i], cov[i]).log_prob(feature)
        posterior_probs /= torch.sum(posterior_probs)
        posterior_probs += 1e-20
        # 选择最大后验概率对应的分布
        max_prob, predicted_class = torch.max(posterior_probs, dim=0)
        print("后验概率 ", posterior_probs)
        print("预测的类别 ", predicted_class.item())
        print("原始类别 ", targets[i])


def true_model(data, category_dim):
    features, labels = data
    index_dict = {}

    for i, num in enumerate(labels):
        if num not in index_dict:
            index_dict[num] = [i]
        else:
            index_dict[num].append(i)

    mean_value_dict = {}
    for key, value in index_dict.items():
        mean_value_dict[key] = torch.mean(features[value], dim=0)

    label_value_dict = list(sorted(mean_value_dict.items(), key=lambda item: item[1][0]))

    final_dict = {}
    for i, item in enumerate(label_value_dict):
        final_dict[item[0]] = i

    for i, label in enumerate(labels):
        labels[i] = final_dict[label]

    new_index_list = [[] for _ in range(category_dim)]

    for i, num in enumerate(labels):
        new_index_list[int(num)].append(i)

    true_a = torch.ones([category_dim]) / category_dim
    true_mean = []
    true_cov = []
    for i in range(category_dim):
        true_mean.append(torch.mean(features[new_index_list[i]], dim=0))
        true_cov.append(torch.cov(features[new_index_list[i]].T))

    true_mean = torch.stack(true_mean, dim=0)
    true_cov = torch.stack(true_cov, dim=0)
    model = GaussianMixtureModel(category_dim)
    model.load_params_by_data(true_a, true_mean, true_cov)

    return labels, model


def our_model(category_dim):
    model = GaussianMixtureModel(category_dim)

    path = os.path.join(BASE_PATH, "data", f"uci{DATA_ID}", "params.pkl")
    model.load_params_by_file(path)
    model.sort_params()
    return model


def sklearn_model(data, category_dim):
    features, labels = data
    g_weights, g_means, g_covs = sklearn_gaussian_mixture(features, category_dim)
    g_weights, g_means, g_covs = torch.from_numpy(g_weights), torch.from_numpy(g_means), torch.from_numpy(g_covs)
    model = GaussianMixtureModel(category_dim)
    model.load_params_by_data(g_weights, g_means, g_covs)
    model.sort_params()
    return model


def kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    k = mu_p.shape[0]  # 多元正态分布的维度

    sigma_q = sigma_q.to(torch.float32)
    sigma_p = sigma_p.to(torch.float32)
    mu_p = mu_p.to(torch.float32)
    mu_q = mu_q.to(torch.float32)
    # 计算协方差矩阵的逆、行列式和差值
    sigma_q_inv = torch.inverse(sigma_q).to(torch.float32)
    det_sigma_p = torch.det(sigma_p)
    det_sigma_q = torch.det(sigma_q)
    sigma_q_inv_sigma_p = torch.matmul(sigma_q_inv, sigma_p)

    # 计算KL散度的各项
    term1 = torch.trace(sigma_q_inv_sigma_p)
    term2 = torch.matmul((mu_q - mu_p).t(), torch.matmul(sigma_q_inv, (mu_q - mu_p)))
    term3 = k - torch.log(det_sigma_q / det_sigma_p)

    # 计算KL散度
    kl = 0.5 * (term1 + term2 + term3)

    return kl


def kl(my_model: GaussianMixtureModel, sk_model: GaussianMixtureModel):
    res = 0
    for i in range(Category_Dim):
        u_p = my_model.gauss_list[i].mean
        sigma_p = my_model.gauss_list[i].cov
        u_q = sk_model.gauss_list[i].mean
        sigma_q = sk_model.gauss_list[i].cov
        res += kl_divergence(u_p, sigma_p, u_q, sigma_q).data

    return res / Category_Dim


def test_kl():
    data = DATA.get_train_data()

    _, true = true_model(data, Category_Dim)
    my_model = our_model(Category_Dim)
    sk_model = sklearn_model(data, Category_Dim)
    print(kl(my_model, true))
    print(kl(sk_model, true))


def test_acc():
    data = DATA.get_train_data()
    features, targets = data
    my_model = our_model(Category_Dim)
    sk_model = sklearn_model(data, Category_Dim)

    targets, _ = true_model(data, Category_Dim)
    acc1 = 0
    for i, feature in enumerate(features):
        res = my_model.get_hidden_value(feature)
        max_prob, predicted_class = torch.max(res, dim=0)
        if predicted_class == int(targets[i]):
            acc1 += 1
    print("ours")
    print(acc1 / features.shape[0])
    acc2 = 0
    for i, feature in enumerate(features):
        res = sk_model.get_hidden_value(feature)
        max_prob, predicted_class = torch.max(res, dim=0)
        if predicted_class == int(targets[i]):
            acc2 += 1

    print("sklearn")
    print(acc2 / features.shape[0])


if __name__ == '__main__':
    test_acc()
