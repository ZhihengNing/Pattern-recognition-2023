import torch
from torch.distributions import MultivariateNormal


class Gauss:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.shape = self.cov.shape[0]
        self.distribution = MultivariateNormal(mean, cov)

    def get_value(self, y):
        res = self.distribution.log_prob(y).exp()
        return res

    def set_attr(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.distribution = MultivariateNormal(mean, cov)


if __name__ == '__main__':
    mean = torch.tensor([1.0, 1.0])
    cov = torch.eye(2)
    gauss = Gauss(mean, cov)
    y = torch.tensor([1, 2])
    print(gauss.get_value(y))
