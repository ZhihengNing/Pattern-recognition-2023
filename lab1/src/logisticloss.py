import torch


def indicator(target: any, need_target: any) -> int:
    if not type(target) == type(need_target):
        raise Exception("类型错误")
    if target == need_target:
        return 1
    return 0


class LogisticLoss:
    # features, [sample_dim, feature_dim]
    # targets, [sample_dim]
    # w, [category_dim, feature_dim]
    # b, [category_dim]
    def __init__(self, w, b, features, targets, target_set: [any]):
        self.w = w
        self.b = b
        self.features = features
        self.targets = targets
        self.target_set = target_set
        self.sample_dim, self.feature_dim = features.size()
        self.category_dim = self.w.size()[0]

    def loss(self) -> float:
        total_loss = 0.0
        for i in range(self.sample_dim):
            _, loss = self.sample_loss(self.features[i], self.targets[i])
            total_loss += loss
        total_loss = total_loss / self.sample_dim
        return total_loss

    def sample_loss(self, feature, need_target: any) -> (torch.Tensor, float):
        numerator = torch.zeros([self.category_dim])
        for i in range(self.category_dim):
            numerator[i] = torch.exp(torch.dot(self.w[i], feature) + self.b[i])
        denominator = torch.sum(numerator)
        loss = 0.0
        for i, target in enumerate(self.target_set):
            loss -= indicator(target, need_target) * torch.log(numerator[i] / denominator)
        probability = numerator.clone() / denominator
        return probability, loss

