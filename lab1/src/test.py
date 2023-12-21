import torch

from src.common import DATA, W, B
from src.data import CategoricalDataset
from src.logisticloss import LogisticLoss


def green(x):
    return '\033[92m' + x + '\033[0m'


def blue(x):
    return '\033[94m' + x + '\033[0m'


def red(x):
    return '\033[91m' + x + '\033[0m'


if __name__ == '__main__':
    target_set = DATA.target_set
    test_dataset = CategoricalDataset(DATA.feature_test_path, DATA.target_test_path)
    features, targets = test_dataset.features, test_dataset.targets

    features = torch.from_numpy(features)
    sample_dim, feature_dim = features.size()

    logistic = LogisticLoss(W, B, features, targets, target_set)
    accept = 0
    for i in range(sample_dim):
        feature, target = features[i], targets[i]
        probability, loss = logistic.sample_loss(feature, target)
        print(
            f"sampler {i} true value is {green(targets[i])}，"
            f"predict value is {blue(target_set[torch.argmax(probability)])}")
        if targets[i] == target_set[torch.argmax(probability)]:
            accept += 1
        else:
            print(f"{red('predict wrong!')}")
        print(f"probability:{probability.data}")
        print(f"loss:{loss}")
        print("_____________________________")
    print(f"acc: {accept / sample_dim * 100:.2f}%")
