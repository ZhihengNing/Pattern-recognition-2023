import torch

from network.common import *
from network.data import DATA
from network.dataset import CategoricalDataset
from network.mlp import *
from network.util import green, blue, red

if __name__ == '__main__':
    target_set = DATA.target_set
    test_dataset = CategoricalDataset(feature_path=DATA.feature_test_path,
                                      target_path=DATA.target_test_path,
                                      features=DATA.test_features,
                                      targets=DATA.test_targets)
    features, targets = test_dataset.features, test_dataset.targets

    if type(features) is not torch.Tensor:
        features = torch.from_numpy(features).type(torch.float64)

    sample_dim, feature_dim = features.size()
    if Data_Type == "uci53":
        mlp = Uci53MLP(feature_dim, target_set)
    elif Data_Type == "uci602":
        mlp = Uci602MLP(feature_dim, target_set)
    elif Data_Type == "mnist":
        mlp = MnistMLP(feature_dim, target_set)
    else:
        raise Exception("no this type")

    mlp.load_param(Params_Dir)
    accept = 0
    for i in range(sample_dim):
        feature, target = features[i], targets[i]
        feature = feature.reshape(1, -1)
        probability = mlp.forward(feature)
        print(
            f"sampler {i} true value is {green(targets[i])}，"
            f"predict value is {blue(target_set[torch.argmax(probability)])}")
        if targets[i] == target_set[torch.argmax(probability)]:
            accept += 1
        else:
            print(f"{red('predict wrong!')}")
        print(f"probability:{probability.data}")
        # print(f"loss:{loss}")
        print("_____________________________")
    print(f"acc: {accept / sample_dim * 100:.2f}%")
