import torch

from network.common import *
from network.data import DATA
from network.dataset import CategoricalDataset
from network.mlp import *
from network.util import blue, green

if __name__ == '__main__':
    target_set = DATA.target_set
    train_dataset = CategoricalDataset(feature_path=DATA.feature_train_path,
                                       target_path=DATA.target_train_path,
                                       features=DATA.train_features,
                                       targets=DATA.train_targets,
                                       seed=Seed)
    feature_dim = train_dataset.feature_dim
    if Data_Type == "uci53":
        mlp = Uci53MLP(feature_dim, target_set)
    elif Data_Type == "uci602":
        mlp = Uci602MLP(feature_dim, target_set)
    elif Data_Type == "mnist":
        mlp = MnistMLP(feature_dim, target_set)
    else:
        raise Exception("no this type")
    batch_size = int(train_dataset.sample_dim / Iterator_Times)
    assert batch_size > 0

    for epoch in range(Epoch_Times):
        train_dataset.shuffle()
        for iterator in range(Iterator_Times):
            features, targets = train_dataset.batch_samples(iterator, batch_size)
            if type(features) is not torch.Tensor:
                features = torch.from_numpy(features).type(torch.float64)
            mlp.forward(features)
            loss = mlp.get_loss(targets)
            mlp.backward()
            mlp.update_param(Lr)
            if iterator % 10 == 0:
                print(f"{blue('epoch')}: {epoch}, "
                      f"{green('iterator')}: {iterator}, "
                      f"loss: {loss.data}")

    mlp.save_params(Params_Dir)
