import torch

from src.common import iterator_times, nepoch, alpha, DATA
from src.data import CategoricalDataset
from src.logisticloss import LogisticLoss

if __name__ == '__main__':
    target_set = DATA.target_set
    category_dim = DATA.category_dim  # category count
    train_dataset = CategoricalDataset(DATA.feature_train_path, DATA.target_train_path, seed=1026)
    feature_dim = train_dataset.feature_dim
    print(f"feature dim:{feature_dim}")
    for epoch in range(nepoch):
        # W = torch.randn([category_dim, feature_dim], dtype=torch.float64, requires_grad=True)
        # B = torch.randn([category_dim], dtype=torch.float64, requires_grad=True)
        # sample_features = torch.randn([sample_dim, feature_dim], dtype=torch.float)
        # outputs = torch.randint(0, sample_dim, [sample_dim])
        W = torch.normal(0, 1, size=[category_dim, feature_dim], dtype=torch.float64, requires_grad=True)
        B = torch.normal(0, 1, size=[category_dim], dtype=torch.float64, requires_grad=True)
        for iterator in range(iterator_times):
            features, targets = train_dataset.samples()
            features = torch.from_numpy(features)
            logistic = LogisticLoss(W, B, features, targets, target_set)
            loss = logistic.loss()
            # The weights and loss function will be output every ten times
            if iterator % 10 == 0:
                print(f"loss:{loss}")
                print(W)
                print(B)
                print("————————————————————————————")
            loss.backward()
            W.data = W.data - alpha * W.grad
            B.data = B.data - alpha * B.grad
            W.grad.zero_()
            B.grad.zero_()
        print(W)
        print(B)
