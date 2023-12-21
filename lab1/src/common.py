import torch

from src.data import CategoricalData

Data_ID = 602
nepoch = 1
alpha = 0.05
iterator_times = 1000
DATA = CategoricalData(data_id=Data_ID, seed=1023, use_binary_classify=True, use_stand=True, use_pca=False, ratio=0.8)


# W ,B are Final arguments of model training
# choose_indices are dimension which will be shown on the result graph
def args(data_id):
    if data_id == 53:
        w = torch.tensor([[-1.4298, 0.3018, -1.0921, -1.8600],
                          [0.8330, -1.4580, -0.0600, 1.0787]], dtype=torch.float64,
                         requires_grad=True)
        b = torch.tensor([-0.5685, 0.2540], dtype=torch.float64, requires_grad=True)
        choose_indices = [0, 1]

    if data_id == 602:
        w = torch.tensor([[0.5096, 0.7614, -0.9598, -0.1769, 0.7528, -0.4486, -2.9262, -0.1143,
                           -1.1600, -0.8770, -1.2030, 0.5643, 0.7357, -0.5942, -1.0887, 1.5416],
                          [-0.1619, 0.6849, -0.3813, -0.8989, 1.7844, -0.9866, 1.7149, 1.0646,
                           0.4542, -0.2212, -0.4238, -0.9241, -1.7113, -0.8212, 0.3565, 0.1087]],
                         dtype=torch.float64, requires_grad=True)
        b = torch.tensor([0.7783, -1.6064], dtype=torch.float64, requires_grad=True)
        choose_indices = [-4, 6]
    return w, b, choose_indices


W, B, Choose_Indices = args(Data_ID)
