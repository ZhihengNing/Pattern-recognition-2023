from network.common import *
from network.process.mnistdata import MnistDataOriginal
from network.process.ucidata import UciDataOriginal

if Data_Type.startswith("uci"):
    DATA = UciDataOriginal(data_id=DATA_ID,
                           seed=Seed,
                           use_binary_classify=Use_Binary_Classify,
                           use_stand=Use_Stand,
                           use_pca=Use_Pca,
                           ratio=Ratio)
elif Data_Type == "mnist":
    DATA = MnistDataOriginal()
