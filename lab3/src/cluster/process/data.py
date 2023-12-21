from cluster.common import *
from cluster.process.ucidata import UciDataOriginal

DATA = UciDataOriginal(data_id=DATA_ID,
                       seed=Seed,
                       ratio=Ratio)

if __name__ == '__main__':
    train_data = DATA.get_train_data()
    # res = DATA.get_data()
    print(train_data)
