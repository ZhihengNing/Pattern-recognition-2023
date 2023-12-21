import os

from network.util import read_json

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MNIST = "mnist"
IRIS = "uci53"
DRY_BEAN = "uci602"
Config_Name = IRIS
# Specify a dataset through a configuration file
path = os.path.join(BASE_PATH, "config", f"{Config_Name}.config.json")
config = read_json(path)
DATA_ID = config['DATA_ID']
Epoch_Times = config['Epoch_Times']
Iterator_Times = config['Iterator_Times']
Lr = config['Lr']
Seed = config['Seed']
Data_Type = config['Data_Type']

Params_Dir = os.path.join(BASE_PATH, "data", Data_Type, "net_params.pkl")

Use_Binary_Classify = config['Use_Binary_Classify']
Use_Stand = config['Use_Stand']
Use_Pca = config['Use_Pca']
Ratio = config['Ratio']
