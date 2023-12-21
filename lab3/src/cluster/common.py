import os

from cluster.util import read_json

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

IRIS = "uci53"
DRY_BEAN = "uci602"
WINE = "uci109"
Config_Name = WINE

# Specify a dataset through a configuration file
path = os.path.join(BASE_PATH, "config", f"{Config_Name}.config.json")
config = read_json(path)
DATA_ID = config['DATA_ID']
Epoch_Times = config['Epoch_Times']
Seed = config['Seed']
Data_Type = config['Data_Type']

Params_Dir = os.path.join(BASE_PATH, "data", Data_Type, "params.pkl")
Ratio = config['Ratio']
Category_Dim = config['Category_Dim']
