# Neural Network

## Introduction

A neural network based on Tensor implementation, including fully connect layer, relu layer, softmax layer, and other network layers, was selected to achieve multi classification tasks using uci dataset and minist dataset.

## Project structure

![image-20231130000536650](.\ReadMe.assets\image-20231130000536650.png)

Config is the network hyperparameter setting, data is the dataset, layer is the network layer set, and process is the data preprocessing.

## Install 

Run:

```
pip install ucimlrepo
pip install torch
pip install numpy
pip install matplotlib
```

## Start

### Basic settings

open the **common.py** and **config folder**,**config_ name** is the prefix of the selected dataset hyperparameter file name,**data_id** in common.py is the dataset number, **ratio** is the percentage of the training set.

### Training

Open **common.py** to set **config_ Name** selects the dataset, sets the hyperparameters in the **"config/{ConfigName}. config. json"** file, and finally runs **train.py** or **better_train.py**（**better_train.py** is is based on matrix multiplication optimization and has a very fast speed)



### Test

Open **common.py** to set **config_ Name** selects the dataset, sets the hyperparameters in the **"config/{ConfigName}. config. json"** file, and finally runs **test.py**.(Please note whether to choose to test KL divergence or accuracy)

ps: check if the **"data/{Config_Name}/net_params.pkl"** file exists

