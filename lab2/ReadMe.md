# Neural Network

## Introduction

A neural network based on Tensor implementation, including fully connect layer, relu layer, softmax layer, and other network layers, was selected to achieve multi classification tasks using uci dataset and minist dataset.

## Project structure

![image-20231105165835219](.\ReadMe.assets\image-20231105165835219.png)

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

open the **common.py** and **config folder**,**config_ name** is the prefix of the selected dataset hyperparameter file name,**data_id** in common.py is the dataset number, **use_use_binary_classify** is whether or not to use binary classification, **use_stand** is whether or not to normalize, **use_pca** is whether or not to use pca, **pca** is the reduced dimensionality, and **ratio** is the percentage of the training set.

### Training

Open **common.py** to set **config_ Name** selects the dataset, sets the hyperparameters in the **"config/{ConfigName}. config. json"** file, and finally runs **train.py**.

### Test

Open **common.py** to set **config_ Name** selects the dataset, sets the hyperparameters in the **"config/{ConfigName}. config. json"** file, and finally runs **test.py**.

ps: check if the **"data/{Config_Name}/net_params.pkl"** file exists

