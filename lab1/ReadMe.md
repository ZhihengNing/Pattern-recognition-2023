# Classification problems

## Introduction

Classification prediction based on uci dataset, modeled using logistic regression, which allows for binary and multiple classification

## Project structure

```
src
|   common.py
|   data.py
|   draw.py
|   logisticloss.py
|   test.py
|   train.py
|
+---53
|   |   feature_names_path.txt
|   |   target_set_path.txt
|   |
|   +---original
|   |       features.txt
|   |       targets.txt
|   |
|   +---test
|   |       features.txt
|   |       targets.txt
|   |
|   \---train
|           features.txt
|           targets.txt
|
\---602
    |   feature_names_path.txt
    |   target_set_path.txt
    |
    +---original
    |       features.txt
    |       targets.txt
    |
    +---test
    |       features.txt
    |       targets.txt
    |
    \---train
            features.txt
            targets.txt
```

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

**data_id** in common.py is the dataset number, **use_use_binary_classify** is whether or not to use binary classification, **use_stand** is whether or not to normalize, **use_pca** is whether or not to use pca, **pca** is the reduced dimensionality, and **ratio** is the percentage of the training set.

### Training

Set **data_id** and run **train.py**.

**ps**:train.py first training will categorize the data and automatically import it to file storage.

### Test

Set data_id and model parameters **W** and **B**, and run **test.py**.

### **Plotting**

Set **data_id** and plot dimensions **choose_indices**, **choose_indices** are dimensions which will be shown on the result graph