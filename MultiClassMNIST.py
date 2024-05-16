

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from ManiFeSt import ManiFeSt
from Manifest2 import ManiFeSt2, construct_kernel


# General Params
random_state = 40
n_samples_each_class = 6000

# load MNIST dataset 4 and 9 digits
(X, y), (_, _) = mnist.load_data()

# Randomize data
idx = np.random.shuffle(np.arange(60000))
x_shuff, y_shuff = X[idx], y[idx]
X, y = x_shuff, y_shuff

# Choose 300 "random" examples from our dataset
x_train, x_test= np.zeros(shape=(3000, 28, 28), dtype=X.dtype), np.zeros(shape=(57000, 28, 28), dtype=X.dtype)
y_train, y_test = np.zeros(shape=(3000,), dtype=y.dtype), np.zeros(shape=(57000,), dtype=y.dtype)
position = 0
for i in range(10):
    elem_count = (y == i).sum()
    test_count = elem_count - 300
    x_train[300*i: 300*(i+1)] = X[y == i][:300]
    y_train[300*i: 300*(i+1)] = i
    x_test[position: position+test_count] = X[y == i][300:]
    y_test[position: position+test_count] = i
    position += test_count

kernels = construct_kernel(x_train, y_train)
print()