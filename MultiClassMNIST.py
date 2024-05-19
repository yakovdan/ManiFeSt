

import cupy as cp
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
from Manifest2 import ManiFeSt2, construct_kernel, spsd_geodesics
from SpsdMean import SpsdMean
from tools import *
from pymanopt.manifolds import Grassmann


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


# Compute kernels
kernels = construct_kernel(x_train, y_train)

# Compute minimal rank in our kernels
r = min([np.linalg.matrix_rank(K, tol=5e-3) for K in kernels])
print(f"r: {r}")
# Compute M from kernels
kernels = np.stack(kernels, axis=0)
kernels = cp.array(kernels)
#M, mG, mP, UU, TT = SpsdMean(kernels, r=r)
M, mG, mP, UU, TT = cp.load('M.cpy.npy'), cp.load('mG.cpy.npy'), cp.load('mP.cpy.npy'), cp.load('UU.cpy.npy'), cp.load('TT.cpy.npy')


N = kernels.shape[0]

G_, _, _, _, _ = spsd_geodesics(M, kernels, 1, r)
logP_ = log_map(mP[None, :], TT)
D_ = symmetrize(G_ @ logP_ @ cp.swapaxes(G_, -1, -2))
eigvals, eigvecs = cp.linalg.eigh(D_)
eigvecs_square = cp.square(eigvecs)
eigvals_abs = cp.expand_dims(cp.abs(eigvals), axis=1)
r = eigvals_abs * eigvecs_square
r = r.sum(axis=2)
score = cp.sum(r, axis=0)
idx = cp.argsort(score, axis=0)[::-1]
idx_top50 = idx[:50]
top50_features = [(x // 28, x % 28) for x in idx_top50]
from ManifestOnBinaryMnist import visualize_digit
visualize_digit(x_train.reshape(3000, -1), 100, top50_features)

print()