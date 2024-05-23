

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
from ManifestOnBinaryMnist import visualize_digit
from SpsdMean import SpsdMean
from tools import *
from pymanopt.manifolds import Grassmann


# General Params
random_state = 40
n_samples_each_class = 6000

# load MNIST dataset 4 and 9 digits
(X, y), (_, _) = mnist.load_data()

# Randomize data
idx = np.arange(60000)
np.random.shuffle(idx)
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


C_range = np.arange(-5, 15, 3)
C_range = np.power(2 * np.ones_like(C_range, dtype=np.float64), C_range)

gamma_range = np.arange(-15, 6, 3)
gamma_range = np.power(2 * np.ones_like(gamma_range, dtype=np.float64), gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = KFold(n_splits=4, shuffle=True, random_state=42)
best_accuracy = 0
best_features_idx = None
best_percentile_for_estimator = None
best_estimator = None

#PERCENTILES: list[int] = [30, 50, 70, 90, 95, 99, 100]
PERCENTILES: list[float] = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                            2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
                            3.9, 4, 5]
for cur_percentile in PERCENTILES:
    print(f"Percentile {cur_percentile}")
    # Compute kernels
    kernels = construct_kernel(x_train, y_train, cur_percentile)

    # Compute minimal rank in our kernels
    min_rank = 200#min([np.linalg.matrix_rank(K, tol=5e-3) for K in kernels])
    print(f"min_rank: {min_rank}")
    # Compute M from kernels
    kernels = np.stack(kernels, axis=0)
    kernels = cp.array(kernels)
    M, mG, mP, UU, TT = SpsdMean(kernels, r=min_rank)
    #M, mG, mP, UU, TT = cp.load('M.cpy.npy'), cp.load('mG.cpy.npy'), cp.load('mP.cpy.npy'), cp.load('UU.cpy.npy'), cp.load('TT.cpy.npy')


    N = kernels.shape[0]

    G_, _, _, _, _ = spsd_geodesics(M, kernels, 1, min_rank)
    logP_ = log_map(mP[None, :], TT)
    D_ = symmetrize(G_ @ logP_ @ cp.swapaxes(G_, -1, -2))
    eigvals, eigvecs = cp.linalg.eigh(D_)
    eigvecs_square = cp.square(eigvecs)
    eigvals_abs = cp.expand_dims(cp.abs(eigvals), axis=1)
    r = eigvals_abs * eigvecs_square
    r = r.sum(axis=2)
    score = cp.max(r, axis=0)
    idx = cp.argsort(score, axis=0)[::-1]
    idx_top50 = idx[:50]
    top50_features = [(x // 28, x % 28) for x in idx_top50]

    #visualize_digit(x_train.reshape(3000, -1), 100, top50_features)

    # Define ranges for C and gamma parameters of SVM


    x_fs = x_train.reshape(x_train.shape[0], -1)[:, idx_top50.get()]
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, scoring="accuracy", verbose=2, n_jobs=4)
    y_fs = y_train.astype(np.int8)
    grid.fit(x_fs, y_fs)
    x_test_fs = x_test.reshape(x_test.shape[0], -1)[:, idx_top50.get()]
    y_test_fs = y_test.astype(np.int8)
    if grid.best_score_ > best_accuracy:
        best_accuracy = grid.best_score_
        best_percentile_for_estimator = cur_percentile
        best_estimator = grid.best_estimator_
        best_features_idx = idx_top50
    print(
        f"Percentile: {cur_percentile}\t The best parameters are {grid.best_params_} with a score of %{grid.best_score_:.5f}")
    #print(f"The best parameters are {grid.best_params_} with a score of %{grid.best_score_:.5f}")

# X_test_fs = X_test[:, best_features_idx]
# y_test_fs = (y_test == DIGIT_1).astype(np.int8)
# print(
#     f"Test: {best_percentile_for_estimator}\t Test Accuracy: {accuracy_score(y_test_fs, best_estimator.predict(X_test_fs)):.5f}")

print()