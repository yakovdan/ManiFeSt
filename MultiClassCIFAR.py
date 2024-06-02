
from oct2py import octave
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

import torchvision
import torchvision.transforms as transforms

STORE_MODE = True
DISABLE_SAVE = True
VISUALIZE = True
grayscale = True

octave.addpath('/home/romariozarik/Desktop/SPSD')
# General Params
random_state = 40
n_samples_each_class = 6000
np.random.seed(random_state)
# load MNIST dataset 4 and 9 digits
(X, y), (_, _) = mnist.load_data()

if grayscale:
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert PIL Image to Tensor
        transforms.Grayscale(num_output_channels=1) # Normalize image data
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL Image to Tensor
    ])

train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=False, download=True, transform=transform)

x_train = [train_dataset[i][0] for i in range(len(train_dataset))]
y_train = [train_dataset[i][1] for i in range(len(train_dataset))]

x_test = [test_dataset[i][0] for i in range(len(test_dataset))]
y_test = [test_dataset[i][1] for i in range(len(test_dataset))]

X = np.zeros((len(x_train) + len(x_test), 32, 32)) if grayscale else np.zeros((len(x_train) + len(x_test), 3, 32, 32))
y = np.zeros((len(x_train) + len(x_test), ))

for i in range(len(x_train) + len(x_test)):
    X[i] = x_train[i] if i < len(x_train) else x_test[i - len(x_train)]
    y[i] = y_train[i] if i < len(y_train) else y_test[i - len(x_train)]

C_range = np.arange(-5, 16, 3)
C_range = np.power(2 * np.ones_like(C_range, dtype=np.float64), C_range)

gamma_range = np.arange(-15, 6, 3)
gamma_range = np.power(2 * np.ones_like(gamma_range, dtype=np.float64), gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
accuracies = []
train_accuracies = []
for exp_idx in range(10):

    # Randomize data
    idx = np.arange(60000)
    np.random.shuffle(idx)
    x_shuff, y_shuff = X[idx], y[idx]
    X, y = x_shuff, y_shuff

    # Choose 300 "random" examples from our dataset
    x_train, x_test = np.zeros(shape=(3000, 32, 32), dtype=X.dtype), np.zeros(shape=(57000, 32, 32), dtype=X.dtype)
    y_train, y_test = np.zeros(shape=(3000,), dtype=y.dtype), np.zeros(shape=(57000,), dtype=y.dtype)

    # reshape dataset
    position = 0
    for i in range(10):
        elem_count = (y == i).sum()
        test_count = elem_count - 300
        x_train[300*i: 300*(i+1)] = X[y == i][:300]
        y_train[300*i: 300*(i+1)] = i
        x_test[position: position+test_count] = X[y == i][300:]
        y_test[position: position+test_count] = i
        position += test_count


    # folds for SVM
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    best_accuracy = 0
    best_features_idx = None
    best_percentile_for_estimator = None
    best_estimator = None
    best_params = None
    PERCENTILES: list[int] = [50, 70, 90, 95, 99]
    for cur_percentile in PERCENTILES:
        print(f"Percentile {cur_percentile}")
        # Compute kernels
        if STORE_MODE:
            kernels = construct_kernel(x_train, y_train, cur_percentile)

            # Compute minimal rank in our kernels
            min_rank = min([np.linalg.matrix_rank(K, tol=5e-3) for K in kernels])
            print(f"min_rank: {min_rank}")
            # Compute M from kernels
            kernels_oct = np.stack(kernels, axis=2)
            kernels_cp = cp.array(np.stack(kernels))
            octave.eval("""function [mC, mG, mP, UU, TT] = roundtrip(y, req_rank)
            %
            l = size(y, 3);
            CC{l} = [];
            for i=1:l
                CC{i} = y(:, :, i);
            end
            size(CC);
            size(CC{1});
            [mC, mG, mP, UU, TT] = SpsdMean(CC, req_rank);
            """)

            M, mG, mP, UU, TT = octave.roundtrip(kernels_oct, min_rank, nout=5)
            M = cp.array(M)
            mG = cp.array(mG)
            mP = cp.array(mP)
            UUU = np.concatenate([UU.item(i)[None, :] for i in range(UU.size)], axis=0)
            TTT = np.concatenate([TT.item(i)[None, :] for i in range(TT.size)], axis=0)
            UU = cp.array(np.copy(UUU))
            TT = cp.array(np.copy(TTT))

            #M, mG, mP, UU, TT = SpsdMean(kernels, r=min_rank)
            #M, mG, mP, UU, TT = cp.load('M.cpy.npy'), cp.load('mG.cpy.npy'), cp.load('mP.cpy.npy'), cp.load('UU.cpy.npy'), cp.load('TT.cpy.npy')


            N = kernels_cp.shape[0]

            G_, _, _, _, _ = spsd_geodesics(M, kernels_cp, 1, min_rank)
            logP_ = log_map(mP[None, :], TT)
            D_ = symmetrize(G_ @ logP_ @ cp.swapaxes(G_, -1, -2))
            eigvals, eigvecs = cp.linalg.eigh(D_)
            eigvecs_square = cp.square(eigvecs)
            eigvals_abs = cp.expand_dims(cp.abs(eigvals), axis=1)
            r = eigvals_abs * eigvecs_square
            r = r.sum(axis=2)
            score = cp.max(r, axis=0)
            idx = cp.argsort(score, axis=0)[::-1]

            if not DISABLE_SAVE:
                cp.save(f'M_{cur_percentile}_1', M)
                cp.save(f'mG_{cur_percentile}_1', mG)
                cp.save(f'mP_{cur_percentile}_1', mP)
                cp.save(f'UU_{cur_percentile}_1', UU)
                cp.save(f'TT_{cur_percentile}_1', TT)
                cp.save(f'G_{cur_percentile}_1', G_)
                cp.save(f'D_{cur_percentile}_1', D_)
                cp.save(f'score_{cur_percentile}_1', score)
                cp.save(f'idx_{cur_percentile}_1', idx)
                cp.save(f'kernels_{cur_percentile}_1', kernels_cp)

        else:
            M = cp.load(f'M_{cur_percentile}_1.npy')
            mG = cp.load(f'mG_{cur_percentile}_1.npy')
            mP = cp.load(f'mP_{cur_percentile}_1.npy')
            UU = cp.load(f'UU_{cur_percentile}_1.npy')
            TT = cp.load(f'TT_{cur_percentile}_1.npy')
            G_ = cp.load(f'G_{cur_percentile}_1.npy')
            D_ = cp.load(f'D_{cur_percentile}_1.npy')
            score = cp.load(f'score_{cur_percentile}_1.npy')
            idx = cp.load(f'idx_{cur_percentile}_1.npy')

            eigvals, eigvecs = cp.linalg.eigh(D_)
            eigvecs_square = cp.square(eigvecs)
            eigvals_abs = cp.expand_dims(cp.abs(eigvals), axis=1)

        idx_top50 = idx[:50]  # cp.random.choice(cp.arange(784), size=50, replace=False)#idx[:50]
        top50_features = [(x // 32, x % 32) for x in idx_top50]
        score_viz = cp.abs(score).get().reshape((1, 32, 32))
        score_viz_sq = np.square(score_viz)
        if VISUALIZE:
            visualize_digit(score_viz, 0, top50_features, some_title=f"score_{cur_percentile}_rank_{min_rank}", mode =0)
            visualize_digit(score_viz_sq, 0, top50_features, some_title=f"score_sq_{cur_percentile}_rank_{min_rank}", mode=0)
            for class_idx in range(10):
                eig_vec_for_viz = cp.abs(eigvecs[class_idx, :, -1]).get().reshape((1, 32, 32))
                visualize_digit(eig_vec_for_viz, 0, top50_features, some_title=f"{class_idx}_{cur_percentile}_rank_{min_rank}")
                eig_vec_for_viz = np.square(eig_vec_for_viz)
                visualize_digit(eig_vec_for_viz, 0, top50_features, some_title=f"{class_idx}_sq_{cur_percentile}_rank_{min_rank}")


        x_fs = x_train.reshape(x_train.shape[0], -1)[:, idx_top50.get()]
        x_fs = x_fs / x_fs.max()
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, scoring="accuracy", verbose=2, n_jobs=4)
        y_fs = y_train.astype(np.int8)
        grid.fit(x_fs, y_fs)
        if grid.best_score_ > best_accuracy:
            best_accuracy = grid.best_score_
            best_percentile_for_estimator = cur_percentile
            best_estimator = grid.best_estimator_
            best_features_idx = idx_top50
            best_params = grid.best_params_
            cp.save('bestM', M)
            cp.save('bestD', D_)
            print("Found new best SVM")
        print(
            f"Percentile: {cur_percentile}\t The best parameters are {grid.best_params_} with a score of %{ 100* grid.best_score_:.5f}")
        train_accuracies.append(grid.best_score_)
        #print(f"The best parameters are {grid.best_params_} with a score of %{grid.best_score_:.5f}")

    x_test_fs = x_test.reshape(x_test.shape[0], -1)[:, best_features_idx.get()]
    y_test_fs = y_test.astype(np.int8)
    x_test_fs = x_test_fs / x_test_fs.max()
    y_target = best_estimator.predict(x_test_fs)
    accuracy = (y_test_fs == y_target).sum() / y_target.shape[0]
    print(f"Final accuracy: {accuracy}")
    accuracies.append(accuracy)
    print(f"train acc: {train_accuracies}, mean: {np.mean(np.array(train_accuracies))}, std: {np.std(np.array(train_accuracies))}")
    # print(
    #     f"Test: {best_percentile_for_estimator}\t Test Accuracy: {accuracy_score(y_test_fs, best_estimator.predict(X_test_fs)):.5f}")

print(f"test acc: {accuracies}, mean: {np.mean(np.array(accuracies))}, std: {np.std(np.array(accuracies))}")