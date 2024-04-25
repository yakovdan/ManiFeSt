import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from ManiFeSt import ManiFeSt
from scipy import linalg
random_state: int = 40
n_samples_each_class: int = 2000


def construct_kernels(X, y, kernel_scale_factor) -> list[np.ndarray]:
    labels = sorted(list(set(y)))
    kernels = []
    for label in labels:
        x_i = X[np.where(y == label)]
        #x_2 = X[np.where(y == label[1])]
        Ki_dis = euclidean_distances(np.transpose(x_i))
        #K2_dis = euclidean_distances(np.transpose(x_2))
        epsilon_i = kernel_scale_factor * np.median(Ki_dis[~np.eye(Ki_dis.shape[0], dtype=bool)])
        #epsilon2 = kernel_scale_factor * np.median(K2_dis[~np.eye(K2_dis.shape[0], dtype=bool)])

        Ki = np.exp(-(Ki_dis ** 2) / (2 * epsilon_i ** 2))
        kernels.append(Ki)
        #K2 = np.exp(-(K2_dis ** 2) / (2 * epsilon2 ** 2))

    return kernels

# Load train and test, concat, reshuffle and split
(X1, y1), (X2, y2) = mnist.load_data()
(X, y) = (np.concatenate((X1, X2)), np.concatenate((y1, y2)))


X = [X[(y == i), :][:n_samples_each_class, :, :] for i in range(10)]
y = [y[(y == i)][:n_samples_each_class] for i in range(10)]
score_list = []
idx_list = []
eig_vecs_list = []

for i in range(10):
    for j in range(i+1, 10):
        X_cat = np.concatenate((X[i], X[j]))
        y_cat = np.concatenate((y[i], y[j]))
        X_cat = (X_cat.reshape(X_cat.shape[0], -1) / 255)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_cat, y_cat, test_size=0.25, stratify=y_cat, random_state=random_state)

        # ManiFeSt Score
        use_spsd = True  # False - use SPD form  - default is SPSD, MNIST is SPSD since there are blank pixels
        kernel_scale_factor = 1  # The RBF kernel scale is typically set to the median of the Euclidean distances up to some scalar defiend by kernel_scale_factor ,  default value 1
        score, idx, eig_vecs = ManiFeSt(X_train, y_train, kernel_scale_factor=kernel_scale_factor,
                                        use_spsd=use_spsd)  # use_spsd=use_spsd

        score_list.append(score)
        idx_list.append(idx)
        eig_vecs_list.append(eig_vecs)