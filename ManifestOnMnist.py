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


X = np.concatenate((X[(y == 0), :][:n_samples_each_class, :, :],
                    X[(y == 1), :][:n_samples_each_class, :, :],
                    X[(y == 2), :][:n_samples_each_class, :, :],
                    X[(y == 3), :][:n_samples_each_class, :, :],
                    X[(y == 4), :][:n_samples_each_class, :, :],
                    X[(y == 5), :][:n_samples_each_class, :, :],
                    X[(y == 6), :][:n_samples_each_class, :, :],
                    X[(y == 7), :][:n_samples_each_class, :, :],
                    X[(y == 8), :][:n_samples_each_class, :, :],
                    X[(y == 9), :][:n_samples_each_class, :, :]))

y = np.concatenate((y[(y == 0)][:n_samples_each_class],
                    y[(y == 1)][:n_samples_each_class],
                    y[(y == 2)][:n_samples_each_class],
                    y[(y == 3)][:n_samples_each_class],
                    y[(y == 4)][:n_samples_each_class],
                    y[(y == 5)][:n_samples_each_class],
                    y[(y == 6)][:n_samples_each_class],
                    y[(y == 7)][:n_samples_each_class],
                    y[(y == 8)][:n_samples_each_class],
                    y[(y == 9)][:n_samples_each_class]))

X = X.reshape(X.shape[0], -1) / 255
k = construct_kernels(X, y, 1)


#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)
