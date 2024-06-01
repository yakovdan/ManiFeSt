# -*- coding: utf-8 -*-
import itertools

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
from Manifest2 import ManiFeSt2

# %%  ManiFeSt Score

DIGIT_0: int = 4
DIGIT_1: int = 9
#PERCENTILES: list[int] = [5, 10, 30, 50, 70, 90, 95]
PERCENTILES: list[int] = [30, 50, 70, 90, 95]
# General Params
random_state = 40
n_samples_each_class = 6000


def load_mnist_dataset():
    (X1, y1), (X2, y2) = mnist.load_data()
    X, y = np.concatenate((X1, X2)), np.concatenate((y1, y2))
    return X, y


def extract_digits(X_arr, y_arr, digit_0: int, digit_1: int) -> tuple[np.ndarray, np.ndarray]:
    x_selected_digits = np.concatenate((X_arr[(y_arr == digit_0), :][:n_samples_each_class, :, :],
                                        X_arr[(y_arr == digit_1), :][:n_samples_each_class, :, :]))
    y_selected_digits = np.concatenate(
        (y_arr[(y_arr == digit_0)][:n_samples_each_class], y_arr[(y_arr == digit_1)][:n_samples_each_class]))
    x_selected_digits = (x_selected_digits.reshape(x_selected_digits.shape[0], -1) / 255)
    return x_selected_digits, y_selected_digits


def visualize_digit(digit_array: np.ndarray, digit_idx: int, feature_coords, resize_factor: int = 1) -> None:
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    digit = digit_array.reshape(28, 28).get()
    #digit = cv2.resize(digit, dsize=(28*resize_factor, 28*resize_factor), interpolation=cv2.INTER_CUBIC)
    ax.imshow(digit)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    for yy, xx in feature_coords[:20]:
        circ = Circle((resize_factor * xx, resize_factor * yy), 0.5, color='yellow', fill=False)
        ax.add_patch(circ)
    for yy, xx in feature_coords[20:]:
        circ = Circle((resize_factor * xx, resize_factor * yy), 0.5, color='red', fill=False)
        ax.add_patch(circ)

    plt.show()


if __name__ == "__main__":
    use_spsd: bool = True # False - use SPD form  - default is SPSD, MNIST is SPSD since there are blank pixels
    X, y = load_mnist_dataset()
    X, y = extract_digits(X, y, DIGIT_0, DIGIT_1)
    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, stratify=y, random_state=random_state)

    # Define ranges for C and gamma parameters of SVM
    C_range = np.arange(-5, 15, 3)
    C_range = np.power(2 * np.ones_like(C_range, dtype=np.float64), C_range)

    gamma_range = np.arange(-15, 6, 3)
    gamma_range = np.power(2 * np.ones_like(gamma_range, dtype=np.float64), gamma_range)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    best_accuracy = 0
    best_features_idx = None
    best_percentile_for_estimator = None
    best_estimator = None
    for cur_percentile in PERCENTILES:
        score, idx, eig_vecs = ManiFeSt2(X_train, y_train, percentile=cur_percentile,
                                            use_spsd=use_spsd)
        eigVecD, eigValD, eigVecM, eigValM = eig_vecs
        sorted_score = score[idx]
        score_top20, score_top50 = sorted_score[:20], sorted_score[:50]
        idx_top20, idx_top50 = idx[:20], idx[:50]
        feature_coords_top20 = [(x // 28, x % 28) for x in idx_top20]
        feature_coords_top50 = [(x // 28, x % 28) for x in idx_top50]

        X_fs = X_train[:, idx_top50]
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, scoring="accuracy", verbose=0, n_jobs=4)
        y_fs = (y_train == DIGIT_1).astype(np.int8)
        grid.fit(X_fs, y_fs)
        X_test_fs = X_test[:, idx_top50]
        y_test_fs = (y_test == DIGIT_1).astype(np.int8)
        if grid.best_score_ > best_accuracy:
            best_accuracy = grid.best_score_
            best_percentile_for_estimator = cur_percentile
            best_estimator = grid.best_estimator_
            best_features_idx = idx_top50
        print(f"Percentile: {cur_percentile}\t The best parameters are {grid.best_params_} with a score of %{grid.best_score_:.5f}")

    X_test_fs = X_test[:, best_features_idx]
    y_test_fs = (y_test == DIGIT_1).astype(np.int8)
    print(f"Test: {best_percentile_for_estimator}\t Test Accuracy: {accuracy_score(y_test_fs, best_estimator.predict(X_test_fs)):.5f}")
    # The RBF kernel scale is typically set to the median of the Euclidean distances up to some scalar
    # defiend by kernel_scale_factor ,  default value 1

    kernel_scale_factor = 1
    score, idx, eig_vecs = ManiFeSt(X_train, y_train, kernel_scale_factor=kernel_scale_factor,
                                    use_spsd=use_spsd)  # use_spsd=use_spsd
    score1, idx1, eig_vecs1 = ManiFeSt2(X_train, y_train, percentile=50, use_spsd=use_spsd)
    assert np.allclose(score, score1)
    assert np.allclose(idx, idx1)
    assert len(eig_vecs) == len(eig_vecs1)
    for i in range(len(eig_vecs)):
        assert np.allclose(eig_vecs[i], eig_vecs1[i])

    sorted_score = score[idx]
    score_top20, score_top50 = sorted_score[:20], sorted_score[:50]
    idx_top20, idx_top50 = idx[:20], idx[:50]
    feature_coords_top20 = [(x // 28, x % 28) for x in idx_top20]
    feature_coords_top50 = [(x // 28, x % 28) for x in idx_top50]

    x_train_DIGIT_1 = X_train[np.where(y_train == DIGIT_1)]
    x_train_DIGIT_0 = X_train[np.where(y_train == DIGIT_0)]
    eigVecD, eigValD, eigVecM, eigValM = eig_vecs

    visualize_digit(x_train_DIGIT_1, 30, feature_coords_top50)


    X_test_fs = X_test[:, idx_top50]
    print(accuracy_score(y_test_fs, grid.predict(X_test_fs)))
