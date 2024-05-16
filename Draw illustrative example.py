# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

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

# General Params
random_state = 40
n_samples_each_class = 6000

# load MNIST dataset 4 and 9 digits
(X1, y1), (X2, y2) = mnist.load_data()
(X, y) = (np.concatenate((X1, X2)), np.concatenate((y1, y2)))

# extract 4 and 9 digits
X = np.concatenate((X[(y == 4), :][:n_samples_each_class, :, :], X[(y == 9), :][:n_samples_each_class, :, :]))
y = np.concatenate((y[(y == 4)][:n_samples_each_class], y[(y == 9)][:n_samples_each_class]))
X = (X.reshape(X.shape[0], -1) / 255)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)

# ManiFeSt Score
use_spsd = True  # False - use SPD form  - default is SPSD, MNIST is SPSD since there are blank pixels
kernel_scale_factor = 1  # The RBF kernel scale is typically set to the median of the Euclidean distances up to some scalar defiend by kernel_scale_factor ,  default value 1
score, idx, eig_vecs = ManiFeSt(X_train, y_train, kernel_scale_factor=kernel_scale_factor,
                                use_spsd=use_spsd)  # use_spsd=use_spsd
# score1, idx1, eig_vecs1 = ManiFeSt2(X_train, y_train,
#                                     use_spsd=use_spsd)  # use_spsd=use_spsd
# assert np.allclose(score, score1)
# assert np.allclose(idx, idx1)
# assert len(eig_vecs) == len(eig_vecs1)
# for i in range(len(eig_vecs)):
#     assert np.allclose(eig_vecs[i], eig_vecs1[i])

sorted_score = score[idx]
score_top20, score_top50 = sorted_score[:20], sorted_score[:50]
idx_top20, idx_top50 = idx[:20], idx[:50]
feature_coords_top20 = [(x // 28, x % 28) for x in idx_top20]
feature_coords_top50 = [(x // 28, x % 28) for x in idx_top50]




# %% Plot Score
label = list(set(y_train))
x_train_9 = X_train[np.where(y_train == 9)]
x_train_4 = X_train[np.where(y_train == 4)]
(eigVecD, eigValD, eigVecM, eigValM) = eig_vecs

fig,ax = plt.subplots(1)
ax.set_aspect('equal')

# Show the image
digit_9_img = x_train_4[50, :].reshape(28, 28)
digit_9_img = cv2.resize(digit_9_img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
ax.imshow(digit_9_img, cmap='gray')

# Now, loop through coord arrays, and create a circle at each x,y pair
for yy, xx in feature_coords_top50:
    circ = Circle((3 * xx, 3 * yy), 1, color='red', fill=False)
    ax.add_patch(circ)

plt.show()
X_fs = X_train[:, idx_top50]
C_range = np.arange(-5, 15, 3)
C_range = np.power(2*np.ones_like(C_range, dtype=np.float32), C_range)

gamma_range = np.arange(-15, 6, 3)
gamma_range = np.power(2*np.ones_like(gamma_range, dtype=np.float32), gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = KFold(n_splits=10, shuffle=True, random_state=42)
grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, scoring="accuracy", verbose=3, n_jobs=4)
y_fs = (y_train == 9).astype(np.int8)
y_test_fs = (y_test == 9).astype(np.int8)
grid.fit(X_fs, y_fs)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)
X_test_fs = X_test[:, idx_top50]
print(accuracy_score(y_test_fs, grid.predict(X_test_fs)))
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

fig = plt.figure(figsize=(10, 4.53), constrained_layout=False, facecolor='0.9', dpi=500)
gs = fig.add_gridspec(nrows=22, ncols=42, left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

# plot samples from each class
ax = fig.add_subplot(gs[1:6, 0:5])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Class 1 - X1', fontsize=7, y=0.92)

inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4,
                                              subplot_spec=gs[1:6, 0:5], wspace=0.0, hspace=0.0)
for j in range(16):
    ax = plt.Subplot(fig, inner_grid[j])
    im = ax.imshow(abs(x_train_9[j, :].reshape((28, 28))), cmap=plt.get_cmap('gray'))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

ax = fig.add_subplot(gs[17:22, 0:5])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Class 2 - X2', fontsize=7, y=0.92)

inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4,
                                              subplot_spec=gs[17:22, 0:5], wspace=0.0, hspace=0.0)
for j in range(16):
    ax = plt.Subplot(fig, inner_grid[j])
    im = ax.imshow(abs(x_train_4[j, :].reshape((28, 28))), cmap=plt.get_cmap('gray'))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

# plot eigenvectors Mean operator M
ax = fig.add_subplot(gs[7:11, 0:4])
im = ax.imshow(abs(eigVecM[:, 0].reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'phi^(M1)', fontsize=7, y=0.94)

ax = fig.add_subplot(gs[12:16, 0:4])
im = ax.imshow(abs(eigVecM[:, 1].reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'phi^(M2)', fontsize=7, y=0.94)

# plot eigenvectors of Difference operator D
ax = fig.add_subplot(gs[2:6, 18:22])
im = ax.imshow(abs(eigVecD[:, 0].reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'phi^(D1)', fontsize=7, y=0.94)

ax = fig.add_subplot(gs[7:11, 18:22])
im = ax.imshow(abs(eigVecD[:, 1].reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'$phi^(D2)$', fontsize=7, y=0.94)

ax = fig.add_subplot(gs[12:16, 18:22])
im = ax.imshow(abs(eigVecD[:, 2].reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'phi^(D3)', fontsize=7, y=0.94)

ax = fig.add_subplot(gs[17:21, 18:22])
im = ax.imshow(abs(eigVecD[:, 3].reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'phi^(D4)', fontsize=7, y=0.94)

# plot ManiFeSt Score
ax = fig.add_subplot(gs[7:16, 33:42])
im = ax.imshow(abs(score.reshape((28, 28))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'ManiFeSt Score - r', fontsize=12, y=0.97)

plt.show()

plt.rc('text', usetex=False)
