# -*- coding: utf-8 -*-
"""
Created on

@author:
"""

# import sys

import scipy.io

import numpy as np
import cupy as cp
from sklearn.metrics.pairwise import euclidean_distances


def construct_multidiag(arr):
    assert arr.ndim == 2
    diag_out = cp.zeros(shape=(arr.shape[0], arr.shape[1], arr.shape[1]))
    id = cp.arange(arr.shape[1])
    diag_out[:, id, id] = arr[:, id]
    return diag_out


def construct_kernel(X, y, percentile=50):
    labels = list(set(y))
    kernels = []

    for i in range(len(labels)):
        elements = X[y == i].shape[0]
        x = X[y == i].reshape(elements, -1)
        K_dis = euclidean_distances(np.transpose(x))
        #epsilon = np.percentile(K_dis[~np.eye(K_dis.shape[0], dtype=bool)], percentile)
        perc = np.percentile(K_dis[~np.eye(K_dis.shape[0], dtype=bool)], percentile)
        #med = np.median(K_dis[~np.eye(K_dis.shape[0], dtype=bool)])
        epsilon = perc

        K = np.exp(-(K_dis ** 2) / (2 * epsilon ** 2))
        kernels.append(K)
    return kernels


def calc_tol(matrix, var_type='float64', energy_tol=0):
    tol = np.max(matrix) * len(matrix) * np.core.finfo(var_type).eps
    tol2 = np.sqrt(np.sum(matrix ** 2) * energy_tol)
    tol = np.max([tol, tol2])

    return tol


def spsd_geodesics(G1, G2, p=0.5, r=None, eigVecG1=None, eigValG1=None, eigVecG2=None, eigValG2=None):
    if eigVecG1 is None:
        eigValG1, eigVecG1 = cp.linalg.eigh(G1)
    if eigVecG2 is None:
        eigValG2, eigVecG2 = cp.linalg.eigh(G2)

    if r is None:
        tol = calc_tol(eigValG1)
        rank_G1 = len(cp.abs(eigValG1)[cp.abs(eigValG1) > tol])

        tol = calc_tol(eigValG2)
        rank_G2 = len(cp.abs(eigValG2)[cp.abs(eigValG2) > tol])

        r = min(rank_G1, rank_G2)

    maxIndciesG1 = cp.flip(cp.argsort(cp.abs(eigValG1))[-r:], 0)
    V1 = eigVecG1[:, maxIndciesG1]
    lambda1 = eigValG1[maxIndciesG1]

    maxIndciesG2 = cp.flip(cp.argsort(cp.abs(eigValG2))[:, -r:], 1)
    lambda2 = cp.take_along_axis(eigValG2, maxIndciesG2, 1)
    maxIndciesG2 = cp.expand_dims(maxIndciesG2, 1)
    V2 = cp.take_along_axis(eigVecG2, maxIndciesG2, axis=2)

    O2, sigma, O1T = cp.linalg.svd(cp.swapaxes(V2, -1, -2) @ V1)
    O1 = cp.swapaxes(O1T, -1, -2)

    sigma[sigma < -1] = -1
    sigma[sigma > 1] = 1
    theta = cp.arccos(sigma)

    U1 = V1 @ O1
    R1 = cp.swapaxes(O1, -1, -2) @ cp.diag(lambda1) @ O1

    lambda2_diag = construct_multidiag(lambda2)
    U2 = V2 @ O2
    R2 = cp.swapaxes(O2, -1, -2) @ lambda2_diag @ O2

    tol = calc_tol(sigma.get())
    valid_ind = cp.where(cp.abs(sigma - 1) > tol)
    pinv_sin_theta = cp.zeros(theta.shape)
    pinv_sin_theta[valid_ind] = 1 / cp.sin(theta[valid_ind])

    UG1G2 = U1 @ construct_multidiag(cp.cos(theta * p)) + (cp.eye(G1.shape[0]) - U1 @ cp.swapaxes(U1, -1, -2)) @ U2 @ construct_multidiag(
        pinv_sin_theta) @ construct_multidiag(cp.sin(theta * p))

    return UG1G2, R1, R2, O1, lambda1


def get_operators(K1, K2, t: float = 0.5, use_spsd=True):
    if use_spsd:
        eigValK1, eigVecK1 = np.linalg.eigh(K1)
        tol = calc_tol(eigValK1)
        rank_K1 = len(np.abs(eigValK1)[np.abs(eigValK1) > tol])

        eigValK2, eigVecK2 = np.linalg.eigh(K2)
        tol = calc_tol(eigValK2)
        rank_K2 = len(np.abs(eigValK2)[np.abs(eigValK2) > tol])

        # create SPSD Mean operator M
        min_rank = min(rank_K1, rank_K2)
        UK1K2, RK1, RK2, OK1, lambdaK1 = spsd_geodesics(K1, K2, p=t, r=min_rank, eigVecG1=eigVecK1, eigValG1=eigValK1,
                                                        eigVecG2=eigVecK2, eigValG2=eigValK2)

        RK1PowerHalf = OK1.T @ np.diag(np.sqrt(lambdaK1)) @ OK1
        RK1PowerMinusHalf = OK1.T @ np.diag(1 / np.sqrt(lambdaK1)) @ OK1
        e, v = np.linalg.eigh(RK1PowerMinusHalf @ RK2 @ RK1PowerMinusHalf)
        e[e < 0] = 0
        RK1K2 = RK1PowerHalf @ v @ np.diag(np.power(e, t)) @ v.T @ RK1PowerHalf  # replaced sqrt with power of t
        M = UK1K2 @ RK1K2 @ UK1K2.T

        eigValM, eigVecM = np.linalg.eigh(M)
        tol = calc_tol(eigValM)
        rank_M = len(np.abs(eigValM)[np.abs(eigValM) > tol])

        # create SPSD Difference operator D
        min_rank = min(rank_K1, rank_M)
        UMK1, RM, RK1, OM, lambdaM = spsd_geodesics(M, K1, p=1, r=min_rank, eigVecG1=eigVecM, eigValG1=eigValM,
                                                    eigVecG2=eigVecK1, eigValG2=eigValK1)

        RMPowerHalf = OM.T @ np.diag(np.sqrt(lambdaM)) @ OM
        RMPowerMinusHalf = OM.T @ np.diag(1 / np.sqrt(lambdaM)) @ OM
        e, v = np.linalg.eigh(RMPowerMinusHalf @ RK1 @ RMPowerMinusHalf)
        tol = calc_tol(e)
        e[e < tol] = tol
        logarithmic = RMPowerHalf @ v @ np.diag(np.log(e)) @ v.T @ RMPowerHalf

        D = UMK1 @ logarithmic @ UMK1.T

    else:  # SPD form
        # create SPD Mean operator M
        K1 = K1 + np.eye(K1.shape[0]) * np.core.finfo('float64').eps * 2
        K2 = K2 + np.eye(K2.shape[0]) * np.core.finfo('float64').eps * 2

        eigValK1, eigVecK1 = np.linalg.eigh(K1)
        tol = calc_tol(eigValK1)
        rank_K1 = len(np.abs(eigValK1)[np.abs(eigValK1) > tol])

        K1PowerHalf = eigVecK1 @ np.diag(np.sqrt(eigValK1)) @ eigVecK1.T
        K1PowerMinusHalf = eigVecK1 @ np.diag(1 / np.sqrt(eigValK1)) @ eigVecK1.T
        e, v = np.linalg.eigh(K1PowerMinusHalf @ K2 @ K1PowerMinusHalf)
        e[e < 0] = 0
        M = K1PowerHalf @ v @ np.diag(np.sqrt(e)) @ v.T @ K1PowerHalf

        eigValM, eigVecM = np.linalg.eigh(M)
        tol = calc_tol(eigValM)
        rank_M = len(np.abs(eigValM)[np.abs(eigValM) > tol])

        # create SPD Difference operator D
        MPowerHalf = eigVecM @ np.diag(np.sqrt(eigValM)) @ eigVecM.T
        MPowerMinusHalf = eigVecM @ np.diag(1 / np.sqrt(eigValM)) @ eigVecM.T
        e, v = np.linalg.eigh(MPowerMinusHalf @ K1 @ MPowerMinusHalf)
        tol = calc_tol(e)
        e[e < tol] = tol
        D = MPowerHalf @ v @ np.diag(np.log(e)) @ v.T @ MPowerHalf

    return M, D


def compute_manifest_score(D):
    eigValD, eigVecD = np.linalg.eigh(D)

    eigVec_norm = eigVecD ** 2
    score = eigVec_norm @ np.abs(eigValD)

    return score


def ManiFeSt2(X, y, t:float = 0.5, percentile: float = 50, use_spsd=True):
    K1, K2 = construct_kernel(X, y, percentile)

    M, D = get_operators(K1, K2, use_spsd=use_spsd)

    score = compute_manifest_score(D)
    idx = np.argsort(score, 0)[::-1]

    ##eig_vecs
    eigValM, eigVecM = np.linalg.eigh(M)
    eigValD, eigVecD = np.linalg.eigh(D)

    sorted_indexes = np.argsort(np.abs(eigValM))[::-1]
    eigVecM = eigVecM[:, sorted_indexes]
    eigValM = eigValM[sorted_indexes]
    sorted_indexes = np.argsort(np.abs(eigValD))[::-1]
    eigVecD = eigVecD[:, sorted_indexes]
    eigValD = eigValD[sorted_indexes]

    eig_vecs = (eigVecD, eigValD, eigVecM, eigValM)

    return score, idx, eig_vecs
