
import numpy as np
# import scipy
from mpmath import mp

mp.dps = 50


def is_symmetric(mat) -> bool:
    return mat.cols == mat.rows and mp.fsum(mat - mat.T) == 0


def symm(mat):
    return (mat + mat.T) / 2

def random_spd_matrix(n: int):
    mat = mp.randmatrix(n)
    count = 0
    eigvals, eigvecs = mp.eig(mat)

    r = sum([ev != 0 for ev in eigvals])
    if r == n:
        return mat.T @ mat
    while r < n and count < 10:
        mat = mp.randmatrix(n)
        r = sum([ev != 0 for ev in eigvals])
    print(count)
    if count == 10:
        raise RuntimeError("Failed creating a random SPD matrix")
    return mat.T @ mat


def matrix_pow(mat, p: float):
    eigvals, eigvecs = mp.eigsy(mat)
    return eigvecs @ mp.powm(mp.diag(eigvals), p) @ eigvecs.T


def matrix_exp(mat):
    eigvals, eigvecs = mp.eigsy(mat)
    return eigvecs @ mp.expm(mp.diag(eigvals)) @ eigvecs.T


def matrix_log(mat):
    eigvals, eigvecs = mp.eigsy(mat)
    assert mp.fsum(eigvals.apply(mp.im)) == 0
    np_eigvecs = np.array(eigvecs.tolist(), dtype=np.float64)
    np_eigvals = np.array(eigvals.tolist(), dtype=np.float64).reshape((10, ))
    a = eigvecs @ mp.logm(mp.diag(eigvals)) @ eigvecs.T
    a = a.apply(mp.re)
    # for i in range(0, a.rows):
    #     for j in range(i+1, a.cols):
    #         a[i, j] = a[j, i]
    return symm(a)


def exponential_map(p1, p2):
    assert is_symmetric(p1) and is_symmetric(p2)
    p1_power_half = symm(matrix_pow(p1, 0.5))
    p1_power_minus_half = symm(matrix_pow(p1, -0.5))
    expmap_mat = p1_power_half @ matrix_exp(symm(p1_power_minus_half @ p2 @ p1_power_minus_half)) @ p1_power_half
    return symm(expmap_mat)


def log_map(p1, p2):
    assert is_symmetric(p1) and is_symmetric(p2)
    p1_power_half = symm(matrix_pow(p1, 0.5))
    p1_power_minus_half = symm(matrix_pow(p1, -0.5))
    logmap_mat = p1_power_half @ matrix_log(symm(p1_power_minus_half @ p2 @ p1_power_minus_half)) @ p1_power_half
    return symm(logmap_mat)


def spd_geodesic(p1, p2, t: float):
    assert is_symmetric(p1) and is_symmetric(p2)
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    return p1_power_half @ matrix_pow(p1_power_minus_half @ p2 @ p1_power_minus_half, t) @ p1_power_half


def spd_matrix_mean(matrices):
    N = mp.mpf(len(matrices))

    spd_mean = symm(matrices[0])
    for i in range(1, len(matrices)):
        spd_mean = symm(spd_mean + matrices[i])
    spd_mean = symm(spd_mean / N)
    norm_val = 1
    count = 0
    while norm_val > (10**-12) and count < 15:
        temp = log_map(spd_mean, matrices[0])
        print(is_symmetric(temp), mp.norm(temp, p=2))
        for i in range(1, len(matrices)):
            temp1 = log_map(spd_mean, matrices[i])
            print(is_symmetric(temp1), mp.norm(temp1, p=2))
            temp = symm(temp + temp1)
            print(is_symmetric(temp), mp.norm(temp, p=2))
        temp = symm(temp / N)
        spd_mean = exponential_map(spd_mean, temp)
        norm_val = mp.norm(temp, p=2)
        print(norm_val)
        print("****")
        count += 1
    return spd_mean


if __name__ == "__main__":
    m1 = random_spd_matrix(10)
    m2 = random_spd_matrix(10)
    m3 = random_spd_matrix(10)
    m4 = random_spd_matrix(10)
    eigs, eigvecs = mp.eigsy(m1)
    spd_matrix_mean([m1, m2, m3])
    print(eigs)


