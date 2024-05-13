
import numpy as np


def symm(mat):
    return (np.swapaxes(mat, -1, -2) + mat) / 2


def random_spd_matrix(n: int, cond_num_limit: int = 1000):
    mat = np.random.rand(n,n)
    eig_vals, _ = np.linalg.eig(mat)
    r = (eig_vals != 0).sum()
    if r == n:
        mat = mat.T @ mat
        eig_vals, eig_vecs = np.linalg.eigh(mat)
        largest_eig = np.max(eig_vals)
        for i in range(len(eig_vals)):
            if eig_vals[i] < largest_eig / cond_num_limit:
                eig_vals[i] += largest_eig / cond_num_limit
        return eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
    raise RuntimeError("Failed creating a random SPD matrix")


def matrix_pow(mat, p: float):
    eig_vals, eig_vecs = np.linalg.eigh(mat)
    diag = np.zeros_like(eig_vecs)
    idx = np.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = np.power(eig_vals, p)
    return eig_vecs @ diag @ np.swapaxes(eig_vecs, -1, -2)


def matrix_exp(mat):
    eig_vals, eig_vecs = np.linalg.eigh(mat)
    diag = np.zeros_like(eig_vecs)
    idx = np.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = np.exp(eig_vals)
    return eig_vecs @ diag @ np.swapaxes(eig_vecs, -1, -2)


def matrix_log(mat):
    eig_vals, eig_vecs = np.linalg.eigh(mat)
    diag = np.zeros_like(eig_vecs)
    idx = np.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = np.log(eig_vals)
    return eig_vecs @ diag @ np.swapaxes(eig_vecs, -1, -2)


def exponential_map(p1, p2):
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    expmap_mat = p1_power_half @ matrix_exp(p1_power_minus_half @ p2 @ p1_power_minus_half) @ p1_power_half
    return expmap_mat


def log_map(p1, p2):
    p1_power_half = matrix_pow(symm(p1), 0.5)
    p1_power_minus_half = matrix_pow(symm(p1), -0.5)
    logmap_mat = p1_power_half @ matrix_log(symm(p1_power_minus_half @ p2 @ p1_power_minus_half)) @ p1_power_half
    return logmap_mat


def spd_geodesic(p1, p2, t: float):
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    return p1_power_half @ matrix_pow(p1_power_minus_half @ p2 @ p1_power_minus_half, t) @ p1_power_half


def spd_matrix_mean(matrices, iter_limit: int = 200, eps: float = 1e-12):
    print(f"Input matrix norms: {np.linalg.norm(matrices, ord='fro', axis=(1, 2))}")
    N = matrices.shape[0]
    spd_mean = symm(np.sum(matrices, axis=0, keepdims=True) / N)

    norm_val = 1
    count = 0
    while norm_val > eps and count < iter_limit:
        sum_projections = log_map(spd_mean, matrices)
        mean_projections = symm(np.sum(sum_projections, axis=0) / N)
        spd_mean = symm(exponential_map(spd_mean, mean_projections))
        norm_val = np.linalg.norm(mean_projections, ord='fro')
        print(count, norm_val)
        count += 1
    print(count)
    return spd_mean


if __name__ == "__main__":
    m1 = random_spd_matrix(10)[None, :]
    m2 = random_spd_matrix(10)[None, :]
    m3 = random_spd_matrix(10)[None, :]
    m4 = random_spd_matrix(10)[None, :]
    m5 = random_spd_matrix(10)[None, :]
    m6 = random_spd_matrix(10)[None, :]
    m7 = random_spd_matrix(10)[None, :]
    m8 = random_spd_matrix(10)[None, :]

    c1 = np.concatenate([m1, m2], axis=0)
    c2 = np.concatenate([m1, m2, m3, m4, m5, m6, m7, m8], axis=0)
    q1 = spd_matrix_mean(c1)
    q2 = spd_matrix_mean(c2)


