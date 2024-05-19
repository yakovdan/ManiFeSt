import cupy as cp


def symmetrize(mat):
    return (cp.swapaxes(mat, -1, -2) + mat) / 2


def random_spd_matrix(n: int, cond_num_limit: int = 1000):
    mat = cp.random.rand(n,n)
    eig_vals, _ = cp.linalg.eig(mat)
    r = (eig_vals != 0).sum()
    if r == n:
        mat = mat.T @ mat
        eig_vals, eig_vecs = cp.linalg.eigh(mat)
        largest_eig = cp.max(eig_vals)
        for i in range(len(eig_vals)):
            if eig_vals[i] < largest_eig / cond_num_limit:
                eig_vals[i] += largest_eig / cond_num_limit
        return eig_vecs @ cp.diag(eig_vals) @ eig_vecs.T
    raise RuntimeError("Failed creating a random SPD matrix")


def matrix_pow(mat, p: float):
    eig_vals, eig_vecs = cp.linalg.eigh(mat)
    diag = cp.zeros_like(eig_vecs)
    idx = cp.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = cp.power(eig_vals, p)
    return eig_vecs @ diag @ cp.swapaxes(eig_vecs, -1, -2)


def matrix_exp(mat):
    eig_vals, eig_vecs = cp.linalg.eigh(mat)
    diag = cp.zeros_like(eig_vecs)
    idx = cp.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = cp.exp(eig_vals)
    return eig_vecs @ diag @ cp.swapaxes(eig_vecs, -1, -2)


def matrix_log(mat):
    eig_vals, eig_vecs = cp.linalg.eigh(mat)
    diag = cp.zeros_like(eig_vecs)
    idx = cp.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = cp.log(eig_vals)
    return eig_vecs @ diag @ cp.swapaxes(eig_vecs, -1, -2)


def exponential_map(p1, p2):
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    expmap_mat = p1_power_half @ matrix_exp(p1_power_minus_half @ p2 @ p1_power_minus_half) @ p1_power_half
    return expmap_mat


def log_map(p1, p2):
    p1_power_half = matrix_pow(symmetrize(p1), 0.5)
    p1_power_minus_half = matrix_pow(symmetrize(p1), -0.5)
    logmap_mat = p1_power_half @ matrix_log(symmetrize(p1_power_minus_half @ p2 @ p1_power_minus_half)) @ p1_power_half
    return logmap_mat


def spd_geodesic(p1, p2, t: float):
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    return p1_power_half @ matrix_pow(p1_power_minus_half @ p2 @ p1_power_minus_half, t) @ p1_power_half


def spd_matrix_mean(matrices, iter_limit: int = 200, eps: float = 1e-12):
    print(f"Input matrix norms: {cp.linalg.norm(matrices, ord='fro', axis=(1, 2))}")
    N = matrices.shape[0]
    spd_mean = symmetrize(cp.sum(matrices, axis=0, keepdims=True) / N)

    norm_val = 1
    count = 0
    while norm_val > eps and count < iter_limit:
        sum_projections = log_map(spd_mean, matrices)
        mean_projections = symmetrize(cp.sum(sum_projections, axis=0) / N)
        spd_mean = symmetrize(exponential_map(spd_mean, mean_projections))
        norm_val = cp.linalg.norm(mean_projections, ord='fro')
        print(count, norm_val)
        count += 1
    print(count)
    return spd_mean
