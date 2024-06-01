
import cupy as cp
import scipy as sp
from Symm import Symm
from tqdm import trange
from tools import matrix_log, matrix_exp, matrix_pow
import numpy as np

def SpdMean(PP, vW=None):
    Np = PP.shape[0]  # len(PP)
    if vW is None:
        vW = cp.ones((Np, 1, 1)) / Np

    if Np == 1:
        M = PP
        return M

    M = cp.mean(PP, axis=0, keepdims=True)

    for c in trange(200):
        A = matrix_pow(M, 0.5)
        B = matrix_pow(M, -0.5)
        BCB = Symm(B @ PP @ B)
        S = (vW * (A @ matrix_log(BCB) @ A)).sum(axis=0, keepdims=True)
        M = Symm(A @ matrix_exp(Symm(B @ S @ B)) @ A)
        eps = cp.linalg.norm(S[0], ord='fro')
        if c % 20 == 0:
            print(f"Iteration: {c}, eps: {eps}")
        if eps < 1e-10:
            break
    print(f"Finished SPD Mean with norm: {eps}")

    return M[0]