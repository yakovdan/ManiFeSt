
import numpy as np
import scipy as sp
from Symm import Symm

def SpdMean(PP, vW=None):
    Np = PP.shape[-1] # len(PP)

    if vW is None:
        vW = np.ones(Np) / Np

    if Np == 1:
        M = PP
        return M

    M = np.mean(PP, axis=2)

    for ii in range(50):
        A = sp.linalg.sqrtm(M)
        B = np.linalg.inv(A)

        S = np.zeros(M.shape)
        for jj in range(Np):
            C = PP[:, :, jj]
            BCB = Symm(B @ C @ B)
            S = S + vW[jj] * (A @ sp.linalg.logm(BCB) @ A)

        M = Symm(A @ sp.linalg.expm(Symm(B @ S @ B)) @ A)

        eps = np.linalg.norm(S, ord='fro')
        if (eps < 1e-10):
            break

    return M