from GrassmanMean import GrassmanMean
from SpdMean import SpdMean
from Symm import Symm
from scipy.sparse.linalg import eigs
import cupy as cp
import numpy as np
import scipy as sp
from tqdm import trange

def SpsdMean(CC, r, mG=None):
    N = CC.shape[0]
    d = CC.shape[1]
    UU = np.zeros((N, d, r))
    EE = np.zeros((N, r))

    X = Symm(CC)
    for ii in trange(N):
        [EE[ii], UU[ii]] = eigs(X[ii].get(), r, return_eigenvectors=True)

    UU = cp.array(UU)
    if mG is None:
        mG = GrassmanMean(UU)

    TT = cp.zeros((N, r, r))
    max_cond_num = 0
    for ii in trange(N):
        Xi = CC[ii]
        Ui = UU[ii]
        [Oi, _, OWi] = cp.linalg.svd(Ui.T @ mG, compute_uv=True)
        GOi = Ui @ Oi @ OWi
        Ti = GOi.T @ Xi @ GOi
        TT[ii] = Symm(Ti)
        evals, evecs = cp.linalg.eigh(TT[ii])
        cond_num = cp.max(evals) / cp.min(evals)
        max_cond_num = cond_num if cond_num > max_cond_num else max_cond_num

    print(f"Largest condition number of spd matrix: {max_cond_num}")
    mP = SpdMean(TT)
    mC = Symm(mG @ mP @ mG.T)

    return mC, mG, mP, UU, TT
