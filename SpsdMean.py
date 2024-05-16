
import numpy as np
from GrassmanMean import GrassmanMean
from SpdMean import SpdMean
from Symm import Symm
from scipy.sparse.linalg import eigs
import scipy as sp
def SpsdMean(CC, r, mG=None):
    N = CC.shape[-1]
    d = CC.shape[0]
    UU = np.zeros((d, r, N))

    for ii in range(N):
        Xi = Symm(CC[:, :, ii])
        [_, UU[:, :, ii]] = eigs(Xi, r, return_eigenvectors=True)

    if mG is None:
        mG = GrassmanMean(UU)

    TT = np.zeros((r, r, N))
    for ii in range(N):
        Xi = CC[:, :, ii]
        Ui = UU[:, :, ii]
        [Oi, _, OWi] = np.linalg.svd(Ui.T @ mG, compute_uv=True)
        GOi = Ui @ Oi @ OWi
        Ti = GOi.T @ Xi @ GOi
        TT[:, :, ii] = Symm(Ti)

    mP = SpdMean(TT)
    mC = Symm(mG @ mP @ mG.T)

    return mC, mG, mP
