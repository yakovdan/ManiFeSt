from GrassmanMean import GrassmanMean
from SpdMean import SpdMean
from Symm import Symm
from scipy.sparse.linalg import eigs
import cupy as cp
import numpy as np
import scipy as sp
from tqdm import trange
from scipy import io
def SpsdMean(CC, r, mG=None):


    N = CC.shape[0]
    d = CC.shape[1]
    #UU = np.zeros((N, d, r))

    EE, UU = cp.linalg.eigh(Symm(CC))
    UU = cp.flip(UU[:, :, -r:], axis=2)
    #UU, _, _ = cp.linalg.svd(Symm(CC))
    #UU = UU[:, :, :r]
    # for ii in trange(N):
    #
    #     _, a = eigs(Symm(np.copy(CC[ii].get())), r, return_eigenvectors=True)
    #     #assert np.all(np.imag(a) == 0)
    #     UU[ii] = np.real(a)

    #UU = cp.array(UU)
    if mG is None:
        mG = GrassmanMean(UU)
    io.savemat("test", dict({"inputMat": CC.get(), "meanMat": mG.get()}))
    TT = cp.zeros((N, r, r))
    max_cond_num = 0
    for ii in trange(N):
        Xi = Symm(CC[ii])
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
