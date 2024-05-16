
import numpy as np
from pymanopt.manifolds import Grassmann

def GrassmanMean(GG, vW=None):
    N = GG.shape[-1]

    if vW is None:
        vW = np.ones(N) / N

    vW = vW / sum(vW)
    [D, d] = GG.shape[:2]
    M = Grassmann(D, d)

    mMeanP = GG[:, :, 1] #TODO: Check why this is true
    maxIter = 200
    vNorm = [None] * maxIter

    for ii in range(maxIter):
        mLogMean = 0 * mMeanP
        for nn in range(N):
            mLogMean += vW[nn] * M.log(mMeanP, GG[:, :, nn])
        mMeanP = M.exp(mMeanP, mLogMean)

        vNorm[ii] = np.linalg.norm(mLogMean, ord='fro')
        if vNorm[ii] < 1e-10:
            break

    return mMeanP