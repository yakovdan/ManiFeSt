
import cupy as cp
from pymanopt.manifolds import Grassmann
from tqdm import trange

def GrassmanMean(GG, vW=None):
    N = GG.shape[0]

    if vW is None:
        vW = cp.ones(shape=(N, 1, 1)) / N

    vW = vW / sum(vW)
    [D, d] = GG.shape[1:]
    M = Grassmann(D, d)

    mMeanP = GG[0]
    maxIter = 200

    for _ in trange(maxIter):
        mLogMean = (vW * M.log(mMeanP, GG)).sum(axis=0)
        mMeanP = M.exp(mMeanP, mLogMean)
        vNorm = cp.linalg.norm(mLogMean, ord='fro')
        if vNorm < 1e-10:
            break

    print(f"Finished Grassmann Mean with norm: {vNorm}")
    return mMeanP
