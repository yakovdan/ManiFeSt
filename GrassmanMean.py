
import cupy as cp
from pymanopt.manifolds import Grassmann
from tqdm import trange

def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return cp.transpose(A, (0, 2, 1))



def mysolve(A, b):
    assert A.shape[-1] == A.shape[-2]


def GrassmanMean(GG, vW=None):
    N = GG.shape[0]

    if vW is None:
        vW = cp.ones(shape=(N, 1, 1)) / N

    vW = vW / sum(vW)
    [D, d] = GG.shape[1:]
    M = Grassmann(D, d)

    mMeanP = GG[0]
    maxIter = 200

    for count in trange(maxIter):
        # g0 = GG[0]
        # g1 = GG[1]
        # p = mMeanP
        # a = M.log(mMeanP, GG[0])
        # b = M.log(mMeanP, GG[1])
        # c = (a+b) /2
        #
        # ytx1 = multitransp(g0) @ p
        # At1 = multitransp(g0) - ytx1 @ multitransp(p)
        # Bt1 = cp.linalg.solve(ytx1, At1)
        # u1, s1, vt1 = cp.linalg.svd(multitransp(Bt1), full_matrices=False)
        # arctan_s1 = cp.expand_dims(cp.arctan(s1), -2)
        # a1 = (u1 * arctan_s1) @ vt1
        #
        # ytx2 = multitransp(g1) @ p
        # At2 = multitransp(g1) - ytx2 @ multitransp(p)
        # Bt2 = cp.linalg.solve(ytx2, At2)
        # u2, s2, vt2 = cp.linalg.svd(multitransp(Bt2), full_matrices=False)
        # arctan_s2 = cp.expand_dims(cp.arctan(s2), -2)
        # b1 = (u2 * arctan_s2) @ vt2
        #
        # c1 = (a1+b1)/2
        #
        # ytx3 = multitransp(GG) @ p
        # At3 = multitransp(GG) - ytx3 @ multitransp(p)
        # Bt3 = cp.linalg.solve(ytx3, At3)
        # u3, s3, vt3 = cp.linalg.svd(multitransp(Bt3), full_matrices=False)
        # arctan_s3 = cp.expand_dims(cp.arctan(s3), -2)
        # res = (u3 * arctan_s3) @ vt3



        mLogMean = M.log(mMeanP, GG).sum(axis=0) / N
        mMeanP = M.exp(mMeanP, mLogMean)
        vNorm = cp.linalg.norm(mLogMean, ord='fro')
        if count % 20 == 0:
            print(f"Iteration {count}: {vNorm}")
        #print(f"Iteration {count}: {vNorm}")
        if vNorm < 1e-10:
            break

    print(f"Finished Grassmann Mean with norm: {vNorm}")
    return mMeanP
