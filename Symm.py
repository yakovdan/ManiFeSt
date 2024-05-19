
def Symm(A):
    if A.ndim == 2:
        return (A + A.T) / 2
    return (A + A.transpose([0, 2, 1])) / 2
