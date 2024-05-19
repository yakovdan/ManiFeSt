
import numpy as np
from SpsdMean import SpsdMean

# Hyperparamaters
d = 10
r = 4
N1 = 20
N2 = 25

# Generate data
mCC1 = np.zeros((d, d, N1))
for ii in range(N1):
    mM = np.random.randn(d, r)
    mCC1[:, :, ii] = (mM @ mM.T)

mCC2 = np.zeros((d, d, N2))
for ii in range(N2):
    mM = np.random.randn(d, r)
    mCC2[:, :, ii] = (mM @ mM.T)

mMean2 = SpsdMean(mCC2, r)
# mMean1Tilde = SpsdMean(mCC1Tilde, r);
print()