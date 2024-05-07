import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("output1012_laststar/0323_Gaia3072637990315089024_930_01.txt")
plt.plot(data[:, 0], data[:, 1])
plt.show()
