import os

from matplotlib import pyplot as plt
import numpy as np

for i, file in enumerate(sorted(os.listdir("testoutput"))):
    data = np.genfromtxt("testoutput/"+file)
    plt.plot(data[:, 0], data[:, 1]+i*1000, label=file[:4])

plt.legend()
plt.tight_layout()
plt.show()