from matplotlib import pyplot as plt
import numpy as np

for n in range(2):
    img = np.genfromtxt(f"imagethingy_{n}.txt", delimiter=" ")
    plt.imshow(img)
    plt.show()