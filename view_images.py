from matplotlib import pyplot as plt
import numpy as np

for n in range(3):
    img = np.genfromtxt(f"debug_{n}.txt", delimiter=" ")
    print(np.unravel_index(img.argmax(), img.shape))
    plt.scatter([np.unravel_index(img.argmax(), img.shape)[1]], [np.unravel_index(img.argmax(), img.shape)[0]], color="lime", marker="x")
    plt.imshow(img, cmap="plasma")
    plt.show()
    try:
        img2 = np.genfromtxt(f"nop_debug_{n}.txt", delimiter=" ")
        plt.imshow(img-img2, cmap="plasma")
        plt.show()
    except FileNotFoundError:
        pass

img = np.genfromtxt(f"alt_debug.txt", delimiter=" ")
print(np.unravel_index(img.argmax(), img.shape))
plt.scatter([np.unravel_index(img.argmax(), img.shape)[1]], [np.unravel_index(img.argmax(), img.shape)[0]], color="lime", marker="x")
plt.imshow(img, cmap="plasma")
plt.show()