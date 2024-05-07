import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

px = [277, 342, 436.2, 593.7, 878.5, 1503.5] # 9 12 2023
wl = [3835.397, 3889.064, 3970.075, 4101.734, 4340.472, 4861.35] # 9 12 2023
# px = [287, 350, 447, 603, 887, 1512] # 9 12 2023
# wl = [3835.397, 3889.064, 3970.075, 4101.734, 4340.472, 4861.35] # 9 12 2023
# px = [0, 395, 582, 825, 869] # blue
# wl = [4730, 4861.35, 4922.3, 5002, 5016] # blue
# px = [0, 830, 1868, 2030]  # red1
# wl = [6033, 6277, 6562.7, 6610]  # red1
# px = [0, 830, 1868, 2030]  # red1
# wl = [6033, 6279, 6562.7, 6610]  # red1


def px_to_wl(px_arr, a, b, c):  # , d):
    return a + b * px_arr + c * px_arr ** 2  # + d * px_arr ** 3


params, errs = curve_fit(px_to_wl,
                         px,
                         wl, )
# bounds=[
#     [-np.inf, -np.inf, -np.inf, 0.82, 2500],
#     [np.inf, np.inf, np.inf, 0.86, 3750]
# ])

print([params[-i - 1] for i in range(len(params))])
pspace = np.linspace(0, 2500, 1000)
plt.plot(pspace, px_to_wl(pspace, *params))
plt.scatter(px, wl)
plt.show()
