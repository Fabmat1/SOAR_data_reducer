import ctypes
import numpy as np
from astropy.io import fits

# Load the shared library
lib = ctypes.cdll.LoadLibrary("./liblinefit.dll")

# Define the argument types
lib.fitlines.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # const double* compspec_x
    ctypes.POINTER(ctypes.c_double),  # double* compspec_y
    ctypes.POINTER(ctypes.c_double),  # double* lines
    ctypes.c_int,                     # int lines_size
    ctypes.c_int,                     # int compspec_size
    ctypes.c_double,                  # double center
    ctypes.c_double,                  # double extent
    ctypes.c_double,                  # double quadratic_ext
    ctypes.c_double,                  # double cubic_ext
    ctypes.c_size_t,                  # const size_t c_size
    ctypes.c_size_t,                  # const size_t s_size (default 50)
    ctypes.c_size_t,                  # const size_t q_size
    ctypes.c_size_t,                  # const size_t cub_size
    ctypes.c_double,                  # double  c_cov
    ctypes.c_double,                  # double  s_cov
    ctypes.c_double,                  # double  q_cov
    ctypes.c_double,                  # double  cub_cov
    ctypes.c_double,                  # double  zoom_fac
    ctypes.c_int                      # double  zoom_fac
]


lines = np.genfromtxt("FeAr_lines.txt", delimiter="  ")[:, 0]

compspec_x = np.array([1,2,3,4,5,6,7,8,9])
compspec_y = np.array([1,2,3,4,5,6,7,8,9])
lines_size = len(lines)

compspec_size = len(compspec_x)
center = 4485.9
extent = 1700
quadratic_ext = -7e-6
cubic_ext = 1.5e-10
c_size  = 100
s_size  = 50
q_size  = 100
cub_size = 100
c_cov    = 100.
s_cov    = 0.05
q_cov    = 2.e-5
cub_cov = 2.5e-10
zoom_fac = 25
n_refine = 3
# Define the return type
lib.fitlines.restype = ctypes.POINTER(ctypes.c_double * 4)

compspec_x_ctypes = compspec_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
compspec_y_ctypes = compspec_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
lines_ctypes = lines.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

result_ptr = lib.fitlines(compspec_x_ctypes, compspec_y_ctypes, lines_ctypes, lines_size, compspec_size,
                          center, extent, quadratic_ext, cubic_ext, c_size, s_size, q_size, cub_size, c_cov,
                          s_cov, q_cov, cub_cov, zoom_fac, n_refine)

