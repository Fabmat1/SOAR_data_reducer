import subprocess
import sys
import os


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None


print("Checking for C++ compiler...")
gcc_exists = False
gpp_exists = False
if is_tool("g++"):
    gpp_exists = True
    process = subprocess.Popen("g++ -O3 -fopenmp -march=native linefit.cpp -o linefit", shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("Setup Successful")
# if is_tool("gcc"):
#     gcc_exists = True
#     process = subprocess.Popen("gcc -O3 -fopenmp -march=native linefit.cpp -o linefit", shell=True, stdout=subprocess.PIPE)
#     process.wait()


if os.name == "nt":
    if not gpp_exists:
        raise ModuleNotFoundError("No suitable C++ compiler found on your system. Install MinGW and make sure to link g++ or gcc to your PATH")
else:
    if not gpp_exists:
        raise ModuleNotFoundError("No suitable C++ compiler found on your system. Install g++.")

