import time
import cupy as cp
import numpy as np
from cupyx.scipy.interpolate import Akima1DInterpolator

def f(x):
    return x**2

def interp(i, n):
    # Create data array and points to interpolate
    # Using i to generate different data for each iteration.
    x = np.arange(0+i,200000+i, dtype=np.float32)
    y = np.array([f(z) for z in x], dtype=np.float32)
    pti = np.arange(0.0+i, 3200+i, 0.001, dtype=np.float32)

    # Transfer data to GPU
    st = time.perf_counter()
    s0 = time.perf_counter()
    x_on_gpu0 = cp.array(x, dtype=cp.float32)
    y_on_gpu0 = cp.array(y, dtype=cp.float32)
    pti_gpu = cp.array(pti, dtype=cp.float32)
    cp.cuda.stream.get_current_stream().synchronize()
    e0 = time.perf_counter()

    # Initialize interpolator (includes slopes)
    s1 = time.perf_counter()
    s3 = time.perf_counter()
    interpolator = Akima1DInterpolator(x_on_gpu0, y_on_gpu0)
    cp.cuda.stream.get_current_stream().synchronize()
    e1 = time.perf_counter()

    # Perform interpolations
    s2 = time.perf_counter()
    r_gpu = interpolator(pti_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    e2 = time.perf_counter()
    e3 = time.perf_counter()

    # Transfer result to CPU memory
    s4 = time.perf_counter()
    r = cp.asnumpy(r_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    e4 = time.perf_counter()
    et = time.perf_counter()

    #print("Timings:")
    #print("Total:", et-st)
    #print("CPU->GPU", e0-s0)
    #print("Compute init:", e1-s1)
    #print("Compute interpolate:", e2-s2)
    #print("Compute total:", e3-s3)
    #print("GPU->CPU:", e4-s4)

    if i >= int(n / 2):
        global tt, t0, t1, t2, t3, t4

        tt += et-st
        t0 += e0-s0
        t1 += e1-s1
        t2 += e2-s2
        t3 += e3-s3
        t4 += e4-s4

# Timers
tt = t0 = t1 = t2 = t3 = t4 = 0.0

# Run interpolation n times
n = 10
for i in range(n):
    interp(i, n)

# Print timings
n_timed = n - int(n / 2)
print("Average timings of last", n_timed, "iterations:")
print("Total:", tt / n_timed)
print("CPU->GPU", t0 / n_timed)
print("Compute init:", t1 / n_timed)
print("Compute interpolate:", t2 / n_timed)
print("Compute total:", t3 / n_timed)
print("GPU->CPU:", t4 / n_timed)
