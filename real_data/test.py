import time
import cupy as cp
import numpy as np
from cupyx.scipy.interpolate import Akima1DInterpolator
from cupyx.scipy.interpolate import BarycentricInterpolator
from cupyx.scipy.interpolate import KroghInterpolator
from cupyx.scipy.interpolate import PchipInterpolator
from cupyx.profiler import benchmark
from decimal import Decimal

def f():
    b = np.load('interpolation_products_reduced_correct_pulse.npz')
    phase_array = b['phase_data_array']                     # x-data: float64
    intensity_array = b['intensity_data_array']             # y-data: float64
    phase_query_array = b['phase_query_array']              # x-query: float64
    intensity_query_array = b['intensity_query_array']      # y-result float64
    cell_radiates = b['cell_radiates']                      # mask int32        

    # Create y_res output array: structure:
    colatitudes = intensity_query_array.shape[0]
    energies = intensity_query_array.shape[1]
    azimuths = intensity_query_array.shape[2]
    query_phases = intensity_query_array.shape[3]

    y_res = np.zeros(shape=(colatitudes, energies, azimuths, query_phases), dtype=np.float64)
    rel_diffs = np.zeros(shape=(colatitudes, energies, azimuths, query_phases), dtype=np.float64)

    # Timers
    ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Transfer data to GPU
    s0 = time.perf_counter()
    phase_array_gpu = cp.array(phase_array, dtype=cp.float64)
    intensity_array_gpu = cp.array(intensity_array, dtype=cp.float64)
    phase_query_array_gpu = cp.array(phase_query_array, dtype=cp.float64)

    cp.cuda.stream.get_current_stream().synchronize()
    e0 = time.perf_counter()
    s1 = time.perf_counter()

    for c in range(colatitudes):
        s2 = time.perf_counter()
        x_data_gpu = phase_array_gpu[c,:]
        cp.cuda.stream.get_current_stream().synchronize()
        e2 = time.perf_counter()
        ts2 += e2-s2

        for e in range(energies):
            s3 = time.perf_counter()
            y_data_gpu = intensity_array_gpu[c,e,:]
            cp.cuda.stream.get_current_stream().synchronize()
            e3 = time.perf_counter()
            ts3 += e3-s3

            s4 = time.perf_counter()
            # Initialize interpolator (includes slopes calculation)
            interpolator = Akima1DInterpolator(x_data_gpu, y_data_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            e4 = time.perf_counter()
            ts4 += e4-s4

            # Flatten x-query points for all azimuths
            s5 = time.perf_counter()
            all_x_query_data_gpu = phase_query_array_gpu[c,:,:].flatten()
            cp.cuda.stream.get_current_stream().synchronize()
            e5 = time.perf_counter()
            ts5 += e5-s5

            # Interpolate
            s6 = time.perf_counter()
            all_y_res_on_gpu = interpolator(all_x_query_data_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            e6 = time.perf_counter()
            ts6 += e6-s6

            # Copy result to CPU memory
            s7 = time.perf_counter()
            all_y_res = cp.asnumpy(all_y_res_on_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            e7 = time.perf_counter()
            ts7 += e7-s7

            # Un-flatten result
            s8 = time.perf_counter()
            all_y_res = all_y_res.reshape(azimuths, query_phases)
            for a in range(azimuths):
                y_res[c,e,a] = all_y_res[a]
            
            cp.cuda.stream.get_current_stream().synchronize()
            e8 = time.perf_counter()
            ts8 += e8-s8

    cp.cuda.stream.get_current_stream().synchronize()
    e1 = time.perf_counter()

    # Save output
    # np.save('y_res_flatten.npz', y_res[c,e,a])
    
    # Print timing info
    print("Timers:\n0 = data copy CPU->GPU\n1 = full interp loop\n2 = x data slicing\n3 = y data slicing\n4 = akima instantiation (slopes calc)\n5 = flatten x query\n6 = interpolation kernel\n7 = GPU->CPU copy\n8 = unflatten result")

    print("e0:", e0-s0)
    print("e1:", e1-s1)
    print("te2:", ts2)
    print("te3:", ts3)
    print("te4:", ts4)
    print("te5:", ts5)
    print("te6:", ts6)
    print("te7:", ts7)
    print("te8:", ts8)

    # Apply mask
    for c in range(colatitudes):
        for a in range(azimuths):
            # print(cell_radiates[c,a])
            if cell_radiates[c,a] == 0:
                for e in range(energies):
                    # Set output to zero.
                    y_res[c,e,a] = np.zeros(shape=y_res[c,e,a].shape)
                    # Also need to set expected output to 0.0. Otherwise relative error will be 100%.
                    intensity_query_array[c,e,a] = np.zeros(shape=y_res[c,e,a].shape)



    ### Caclulate & plot relative differences between output and expected output#
    for c in range(colatitudes):
        for a in range(azimuths):
            for e in range(energies):
                rel_diff = abs(y_res[c,e,a] - intensity_query_array[c,e,a]) / intensity_query_array[c,e,a] * 100
                
                for i in range(len(rel_diff)):
                    # Correct relative difference if results and expected result are both 0.
                    if (y_res[c,e,a][i] == intensity_query_array[c,e,a][i] and intensity_query_array[c,e,a][i] == 0.0):
                        rel_diff[i] = 0.0
                
                # Set cases where rel diff is inf to 0.
                rel_diff[rel_diff == np.inf] = 0
                rel_diffs[c,e,a] = rel_diff

    # Plot histogram of relative differences. 
    hist_data = rel_diffs.flatten()
    y_res_flattened = y_res.flatten()
    y_exp_flattened = intensity_query_array.flatten()


    ### Dump data ###
    print("Start dump")

    rel_err_sum = 0.0
    rel_err_max = 0.0

    with open("cupy_dump.txt", "w") as f:
        f.write("Interpolation result |  Expected result    |    Abs error       |       Rel error       |       Rel error (cum)    |    Rel error (max)\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------------------\n")

        for i, y_res_val in enumerate(y_res_flattened):
            rel_diff_val = hist_data[i] 
            rel_err_sum += rel_diff_val
            if rel_diff_val > rel_err_max:
                    rel_err_max = rel_diff_val
            abs_err_val = np.abs(y_res_val - y_exp_flattened[i])
            f.write("%2E\t\t%2E\t\t%2E\t\t%2E\t\t%2E\t\t%2E\n" % (Decimal(y_res_val), Decimal(y_exp_flattened[i]), Decimal(abs_err_val), Decimal(rel_diff_val), Decimal(rel_err_sum), Decimal(rel_err_max)))


f()
