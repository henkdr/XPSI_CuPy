# Tests with cupy on Snellius for akima-1D interpolation for X-PSI

## Setup
To do setup, submit job:
```
sbatch setup.sh
```

## Synthetic data
To run benchmark with synthetically generated data (fp32)
```
cd synthetic
sbatch job.sh
```

## Real data
To use real data `interpolation_products_reduced_correct_pulse.npz` (fp64):

```
cd real_data
sbatch job.sh
```

The npz file is assumed to be in the `real_data` directory.

This will generate a dump of the interpolation results called `cupy_dump.txt` showing interpolation results, including absolute and relative errors compared to the expected results."

