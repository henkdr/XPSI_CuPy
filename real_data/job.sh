#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16

ml 2024
ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0
ml cuDNN/8.8.0.121-CUDA-12.6.0
source ../venv/cupy/bin/activate

srun python test.py

# For a profiling run with nsys:
# srun nsys profile \
#    --trace cuda,osrt,nvtx \
#    --gpu-metrics-device=all \
#    --cuda-memory-usage true \
#    --force-overwrite true \
#    --output prof_output \
#    python test.py

