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

mkdir venv
python -m venv venv/cupy
source venv/cupy/bin/activate

# Apply patch
git clone https://github.com/cupy/cupy.git
cd cupy
git checkout 25e552d5d679dcdc6f7cc3d81310a9b265463137
git apply ../cupy.patch
git submodule update --init
pip install -e .

