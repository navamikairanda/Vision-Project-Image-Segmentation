#!/bin/bash
#SBATCH -p gpu20
#SBATCH -o /HPS/Navami/work/code/nnti/Vision-Project-Image-Segmentation/slurm-output/slurm-%j.out
#SBATCH -t 0-08:00:00
#SBATCH --gres gpu:1

cd /HPS/Navami/work/code/nnti/Vision-Project-Image-Segmentation

## RUN
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate nnti

python task_1.py

