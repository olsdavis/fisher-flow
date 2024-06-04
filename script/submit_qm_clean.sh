#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=36000
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
conda activate sfm
srun -u python -m src.train experiment=qm_clean_sfm trainer=gpu logger=wandb seed=$1
