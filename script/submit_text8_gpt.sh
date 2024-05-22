#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=23:00:00
#SBATCH --mem=36000
#SBATCH --qos=medium
#SBATCH --gres=gpu:4
conda activate sfm
srun -u python -m src.train experiment=text8_sfm_gpt trainer=ddp trainer.devices=4 logger=wandbsqueue