#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=23:00:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
echo "Here: dim = $1"
conda activate sfm
srun -u python -m src.train experiment=toy_dfm_cnn data.dim=$1 trainer=gpu trainer.max_epochs=1000 logger=wandb
