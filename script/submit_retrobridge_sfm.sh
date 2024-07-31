#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodelist=hopper
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
conda activate sfm
srun -u python -m src.train experiment=retrobridge_sfm seed=$1 data.batch_size=128 trainer.max_epochs=500 trainer=gpu logger=wandb