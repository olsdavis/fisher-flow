#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:1
conda activate sfm
srun -u python -m src.train experiment=promoter_sfm_promdfm ckpt_path=$1 trainer=gpu trainer.max_epochs=$2 logger=wandb