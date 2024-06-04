#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=36000
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
conda activate sfm
python -m src.train experiment=qm_vecfield_sfm seed=$1 data.batch_size=256 trainer.max_epochs=500 trainer=gpu logger=wandb
