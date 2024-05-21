#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=36000
#SBATCH --qos=short
#SBATCH --gres=gpu:2
conda activate sfm
python -m src.train experiment=qm_vecfield_sfm seed=$1 data.batch_size=384 trainer.max_epochs=200 trainer=ddp trainer.devices=2 logger=wandb
