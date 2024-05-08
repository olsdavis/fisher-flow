#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=23:00:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:4
conda activate sfm
python -m src.train experiment=promoter_sfm_unet1d model.net.depth=$1 model.net.filters=$2 trainer=ddp trainer.devices=4 data.batch_size=2048 logger=wandb