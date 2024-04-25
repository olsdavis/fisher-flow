#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=10:00:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
echo "Here: dim = $1"
conda activate sfm
srun -u python -m src.train experiment=toy_dfm_sfm_cnn data.dim=$1 model.label_smoothing=$2 trainer=gpu trainer.max_epochs=250 logger=wandb data.batch_size=1024
