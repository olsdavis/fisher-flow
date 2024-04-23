#!/bin/bash
#SBATCH --job-name=python-training
#SBATCH --output=result-%j.out
#SBATCH --error=result-%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G  # Adjust memory to your requirement
#SBATCH --partition=long

module load anaconda/3  # Load CUDA module if necessary
source activate ~/scratch/venvs/dem/

#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=20
#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=40
#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=60
#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=80
#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=100
#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=120
#python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=140
python -m src.train experiment=toy_dfm_cnn trainer=gpu logger=wandb data.dim=160
