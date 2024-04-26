#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=23:00:00
#SBATCH --mem=36000
#SBATCH --qos=long
#SBATCH --gres=gpu:1
conda activate sfm
# srun -u python main.py -e dfm_toy -c config/toy_dfm/bmlp.yml -m sphere --wandb
srun -u python -m src.train experiment=promoter_sfm_promdfm trainer.max_epochs=200 trainer=gpu logger=wandb data.batch_size=64
