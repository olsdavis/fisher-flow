#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=3:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:1
echo "Here: dim = $1; seed = $2"
conda activate sfm
srun -u python -m src.train experiment=toy_dfm_bmlp data.dim=$1 seed=$2 model.closed_form_drv=true model.manifold=simplex model.ot_method=None logger.wandb.project=sfm-synth-ablation trainer=gpu trainer.max_epochs=500 logger=wandb
