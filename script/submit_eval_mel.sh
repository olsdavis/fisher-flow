#!/bin/bash
#SBATCH -N 1   # nodes requested
#SBATCH --job-name=sfm_eval 
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=00:45:00
#SBATCH --qos=short
#SBATCH --gres=gpu:1
conda activate sfm
srun -u python -m src.train experiment=enhancer_mel_sfm_cnn model.eval_fbd=true model.eval_ppl=true trainer=gpu logger=wandb data.batch_size=512 trainer.max_epochs=$2 ckpt_path=$1