#!/bin/bash
dims=(5 10 20 40 60 80 100 120 140 160)
for dim in ${dims[@]}; do
    sbatch ./script/submit_bmlp_sfm_lin.sh $dim $1
done
