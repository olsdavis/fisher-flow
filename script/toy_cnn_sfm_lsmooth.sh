#!/bin/bash
dims=(0 0.7 0.8 0.9 0.95 0.999 0.9999)
for dim in ${dims[@]}; do
    sbatch ./script/submit_cnn_sfm_toy_lsmooth.sh 100 $dim
done
