#!/bin/bash
# dims=(5 10 20 40 60 80 100 120 140 160)
dims=(80 100 120 140 160)
for dim in ${dims[@]}; do
    sbatch ./script/submit_cnn_sfm_toy.sh $dim
done
