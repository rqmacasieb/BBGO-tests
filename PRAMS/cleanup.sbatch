#!/bin/bash
#SBATCH --account=SC-001095
#SBATCH --job-name=prams_cleanup
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=50

for iteration in {1..18}; do
    cd outer_$iteration
    rm -rf model/ 
    echo "deleted outer_$iteration model directory"
    cd ..   

done

