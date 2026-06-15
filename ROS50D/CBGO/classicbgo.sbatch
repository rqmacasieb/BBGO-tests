#!/bin/bash
#
#SBATCH --account=OD-224554
#
#SBATCH --job-name=ros50d_10b1.1
#SBATCH --array=0-99
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=500

function runpack()
{
    (( seed = SLURM_ARRAY_TASK_ID * SLURM_NTASKS_PER_NODE + SLURM_PROCID))
    ./runworker $seed > runlog_${seed}.out 2>&1
    
}

export -f runpack

srun bash -c runpack