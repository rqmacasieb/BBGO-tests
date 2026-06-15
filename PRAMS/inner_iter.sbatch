#!/bin/bash
#
#SBATCH --account=SC-001095
#
#SBATCH --job-name=prams_bbgo_run8b4
#SBATCH --array=0-99
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8000

function runpack()
{
     (( n = SLURM_ARRAY_TASK_ID * SLURM_NTASKS_PER_NODE + SLURM_PROCID))
     ./inner_runworker $n
}

export -f runpack

srun bash -c runpack

