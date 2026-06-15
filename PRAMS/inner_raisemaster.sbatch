#!/bin/bash
#
#SBATCH --account=SC-001095
#
#SBATCH --job-name=prams_bbgo_run8b4_master
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8000

cd master
./pestpp-mou pest/prams_bbgo.pst /H :4020

cd ..
rm -r worker*
