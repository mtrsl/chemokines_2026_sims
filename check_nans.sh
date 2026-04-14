#!/bin/bash

#SBATCH --job-name=ck_check_nans
#SBATCH -t 1:00:00
#SBATCH --mem=4G
#SBATCH -c 1
#SBATCH -e /hps/nobackup/jlees/mjr/slurm_logs/ck_check_nans/%A/%a.err
#SBATCH -o /hps/nobackup/jlees/mjr/slurm_logs/ck_check_nans/%A/%a.out
#SBATCH --mail-type ALL
#SBATCH --array=1-480

task_id=$SLURM_ARRAY_TASK_ID

output_dir=/hps/nobackup/jlees/mjr/chemokines/sims/${task_id}

rg -i -e nan -e inf ${output_dir}
