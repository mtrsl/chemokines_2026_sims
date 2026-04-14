#!/bin/bash

#SBATCH --job-name=ck_gen_params
#SBATCH -t 1:00:00
#SBATCH --mem=1G
#SBATCH -c 1
#SBATCH -e /hps/nobackup/jlees/mjr/slurm_logs/ck_gen_params/%A/%a.err
#SBATCH -o /hps/nobackup/jlees/mjr/slurm_logs/ck_gen_params/%A/%a.out
#SBATCH --mail-type ALL

base_dir=/nfs/research/jlees/mjr/chemokines/chemokines_2026_paper_sims

fixed_params_file=${base_dir}/fixed_params.csv
all_params_file=${base_dir}/all_params.csv

uv run \
    --project ${base_dir} \
    ${base_dir}/generate_param_sets.py \
    ${fixed_params_file} \
    ${all_params_file}

n_lines=$(wc -l ${all_params_file} | cut -d' ' -f1)
n_param_sets=$((n_lines - 1))

sbatch --array=1-${n_param_sets} ${base_dir}/param_sweep_array.sh
