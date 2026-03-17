#!/bin/bash

#SBATCH --job-name=ck_param_sweep
#SBATCH -t 4:00:00
#SBATCH --mem=2G
#SBATCH -c 1
#SBATCH -e /hps/nobackup/jlees/mjr/slurm_logs/param_sweep_array/%A/%a.err
#SBATCH -o /hps/nobackup/jlees/mjr/slurm_logs/param_sweep_array/%A/%a.out
#SBATCH --mail-type ALL
#SBATCH --array=1-100

task_id=$SLURM_ARRAY_TASK_ID

base_dir=/nfs/research/jlees/mjr/chemokines/chemokines_2026_paper_sims

all_params_file=${base_dir}/all_params.csv

# Extract the row corresponding to this task (add 1 to skip header)
params=$(sed -n "$((task_id + 1))p" $all_params_file)

# Parse columns
IFS=',' read -r alpha Pe D_ratio n_cells cell_motility chi cell_init CCL21_added <<< "$params"

output_dir=/hps/nobackup/jlees/mjr/chemokines/sims/${task_id}
mkdir -p $output_dir

uv run \
    --project ${base_dir} \
    ${base_dir}/main.py \
    --output_dir ${output_dir} \
    --alpha ${alpha} \
    --Pe ${Pe} \
    --D_ratio ${D_ratio} \
    --n_cells ${n_cells} \
    --cell_motility ${cell_motility} \
    --chi ${chi} \
    --cell_init ${cell_init} \
    --CCL21_added ${CCL21_added}
