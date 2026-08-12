#!/bin/bash
#SBATCH --job-name=SCmodel
#SBATCH --output=behav_model_%A_%a.out
#SBATCH --error=behav_model_%A_%a.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --array=0-5
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate bonsai_abcd 

# Mouse IDs
MICE=("01" "02" "03" "04" "05" "07")

# Number of Y landmarks for each mouse
NUM_YS=(3 2 3 3 3 2)

# Get mouse and number of Y landmarks for this array task
mouse=${MICE[$SLURM_ARRAY_TASK_ID]}
num_Ys=${NUM_YS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Running behaviour model"
echo "Mouse: $mouse"
echo "num_Ys: $num_Ys"
echo "SLURM array task: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

python fit_behaviour_model.py \
    --mouse "$mouse" \
    --num_Ys "$num_Ys" \
    --grid_size 10
    
# PAIRS=(
#     "mouse=01 num_Ys=3"
#     "mouse=02 num_Ys=2"
#     "mouse=03 num_Ys=3"
#     "mouse=04 num_Ys=3"
#     "mouse=05 num_Ys=2"
#     "mouse=07 num_Ys=2"
# )

# for ENTRY in "${PAIRS[@]}"; do
#     read -a FIELDS <<< "$ENTRY"

#     mouse="${FIELDS[0]#mouse=}"
#     num_Ys="${FIELDS[1]#num_Ys=}"

#     echo "Fitting behaviour model for mouse=$mouse (num_Ys=$num_Ys)"

#     python fit_behaviour_model.py \
#         --mouse "$mouse" \
#         --num_Ys "$num_Ys" \
#         --grid_size 10
#     done
# done

