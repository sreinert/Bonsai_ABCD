#!/bin/bash
#SBATCH --job-name=SCmodel
#SBATCH --output=behav_model_%A_%a.out
#SBATCH --error=behav_model_%A_%a.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate bonsai_abcd 

PAIRS=(
    "mouse=01 num_Ys=3"
    "mouse=02 num_Ys=2"
    "mouse=03 num_Ys=3"
    "mouse=04 num_Ys=3"
    "mouse=05 num_Ys=2"
    "mouse=06 num_Ys=3"
    "mouse=07 num_Ys=2"
    "mouse=08 num_Ys=3"
    "mouse=09 num_Ys=3"
)

for ENTRY in "${PAIRS[@]}"; do
    read -a FIELDS <<< "$ENTRY"

    mouse="${FIELDS[0]#mouse=}"
    num_Ys="${FIELDS[1]#num_Ys=}"

    echo "Fitting behaviour model for mouse=$mouse (num_Ys=$num_Ys)"

    python fit_behaviour_model.py \
        --mouse "$mouse" \
        --num_Ys "$num_Ys" \
        --grid_size 3
    done
done

