#!/bin/bash
#SBATCH --job-name=roicat
#SBATCH --output=roicat_%j.out
#SBATCH --error=roicat_%j.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate roicat 

animal=004
cohort=3

sessions=(
    full020
    full030
)

python roicat_tracking.py --animal ${animal} --cohort ${cohort} --sessions "${sessions[@]}"
