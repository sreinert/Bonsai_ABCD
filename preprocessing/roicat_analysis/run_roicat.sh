#!/bin/bash
#SBATCH --job-name=roicat
#SBATCH --output=roicat_%j.out
#SBATCH --error=roicat_%j.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate roicat 

animal=TAA0000059

sessions=(
    ses-002_date-20250220_protocol-t0
    ses-003_date-20250225_protocol-t1
    ses-005_date-20250311_protocol-t2
    ses-007_date-20250320_protocol-t4
    ses-008_date-20250321_protocol-t5
    ses-009_date-20250323_protocol-t6
    ses-011_date-20250325_protocol-t7
)

python roicat_tracking.py --animal ${animal} --sessions "${sessions[@]}"
