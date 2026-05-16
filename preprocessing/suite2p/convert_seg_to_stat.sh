#!/bin/bash
#SBATCH --job-name=suite2p
#SBATCH --output=suite2p_%j.out
#SBATCH --error=suite2p_%j.err
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
source activate suite2p 

sessions=(
    TAA0000065/ses-015_date-20250427_protocol-t9
)

python Convert_seg_to_stat.py --sessions "${sessions[@]}"
