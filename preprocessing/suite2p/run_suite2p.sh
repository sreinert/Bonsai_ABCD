#!/bin/bash
#SBATCH --job-name=suite2p
#SBATCH --output=suite2p_%j.out
#SBATCH --error=suite2p_%j.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --mem=128G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate suite2p 

animal=TAA0000066
session=ses-021_date-20250426_protocol-t15

python run_suite2p.py --animal ${animal} --session ${session} 