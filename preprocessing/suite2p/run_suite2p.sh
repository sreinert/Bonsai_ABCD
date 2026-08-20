#!/bin/bash
#SBATCH --job-name=suite2p
#SBATCH --output=suite2p_%j.out
#SBATCH --error=suite2p_%j.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a4500:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate suite2p 

animal=TAA0000232
session=ses-000
basepath=AtAp_20260119_SequenceCompression/funcimg_screening

python run_suite2p.py --animal ${animal} --session ${session} --basepath ${basepath}
# python Convert_seg_to_stat.py 