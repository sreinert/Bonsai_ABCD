#!/bin/bash
#SBATCH --job-name=roicat
#SBATCH --output=roicat_%j.out
#SBATCH --error=roicat_%j.err
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
source activate roicat 

animal=TAA0000066
sessions=(
    ses-005_date-20250218_protocol-t0
    ses-006_date-20250224_protocol-t1
    ses-007_date-20250304_protocol-t2
    ses-008_date-20250306_protocol-t3
    ses-010_date-20250314_protocol-t4
    ses-011_date-20250315_protocol-t5
    ses-012_date-20250318_protocol-t6
    ses-013_date-20250320_protocol-t7
    ses-014_date-20250326_protocol-t8
    ses-015_date-20250327_protocol-t9
    ses-016_date-20250330_protocol-t10
    ses-017_date-20250331_protocol-t11
    ses-018_date-20250403_protocol-t12
    ses-019_date-20250404_protocol-t13
    ses-020_date-20250412_protocol-t14
    ses-021_date-20250426_protocol-t15
    ses-022_date-20250509_protocol-t16
    ses-023_date-20250516_protocol-t17
)

python roicat_tracking.py --animal ${animal} --sessions "${sessions[@]}"
