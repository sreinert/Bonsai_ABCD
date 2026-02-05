#!/bin/bash
#SBATCH --job-name=dF2chan
#SBATCH --output=dF2chan_%A_%a.out
#SBATCH --error=dF2chan_%A_%a.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH --mem=128G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate suite2p 

# Directory where THIS script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to animal_sessions.txt
PAIR_FILE="${SCRIPT_DIR}/../analysis/sequence_compression/animal_sessions.txt"

PAIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PAIR_FILE")
animal=$(echo $PAIR | awk '{print $1}')
session=$(echo $PAIR | awk '{print $2}')

echo "Running animal=$animal session=$session"

python extract_dF_2channels.py --animal "$animal" --session "$session"
