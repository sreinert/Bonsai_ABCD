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

PAIRS=(
    "004 full020 full030"
    "006 full011 full014"
    "007 full010 full012"
    "008 full011 full014"
    "010 full010 full012"
    "011 full009 full013"
    "012 full011 full017"
    "013 full010 full014"
    "001 2LM015"
    "002 2LM016"
    "003 2LM015"
    "005 2LM009"
    "009 2LM008"
    "014 2LM011"
)

for PAIR in "${PAIRS[@]}"; do
    # Split line into array
    read -a FIELDS <<< "$PAIR"

    animal="${FIELDS[0]}"

    # Loop over all sessions for this animal
    for ((i=1; i<${#FIELDS[@]}; i++)); do
        session="${FIELDS[$i]}"

        echo "Running animal=$animal session=$session"
        python extract_dF_2channels.py --animal "$animal" --session "$session"
    done
done
