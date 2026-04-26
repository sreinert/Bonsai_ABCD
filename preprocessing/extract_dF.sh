#!/bin/bash
#SBATCH --job-name=dF
#SBATCH --output=dF_%A_%a.out
#SBATCH --error=dF_%A_%a.err
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

PAIRS=(
    # "mouse=TAA0000059 cohort=2 t3 t4 t5 t6"
    # "mouse=TAA0000066 cohort=2 t3 t4 t5 t6"
    "mouse=001 cohort=3 2LM015"
    "mouse=002 cohort=3 2LM016"
    "mouse=003 cohort=3 2LM015"
    "mouse=005 cohort=3 2LM009"
    "mouse=009 cohort=3 2LM008"
    "mouse=014 cohort=3 2LM011"
    "mouse=004 cohort=3 full020 full030"
    "mouse=006 cohort=3 full011 full014"
    "mouse=007 cohort=3 full010 full012"
    "mouse=008 cohort=3 full011 full014"
    "mouse=010 cohort=3 full010 full012"
    "mouse=011 cohort=3 full009 full013"
    "mouse=012 cohort=3 full011 full017"
    "mouse=013 cohort=3 full010 full014"
)

for PAIR in "${PAIRS[@]}"; do
    read -a FIELDS <<< "$PAIR"

    mouse="${FIELDS[0]#mouse=}"
    cohort="${FIELDS[1]#cohort=}"

    # Loop over sessions
    for ((i=2; i<${#FIELDS[@]}; i++)); do
        session="${FIELDS[$i]}"

        echo "Running mouse=$mouse session=$session cohort=$cohort"

        python extract_dF.py \
            --mouse "$mouse" \
            --session "$session" \
            --cohort "$cohort"
    done
done
