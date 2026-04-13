#!/bin/bash
#SBATCH --job-name=cpa
#SBATCH --output=cpa_%A_%a.out
#SBATCH --error=cpa_%A_%a.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate bonsai_abcd 

PAIRS=(
    "mouse=TAA0000059 cohort=2 t3:t3 t4:t4"
    "mouse=TAA0000066 cohort=2 t3:t3 t4:t4"
    "mouse=001 cohort=3 2LM015:t3"
    "mouse=002 cohort=3 2LM016:t3"
    "mouse=003 cohort=3 2LM015:t3"
    "mouse=005 cohort=3 2LM009:t3"
    "mouse=009 cohort=3 2LM008:t3"
    "mouse=014 cohort=3 2LM011:t3"
)

for ENTRY in "${PAIRS[@]}"; do
    read -a FIELDS <<< "$ENTRY"

    mouse="${FIELDS[0]#mouse=}"
    cohort="${FIELDS[1]#cohort=}"

    for ((i=2; i<${#FIELDS[@]}; i++)); do
        session_t="${FIELDS[$i]}"
        session="${session_t%%:*}"
        stage="${session_t##*:}"

        echo "\nFitting CPA for mouse=$mouse session=$session stage=$stage cohort=$cohort"

        python fit_cpa_yy_diff.py \
            --mouse "$mouse" \
            --session "$session" \
            --stage "$stage" \
            --cohort "$cohort"
    done
done

