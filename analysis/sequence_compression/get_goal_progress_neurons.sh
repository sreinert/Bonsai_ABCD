#!/bin/bash
#SBATCH --job-name=goal_progress
#SBATCH --output=goal_progress_%A_%a.out
#SBATCH --error=goal_progress_%A_%a.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate bonsai_abcd 

PAIRS=(
  "mouse=004 cohort=3 full020:t5 full030:t6"
  "mouse=006 cohort=3 full011:t5 full014:t6"
  "mouse=007 cohort=3 full010:t5 full012:t6"
  "mouse=008 cohort=3 full011:t5 full014:t6"
  "mouse=010 cohort=3 full010:t5 full012:t6"
  "mouse=011 cohort=3 full009:t5 full013:t6"
  "mouse=012 cohort=3 full011:t5 full017:t6"
  "mouse=013 cohort=3 full010:t5 full014:t6"
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

        echo "Extracting goal progress for mouse=$mouse session=$session stage=$stage cohort=$cohort"

        python get_goal_progress_neurons.py \
            --mouse "$mouse" \
            --session "$session" \
            --stage "$stage" \
            --cohort "$cohort"
    done
done

