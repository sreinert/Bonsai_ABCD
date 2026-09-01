#!/bin/bash
#SBATCH --job-name=lr
#SBATCH --output=lr_%A_%a.out
#SBATCH --error=lr_%A_%a.err
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH --mem=8G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user athina.apostolelli.24@ucl.ac.uk

# source ~/.bashrc

module load mamba
source activate bonsai_abcd 

condition="XY_order"
# condition="Y2_ramp"
# condition="YY_diff"

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

        if [[ "$condition" == "Y2_ramp" ]]; then
            echo "Fitting linear regression against number of XY repeats for mouse=$mouse session=$session stage=$stage cohort=$cohort"

            python fit_linear_regression_y2_ramp.py \
                --mouse "$mouse" \
                --session "$session" \
                --stage "$stage" \
                --cohort "$cohort"

        elif [[ "$condition" == "XY_order" ]]; then
            echo "Fitting linear regression against order of X/Y in XY sequences for mouse=$mouse session=$session stage=$stage cohort=$cohort"

            python fit_linear_regression_xy_order.py \
                --mouse "$mouse" \
                --session "$session" \
                --stage "$stage" \
                --cohort "$cohort"

        elif [[ "$condition" == "YY_diff" ]]; then
            echo "Fitting CPA for mouse=$mouse session=$session stage=$stage cohort=$cohort"

            python fit_cpa_yy_diff.py \
                --mouse "$mouse" \
                --session "$session" \
                --stage "$stage" \
                --cohort "$cohort"

        else
            echo "ERROR: Unknown condition '$condition'"
            exit 1
        fi
    done
done

