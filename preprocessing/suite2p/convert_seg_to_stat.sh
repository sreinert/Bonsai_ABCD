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

set -euo pipefail

module load mamba
source activate suite2p 

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 SUBJECT SESSION [CONVERTER_OPTIONS ...]" >&2
    echo "Example: sbatch $0 sub-02 ses-full002_date-20260818T134310 --meanimg-folder plane0" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
subject=$1
session=$2
shift 2
python "${script_dir}/Convert_seg_to_stat.py" \
    "${subject}" "${session}" --device cuda "$@"
