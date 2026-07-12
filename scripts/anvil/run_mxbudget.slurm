#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -J hl_mxb
#SBATCH -o mxb_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=60}"; : "${BUDGETS:=8,16,25,50}"
: "${KS:=0,1,2,4}"; : "${MFACTS:=4}"
echo "RUN mxbudget MODEL=$MODEL N=$N BUDGETS=$BUDGETS KS=$KS MFACTS=$MFACTS on $(hostname)"
python mxbudget.py hf "$MODEL" "$N" "$BUDGETS" "$KS" "$MFACTS"
echo "EXIT=$?"
