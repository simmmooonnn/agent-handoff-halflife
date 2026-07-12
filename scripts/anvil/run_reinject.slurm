#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -J hl_ri
#SBATCH -o ri_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=60}"; : "${K:=8}"; : "${BUDGET:=25}"; : "${MFACTS:=8}"
echo "RUN reinject MODEL=$MODEL N=$N K=$K BUDGET=$BUDGET MFACTS=$MFACTS on $(hostname)"
python reinject_eval.py hf "$MODEL" "$N" "$K" "$BUDGET" "$MFACTS"
echo "EXIT=$?"
