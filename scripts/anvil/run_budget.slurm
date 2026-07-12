#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 00:40:00
#SBATCH -J hl_budget
#SBATCH -o budget_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${BUDGETS:=8,16,25,50,100}"; : "${N:=40}"
echo "RUN budget sweep MODEL=$MODEL BUDGETS=$BUDGETS on $(hostname)"
python budget_sweep.py hf "$MODEL" "$N" "$BUDGETS"
echo "EXIT=$?"
