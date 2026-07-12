#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -J hl_disambig
#SBATCH -o disambig_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=100}"; : "${KS:=0,1,2,4,8,16}"
: "${BUDGET:=25}"; : "${MFACTS:=8}"
echo "RUN disambig MODEL=$MODEL N=$N KS=$KS BUDGET=$BUDGET MFACTS=$MFACTS on $(hostname)"
python disambig.py hf "$MODEL" "$N" "$KS" "$BUDGET" "$MFACTS"
echo "EXIT=$?"
