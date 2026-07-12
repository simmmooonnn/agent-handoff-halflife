#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p ai
#SBATCH --gpus-per-node=1
#SBATCH -t 10:00:00
#SBATCH -J hl_e3
#SBATCH -o e3_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=100}"; : "${K:=8}"; : "${BUDGET:=25}"; : "${MFACTS:=8}"
THETAS="0.5,0.35,0.2"
echo "RUN e3 MODEL=$MODEL N=$N K=$K BUDGET=$BUDGET MFACTS=$MFACTS THETAS=$THETAS on $(hostname)"
python e3.py hf "$MODEL" "$N" "$K" "$BUDGET" "$MFACTS" "$THETAS"
echo "EXIT=$?"
