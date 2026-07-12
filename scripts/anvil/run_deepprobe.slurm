#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p ai
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J hl_dp
#SBATCH -o dp_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=100}"; : "${MFACTS:=8}"
echo "RUN deepprobe MODEL=$MODEL N=$N MFACTS=$MFACTS on $(hostname)"
python deepprobe.py hf "$MODEL" "$N" "$MFACTS"
echo "EXIT=$?"
