#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 02:00:00
#SBATCH -J hl_pilot
#SBATCH -o pilot_%j.out
# Pilot go/no-go gate. Reuses the existing wmf env (torch/transformers/scipy/bnb)
# and the already-cached Qwen2.5 text models. Runs fully offline from hf_cache.
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=40}"; : "${KS:=0,1,2,4,8,16}"
: "${BUDGET:=25}"; : "${MFACTS:=8}"; : "${LOAD:=light}"  # BUDGET is now a WORD limit
echo "RUN pilot MODEL=$MODEL N=$N KS=$KS BUDGET=$BUDGET MFACTS=$MFACTS LOAD=$LOAD on $(hostname) $(nvidia-smi -L 2>/dev/null | head -1)"
python pilot_gate.py hf "$MODEL" "$N" "$KS" "$BUDGET" "$MFACTS" "$LOAD"
echo "EXIT=$?"
