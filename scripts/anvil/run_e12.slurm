#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p ai
#SBATCH --gpus-per-node=1
#SBATCH -t 03:00:00
#SBATCH -J hl_e12
#SBATCH -o e12_%j.out
# Discriminative predict-then-verify provisioning on Qwen-7B (prereg
# out/pilot/provisioning_qwen_predictions.json, commit 3cb458c). Env vars are
# hardcoded here (NOT --export) to dodge the sbatch comma-splitting trap.
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
export PROV_PROBES=8
export PROV_CAL_B=8,16,40
export PROV_CAL_K=0,1,2,4
export PROV_HELD=25:2,25:4,64:1,64:4,64:8,16:8
export PROV_T=0.5
export PROV_KT=2
echo "RUN e12 Qwen-7B on $(hostname) $(nvidia-smi -L 2>/dev/null | head -1)"
python e12_provisioning.py hf Qwen/Qwen2.5-7B-Instruct
echo "EXIT=$?"
