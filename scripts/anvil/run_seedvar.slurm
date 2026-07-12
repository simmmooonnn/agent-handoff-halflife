#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p ai
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J hl_sv
#SBATCH -o sv_%j.out
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:=Qwen/Qwen2.5-7B-Instruct}"; : "${N:=100}"; : "${DSEED:?set DSEED}"
echo "RUN seedvar DSEED=$DSEED MODEL=$MODEL N=$N on $(hostname)"
python seedvar.py run "$DSEED" "$MODEL" "$N"
echo "EXIT=$?"
