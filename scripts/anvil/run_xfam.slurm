#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p ai
#SBATCH --gpus-per-node=1
#SBATCH -t 05:00:00
#SBATCH -J hl_xf
#SBATCH -o xf_%j.out
# R1.1 cross-family replication of the two headline laws on a second family:
#   (1) budget law  -- budget sweep 8/16/25/50 (n=40, matched to the Qwen run)
#   (2) framing law -- reduced arm set (n=100, matched to the Qwen sweep):
#       collapse {node, rolecont} vs neutral {handoff} vs preservation
#       {faithful, manifest}; claim = the ordinal split reproduces.
module load anaconda/2024.02-py311
source activate $SCRATCH/envs/halflife
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $SCRATCH/agent-handoff-halflife
: "${MODEL:?set MODEL via --export}"
BUDGETS="8,16,25,50"
BKS="0,1,2,4,8"
FKS="0,1,2,4"
CONDS="node,rolecont,handoff,faithful,manifest"
echo "RUN xfam MODEL=$MODEL budgets=$BUDGETS framing conds=$CONDS on $(hostname)"
python budget_sweep.py hf "$MODEL" 40 "$BUDGETS" "$BKS" 8
echo "BUDGET_EXIT=$?"
python framing_sweep.py hf "$MODEL" 100 "$FKS" 25 8 "$CONDS"
echo "FRAMING_EXIT=$?"
echo "EXIT=$?"
