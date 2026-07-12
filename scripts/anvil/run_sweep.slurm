#!/bin/bash
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J handoff
source anvil/setup_env.sh
: "${MODEL:?set MODEL}"; : "${KS:=0,1,2,4,8,16}"; : "${LOAD4BIT:=0}"
export LOAD4BIT
python run.py --provider hf --model "$MODEL" --ks "$KS" --n 100 \
  --conditions handoff,longctx,verbatim \
  --out "out/$(basename $MODEL).jsonl" --cache "data/cache_$(basename $MODEL).json"
