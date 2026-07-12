#!/bin/bash
module load anaconda/2024.02-py311
export PATH=$SCRATCH/envs/handoff/bin:$PATH
export HF_HOME=$SCRATCH/hf_cache
# one-time: conda create -p $SCRATCH/envs/handoff python=3.11 -y
#           pip install torch transformers bitsandbytes accelerate numpy scipy matplotlib
