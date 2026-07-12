# SLURM scripts for the open-model ladder

These scripts produced the Qwen2.5 (0.5B–32B), Mistral-7B, and Phi-3 results on an
NSF ACCESS GPU cluster (A100/H100 nodes). They are written to be portable:

- Replace `YOUR_ALLOCATION` in each `#SBATCH -A` line with your allocation/account.
- `$SCRATCH` is assumed to point at your cluster scratch directory (most clusters set
  it; otherwise `export SCRATCH=/path/to/your/scratch`).
- One-time setup: `bash setup_env.sh` creates the conda env at
  `$SCRATCH/envs/halflife` and `python prefetch.py` downloads the HF models into
  `$SCRATCH/hf_cache` (jobs run with `HF_HUB_OFFLINE=1`).
- The repo is expected at `$SCRATCH/agent-handoff-halflife`.
- Module names (e.g. `module load anaconda/...`) are site-specific — adjust to your
  cluster's module system.

Submit with parameter overrides via environment variables, e.g.:

```bash
export MODEL=Qwen/Qwen2.5-7B-Instruct N=100 KS=0,1,2,4,8 BUDGET=25 MFACTS=8
sbatch scripts/anvil/run_pilot.slurm
```

Note: pass comma-valued variables as exported environment variables (as above), not
inside `sbatch --export=...` — SLURM splits that list on commas.
