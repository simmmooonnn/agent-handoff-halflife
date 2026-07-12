# Experiment Map

Every experiment below was **pre-registered**: predictions, readout bands, and success
criteria were committed to `preregistrations/` *before* the run, and the verdict was
taken from the pre-registered bands afterwards — including the negative ones. Result
files live in `out/pilot/`; response caches in `data/` allow most analyses to be
replayed with **zero API calls**.

## How to run things

Three tiers, from free to expensive:

```bash
# Tier 0 — zero-API replay (uses the shipped response caches; no key needed)
python experiments/e_ci_backfill.py          # bootstrap CIs for the headline contrasts
python experiments/uep_allocation.py         # allocation simulation on cached rows

# Tier 1 — rerun an API experiment (Anthropic/OpenAI key; caches make reruns cheap)
export ANTHROPIC_API_KEY=...
python experiments/e14_real_anchor.py anthropic claude-sonnet-4-6

# Tier 2 — local model ladder on a GPU cluster (SLURM scripts in scripts/anvil/)
sbatch scripts/anvil/run_pilot.slurm
```

All commands run from the repository root; `pytest.ini` puts `harness/` and
`experiments/` on the import path (for plain `python`, prepend
`PYTHONPATH=harness:experiments` on Linux/macOS or set both dirs on Windows).

Run the test suite as an install check: `python -m pytest -q` → `28 passed`.

## Arc 1 — Protocol and the boundary control

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `pilot_gate.py` | — (design doc) | `verdict_Qwen2.5-7B-Instruct.json` | Handoff survival decays exponentially (τ≈1.5, r²≈0.99); token-matched long-context and verbatim controls stay flat → the loss is **compression at the boundary**, not context length. |
| `shape7.py`, `regrade.py` | — | `shape7_*.json` | Decay shape: exponential beats linear/constant/power-law by AICc. |

## Arc 2 — Scaling laws (what governs τ)

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `budget_sweep.py` | — | `budget_sweep_*.json`, `rows_*.jsonl` | **Channel law:** τ ∝ B^0.82 (r²≈1.00) on Qwen-7B; ~×6.6 per decade of word budget. |
| `pilot_gate.py` ladder | — | `verdict_Qwen2.5-{0.5B..32B}.json` | **Size is second-order:** clean (negation-excluded) α≈0.06, CI includes 0. |
| `cross_family.py` | — | `cross_family.json` | Mistral-7B ≈ Qwen-7B at fixed size (architecture-invariant). |
| `mxbudget.py` | — | in `tier1_stats.json` | Capacity-arithmetic null rejected: τ ∝ B^0.86·M^−0.56, budget acts beyond B/M. |
| `tier1_stats.py` | — | `tier1_stats.json` | Bootstrap slope CIs for the above. |

## Arc 3 — The hand-off boundary is a causal, prompt-dependent lever

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `disambig.py` | — | `disambig_*.json` | Same compression op, different framings: persona τ=0.26 ≪ self-notes 0.97 ≪ neutral 1.9 ≪ length-matched neutral 5.0 (~×19, CIs disjoint) → agent-identity content, not prompt length. |
| `realframe.py` | — | in repo | LangGraph reproduction: decay law transfers (τ=2.7, r²=0.99); framing *direction* is prompt-dependent — honest scope. |

## Arc 4 — Fact types, repair, and in-band rescue attempts (mostly negative)

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `e4dropwatch.py` | `e4dropwatch_predictions.json` | `e4dropwatch_*.json` (5 models) | Re-injection helps only when repairs stick; repair-retention is model-dependent; numbers resist in-band repair. |
| `e5_stance.py`, `e5_reanalyze.py` | `e5_stance_predictions.json` | `e5_reanalysis_contrasts.json` | Fidelity stance helps on mean, fails the strict-min gate; adaptive ≈ simple. |
| `e6_diversity.py` | `diversity_coding_predictions.json` | temp-scan notes | **Negative:** parallel-chain diversity coding fails — erasures are correlated across chains. |
| `semantic_modulation.py`, `rescue_ceiling.py` | matching `*_predictions.json` | partial (quota) | Re-encoding helps entities/preferences, not numbers. |

## Arc 5 — Measurement-driven allocation (out-of-band budget)

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `uep_allocation.py` | `uep_allocation_predictions.json` | `uep_allocation_results.json` | Simulation on real cached survival, 5 models: measured-greedy ≥ folk everywhere, near-oracle; fragility ranking is model-dependent (folk mis-picks on 3/5 models). |
| `e7_uep_live.py` | `uep_live_predictions.json`, `uep_live_medium_predictions.json` | `e7_uep_live_*.json` | Live end-to-end on Sonnet. Extreme pressure (M=24): PARTIAL (gradient collapses). Medium (M=12): **CONFIRMS** — measured ≥ folk at every budget, on-oracle; folk mis-picks live (preference, not numeric, is worst on Sonnet). |

## Arc 6 — The mechanism: budget exemption

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `e8_structured_slots.py` | `structured_slots_predictions.json` | `e8_structured_slots_*.json` | In-band `[SLOTS]` block: free-text 0.25 → 0.98 average (numeric 0.08 → 1.00), ≈ external-store level without external infrastructure. Exemption effect +0.729 [0.60, 0.85], every type CI-excl-0. |
| `e8b_slot_depth.py` | `slot_depth_predictions.json` | `e8b_slot_depth_*.json` | K=8 vs K=20: copy−rewrite gap +0.075 [0.000, 0.175], flat in depth → **exemption is the lever; verbatim copying is not required** (a slot rewritten every hop still survives). Refines the folk rule "keep IDs verbatim in a dedicated section". |

## Arc 7 — Stress-testing deployed memory strategies (pre-registered negatives)

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `e9_sota_showdown.py` | `sota_showdown_predictions.json` | `e9_sota_showdown_*.json` | Mem0-style extract+merge memory **holds** a single number flat (0.92) where free-text collapses (0.92→0.08). Deployed compact memory is robust on this axis — its authors never measured it. |
| `e10_scarcity_showdown.py` | `scarcity_showdown_predictions.json` | `e10_scarcity_showdown_*.json` | Under scarcity, verbatim-sentence slots lose to Mem0's compact list (representation efficiency). |
| `e11_true_scarcity.py` | `true_scarcity_predictions.json` | `e11_true_scarcity_*.json` | Compact ledger + measured eviction: beats folk (+0.071) but Mem0 still edges it (−0.039). **Soft adaptive compression beats hard slotting at extreme scarcity.** Three losses reported in full. |

## Arc 8 — The survival surface as a provisioning rule

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `e12_provisioning.py` (Sonnet) | `provisioning_predictions.json` | `e12_provisioning_claude-sonnet-4-6.json` | FAIL by prereg letter — but the failure mode is a finding: Sonnet's surface is floor-degenerate (τ 0.11–0.34 for B=15–40; decay complete by k=1). |
| `e12_provisioning.py` (Qwen, HPC) | `provisioning_qwen_predictions.json` | `e12_provisioning_Qwen2.5-7B-Instruct.json` | PARTIAL: 4-parameter surface predicts held-out (B,k) cells within MAE 0.072 and inverts to a budget target within 0.09 — but dense-grid interpolation matches it. Honest claim: compact validated functional form + inversion capability, not superior accuracy. |

## Arc 9 — Budget-response generality across the frontier

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `e13_saturation.py` | `saturation_predictions.json` | `e13_saturation_*.json` | Haiku 4.5 **binds** (k=4: 0.22→0.56 over 6.7× budget), Opus 4.8 **binds strongly** (0.34→0.88). The budget lever generalizes to the frontier. |
| `e13b_nbump.py` | declared pre-data | printed CIs | Sonnet is the outlier: gain +0.27 vs Haiku +0.59, paired contrast +0.33 [0.16, 0.50] — **budget responsiveness is itself model-dependent**. |
| — | — | `e13_saturation_claude-fable-5.json` (archived) | Fable 5 excluded: deterministic `refusal` stop on ~34% of recall probes under both decoding regimes; grid confounded, archived, not used. |

## Arc 10 — Real-content external-validity anchor

| Script | Prereg | Results | Outcome |
|---|---|---|---|
| `e14_real_anchor.py` + `harness/realdocs.py` | `real_anchor_predictions.json` | `e14_real_anchor_*.json`, `e14_rows_*.jsonl` | On six coherent workplace documents (48 naturally-embedded facts): budgeted hand-offs decay 1.00→0.50 by k=6; the **fragility ordering replicates** (preference worst, negation best — matching the synthetic calibration); slots rescue +0.417 [0.27, 0.56]. Scope: the uncapped arm barely decays — the half-life is a property of **compression under contention**, not of relaying per se. |

## Arc 11 — Statistics

| Script | Results | Purpose |
|---|---|---|
| `e_ci_backfill.py` | printed CIs | Cache-only bootstrap (over facts) for the headline contrasts. |
| `seedvar.py`, `c10_seed_tau_ci.py`, `f_paired_ci.py` | `seedvar_*.json` | Seed sensitivity; bootstrap-over-facts justification (seeds are not replication under temp-0 greedy decoding). |
