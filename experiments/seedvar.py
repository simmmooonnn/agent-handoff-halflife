# R1.3 decoding-variance check (round-4): every headline CI in the paper bootstraps
# over FACTS along a single greedy trajectory. This measures how much the S(k)
# estimates -- and the deepprobe front-loading result, whose 0.000 error is exact
# only under greedy decoding -- move under sampling (temp=0.7, 3 independent
# decoding seeds, same fact set and prompts).
#
# Cells: 4 anchor arms spanning the surface -- handoff (neutral), manifest
# (preservation), ledger (new stance, role-poisoned), node (collapse); k to 8 for
# the first three so front-loading is tested under sampling.
#
# PRE-REGISTERED GATES (predictions = the greedy values, committed before the run):
#   G1 robustness:     median over cells of |mean_over_seeds S - S_greedy| <= 0.10
#   G2 front-loading:  for handoff/manifest/ledger, |S(8)-S(4)| of the seed-mean
#                      <= 0.05 each (median over the 3 arms also reported)
# FAIL on G1 -> greedy estimates are decoding artifacts; all headline numbers get
# a temp-sensitivity caveat and CIs must be widened to include seed variance.
# FAIL on G2 only -> front-loading is a greedy artifact; deepprobe claim narrows
# to "under deterministic decoding".
#
# usage:  python seedvar.py run <dseed> [model] [n]     (one SLURM job per dseed)
#         python seedvar.py score [model]               (after all 3 land)
import json, os, sys
from statistics import mean, median, pstdev
from facts import make_facts
from run import run_sweep, FILLER, _build_backend

CELLS = [("handoff", [0, 1, 2, 4, 8]), ("manifest", [0, 1, 2, 4, 8]),
         ("ledger", [0, 1, 2, 4, 8]), ("node", [0, 1, 2, 4])]
BUDGET, M_FACTS, TEMP, DSEEDS = 25, 8, 0.7, (1, 2, 3)


def run_one(dseed, model="Qwen/Qwen2.5-7B-Instruct", n=100):
    os.environ["TEMP"] = str(TEMP)
    os.environ["DSEED"] = str(dseed)
    tag = model.split("/")[-1]
    facts = make_facts(int(n), seed=0)
    # per-dseed cache file: 3 dseed jobs run concurrently and CachedBackend's
    # whole-file rewrite is not safe under concurrent writers
    backend = _build_backend("hf", model, f"data/cache_seedvar_{tag}_s{dseed}.json")
    rows = []
    for cond, ks in CELLS:
        rows += run_sweep(backend, facts, ks=ks, conditions=[cond], budget=BUDGET,
                          load="light", filler=FILLER, seeds=[0], level="actionable",
                          m_facts=M_FACTS)
        with open(f"out/pilot/seedvar_rows_{tag}_s{dseed}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(f"=== seedvar dseed={dseed} temp={TEMP} (n={n}) ===")
    for cond, ks in CELLS:
        s = [round(mean([r["correct"] for r in rows
                         if r["condition"] == cond and r["k"] == k]), 3) for k in ks]
        print(f"  {cond:10} S={s}")


def score(model="Qwen/Qwen2.5-7B-Instruct"):
    tag = model.split("/")[-1]
    P = json.load(open("out/pilot/seedvar_predictions.json"))["greedy"]
    per_seed = {}
    for d in DSEEDS:
        f = f"out/pilot/seedvar_rows_{tag}_s{d}.jsonl"
        if not os.path.exists(f):
            print(f"missing {f} -- run all dseeds first"); return
        per_seed[d] = [json.loads(l) for l in open(f)]

    def s_of(rows, cond, k):
        v = [r["correct"] for r in rows if r["condition"] == cond and r["k"] == k]
        return mean(v) if v else None

    print(f"{'cell':>14} {'greedy':>7} {'seedmean':>9} {'sd':>6} {'|dev|':>6}")
    devs, cells_out = [], []
    fl = {}
    for cond, ks in CELLS:
        for k in ks:
            g = P[cond][str(k)]
            ss = [s_of(per_seed[d], cond, k) for d in DSEEDS]
            m, sd = mean(ss), pstdev(ss)
            dev = abs(m - g)
            devs.append(dev)
            fl[(cond, k)] = m
            cells_out.append({"cell": f"{cond}@k{k}", "greedy": g,
                              "seed_mean": round(m, 3), "seed_sd": round(sd, 3),
                              "abs_dev": round(dev, 3)})
            print(f"{cond + '@k' + str(k):>14} {g:>7.2f} {m:>9.3f} {sd:>6.3f} {dev:>6.3f}")
    g1 = median(devs)
    fronts = {c: abs(fl[(c, 8)] - fl[(c, 4)]) for c in ("handoff", "manifest", "ledger")}
    g2_all = all(v <= 0.05 for v in fronts.values())
    verdict = (g1 <= 0.10) and g2_all
    print(f"\nG1 median |seed-mean - greedy| = {g1:.3f} (<=0.10 needed)")
    print("G2 front-loading under sampling |S8-S4|: " +
          ", ".join(f"{c}={v:.3f}" for c, v in fronts.items()) + " (each <=0.05 needed)")
    result = {"model": model, "cells": cells_out, "g1_median_dev": round(g1, 3),
              "g2_front": {c: round(v, 3) for c, v in fronts.items()},
              "pass": bool(verdict)}
    json.dump(result, open(f"out/pilot/seedvar_{tag}.json", "w"), indent=2)
    print("GATE: " + ("PASS -> greedy estimates and front-loading are robust to "
                      "decoding stochasticity (temp 0.7, 3 seeds)"
                      if verdict else
                      "FAIL -> apply the pre-registered fallback caveat "
                      "(G1 fail: widen CIs / G2 fail: front-loading narrows to "
                      "deterministic decoding)"))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_one(int(sys.argv[2]), *sys.argv[3:])
    else:
        score(*sys.argv[2:])
