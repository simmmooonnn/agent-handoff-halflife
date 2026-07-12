# Pre-registered DEEP-PROBE test: shallow probes (k<=4) forecast unseen depth.
#
# Two cached findings motivate this (both post-hoc so far, hence this pre-registered
# confirmation):
#   (a) front-loaded loss: |S(8)-S(4)| median 0.01 across 9 models / 3 families
#       (pilot ladder rows) -- survival converges to its floor by k~4;
#   (b) 2-hop geometric probe predicts S(4) at median err 0.027 (0.005 on the four
#       stances never used to build anything).
# Together: probe an ARBITRARY per-hop instruction to depth <=4 and the law gives
# the whole depth curve -- the text-level shortcut is impossible (newstance FAIL:
# role cues dominate retention directives), but probing is cheap and sufficient.
#
# This script runs k=8 for 7 arms (incl. all 4 genuinely-new stances) and k=16 for
# 2 arms, then scores the PERSISTENCE forecast S(k>=8) = S(4)_measured against
# observations. Predictions locked in out/pilot/deepprobe_predictions.json and
# committed BEFORE the run.
# PRE-REGISTERED SUCCESS: median |S8_pred - S8_obs| <= 0.05 over the 7 cells AND
# both S(16) cells within 0.12.
import json, os, sys
from statistics import mean, median
from facts import make_facts
from run import run_sweep, FILLER, _build_backend

ARMS_K8 = [("ledger", 25), ("ledger", 12), ("editor", 25), ("link", 25),
           ("manifest", 25), ("faithman", 25), ("neutral_long", 25)]
ARMS_K16 = [("ledger", 25), ("manifest", 25)]


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_pilot_{tag}.json")
    P = json.load(open("out/pilot/deepprobe_predictions.json"))["predictions"]
    os.makedirs("out/pilot", exist_ok=True)

    all_rows = []

    def s_of(rows, k):
        v = [r["correct"] for r in rows if r["k"] == k]
        return mean(v) if v else None

    obs8, obs16 = {}, {}
    for arm, b in ARMS_K8:
        ks = [8, 16] if (arm, b) in ARMS_K16 else [8]
        rows = run_sweep(backend, facts, ks=ks, conditions=[arm], budget=b,
                         load="light", filler=FILLER, seeds=[0], level="actionable",
                         m_facts=m_facts)
        for r in rows:
            r["label"] = f"{arm}@{b}"
        all_rows.extend(rows)
        with open(f"out/pilot/deepprobe_rows_{tag}.jsonl", "w") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
        obs8[f"{arm}@{b}"] = s_of(rows, 8)
        if 16 in ks:
            obs16[f"{arm}@{b}"] = s_of(rows, 16)

    print(f"=== deep-probe persistence test (n={n}, M={m_facts}) ===")
    print(f"{'cell':>16} {'S4_meas':>8} {'S8_pred':>8} {'S8_obs':>7} {'|e8|':>5} "
          f"{'S16_obs':>8} {'|e16|':>6}")
    e8s, e16s, rows_out = [], [], []
    for arm, b in ARMS_K8:
        key = f"{arm}@{b}"
        p = P[key]
        s8o = obs8[key]
        e8 = abs(p["S8_pred"] - s8o); e8s.append(e8)
        s16o = obs16.get(key); e16 = None
        if s16o is not None:
            e16 = abs(p["S16_pred"] - s16o); e16s.append(e16)
        rows_out.append({"cell": key, "S4_meas": p["S4_meas"],
                         "S8_pred": p["S8_pred"], "S8_obs": round(s8o, 3),
                         "abs_err8": round(e8, 3),
                         "S16_obs": (round(s16o, 3) if s16o is not None else None),
                         "abs_err16": (round(e16, 3) if e16 is not None else None)})
        print(f"{key:>16} {p['S4_meas']:>8.2f} {p['S8_pred']:>8.2f} {s8o:>7.2f} "
              f"{e8:>5.2f} "
              f"{(s16o if s16o is not None else float('nan')):>8.2f} "
              f"{(e16 if e16 is not None else float('nan')):>6.2f}", flush=True)
    med8 = median(e8s)
    ok16 = all(e <= 0.12 for e in e16s) if e16s else False
    verdict = (med8 <= 0.05) and ok16
    result = {"model": model, "cells": rows_out,
              "median_abs_err8": round(med8, 3),
              "max_abs_err16": (round(max(e16s), 3) if e16s else None),
              "pass": bool(verdict)}
    json.dump(result, open(f"out/pilot/deepprobe_{tag}.json", "w"), indent=2)
    print(f"\nmedian |S8 err| = {med8:.3f} (<=0.05 needed); "
          f"S16 errs {[round(e,3) for e in e16s]} (each <=0.12 needed)")
    v_pass = ("PASS -> loss is front-loaded and shallow probes forecast unseen depth "
              "for unseen stances: probe-and-forecast is a validated method")
    v_fail = ("FAIL -> persistence does not hold for these arms; report depth "
              "forecasting as measured-depth-only")
    print(f"GATE: {v_pass if verdict else v_fail}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
