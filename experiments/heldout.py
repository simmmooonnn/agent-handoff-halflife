# Pre-registered HELD-OUT prediction test for the two-dial instruction-surface model.
#
# The model (locked in out/pilot/heldout_predictions.json, committed BEFORE this run):
#   S4(arm,budget) = clip( L(w_hat) + bonus[arm] )
#   w_hat = 2.78*budget^0.5355 + delta[arm]      (neutral length model + per-arm shift)
#   L(w)  = floor + (S0-floor)*exp(-4/tau), tau = 0.0192*w^1.531   (fit on 4 neutral anchors)
# delta[arm], bonus[arm] fit ONLY on the 12 training cells (framing_sweep + protocol + proto2).
#
# This script runs 6 held-out (arm,budget) combinations that appear in NO training cell,
# measures observed realized-length and S4, and scores them against the locked predictions.
# PRE-REGISTERED SUCCESS: median |S4_pred - S4_obs| <= 0.10 over the 6 cells AND the
# two-dial model beats the length-only null (S4 = L(w_obs), no bonus) on held-out MSE.
# The second clause is what proves the preservation-bonus dial is real, not overfit noise.
import hashlib, json, os, sys
from statistics import mean, median
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import run_sweep, FILLER, _build_backend
from analyze import empirical_chance

HELDOUT = [("faithful", 8), ("node", 50), ("manifest", 12),
           ("manifest", 50), ("faithman", 8), ("persona", 50)]
FLOOR, S0, C, SLOPE = 0.22158169050544804, 0.91, 0.019200023337236097, 1.5312774407034195

def L(w):
    import math
    return FLOOR + (S0 - FLOOR) * math.exp(-4.0 / (C * w ** SLOPE))

def realized_w(cache, tag, facts, by_type, cond, budget, m_facts):
    prefix = f"hf/{tag}"
    lens = []
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:m_facts - 1])
        dctx = ",".join(d.fact_id for d in distractors)
        ctx = hashlib.sha1(f"{FILLER}|{TEMPLATE_VERSION}|{dctx}".encode()).hexdigest()[:8]
        for k in (1, 2, 4):
            iid = item_id_for(f, k, cond, budget, "light", 0)
            for i in range(k):
                key = f"{prefix}:{iid}:{ctx}:h{i}"
                if key in cache:
                    lens.append(len(cache[key].split()))
    return mean(lens) if lens else float("nan")

def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_pilot_{tag}.json")
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    pred = json.load(open("out/pilot/heldout_predictions.json"))["predictions"]

    # run the 6 held-out cells
    obs = {}
    for arm, b in HELDOUT:
        rows = run_sweep(backend, facts, ks=[0, 1, 2, 4], conditions=[arm], budget=b,
                         load="light", filler=FILLER, seeds=[0], level="actionable", m_facts=m_facts)
        s4 = mean([r["correct"] for r in rows if r["k"] == 4])
        obs[f"{arm}@{b}"] = {"S4_obs": s4, "arm": arm, "budget": b}
    os.makedirs("out/pilot", exist_ok=True)

    cache = backend._cache
    print(f"=== held-out prediction test (n={n}, M={m_facts}) ===")
    print(f"{'cell':>14} {'w_pred':>7} {'w_obs':>6} {'S4_pred':>8} {'S4_obs':>7} "
          f"{'|err|':>6} {'lenonly':>8} {'|err_L|':>8}")
    errs, errs_len, rows_out = [], [], []
    for arm, b in HELDOUT:
        key = f"{arm}@{b}"
        p = pred[key]
        wob = realized_w(cache, tag, facts, by_type, arm, b, m_facts)
        s4o = obs[key]["S4_obs"]
        # length-only null uses the OBSERVED w (charitable to the null)
        lonly = min(1.0, max(0.0, L(wob)))
        e = abs(p["S4_pred"] - s4o); el = abs(lonly - s4o)
        errs.append(e); errs_len.append(el)
        rows_out.append({"cell": key, "w_pred": p["w_pred"], "w_obs": round(wob, 1),
                         "S4_pred": p["S4_pred"], "S4_obs": round(s4o, 3),
                         "abs_err": round(e, 3), "S4_lenonly": round(lonly, 3),
                         "abs_err_lenonly": round(el, 3)})
        print(f"{key:>14} {p['w_pred']:>7.1f} {wob:>6.1f} {p['S4_pred']:>8.2f} {s4o:>7.2f} "
              f"{e:>6.2f} {lonly:>8.2f} {el:>8.2f}", flush=True)
    med = median(errs); mse2 = mean([e * e for e in errs]); mseL = mean([e * e for e in errs_len])
    verdict = (med <= 0.10) and (mse2 < mseL)
    result = {"model": model, "cells": rows_out, "median_abs_err": round(med, 3),
              "mse_twodial": round(mse2, 4), "mse_lenonly": round(mseL, 4),
              "pass": bool(verdict)}
    json.dump(result, open(f"out/pilot/heldout_{tag}.json", "w"), indent=2)
    print(f"\nmedian |S4 err| = {med:.3f} (<=0.10 needed)")
    print(f"held-out MSE: two-dial={mse2:.4f}  length-only={mseL:.4f} "
          f"-> two-dial {'WINS' if mse2 < mseL else 'does NOT beat'} length-only")
    print(f"GATE: {'PASS -> two-dial surface is predictive out-of-sample (non-circular method contribution)' if verdict else 'FAIL -> surface does not generalize; report as descriptive decomposition only'}")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
