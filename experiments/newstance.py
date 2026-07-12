# R0.1b -- Pre-registered NEW-STANCE prediction test (the stance-generalization
# evidence round 4 demanded; heldout.py only tested budget-transfer of stances
# whose bonus was already fit).
#
# Three instruction heads that appear in NO training run (relay.py modes added
# 2026-07-04): ledger (preservation class), editor (rewrite class), link
# (neutral class). Their bonus/delta were NEVER fit -- they are predicted purely
# from the class rule + class means of OTHER stances, locked in
# out/pilot/newstance_predictions.json and git-committed BEFORE this run.
#
# PRE-REGISTERED SUCCESS: median |S4_pred - S4_obs| <= 0.10 over the 4 cells AND
# the class model beats the length-only null (at OBSERVED w, charitable) on MSE.
import hashlib, json, os, sys
from statistics import mean, median
from facts import make_facts
from relay import item_id_for, TEMPLATE_VERSION
from run import run_sweep, FILLER, _build_backend

NEWCELLS = [("ledger", 25), ("ledger", 12), ("editor", 25), ("link", 25)]


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
    P = json.load(open("out/pilot/newstance_predictions.json"))
    pred = P["predictions"]
    mp = P["model"]

    def L(w):
        import math
        return mp["floor"] + (mp["S0"] - mp["floor"]) * math.exp(
            -4.0 / (mp["c"] * w ** mp["slope"]))

    obs = {}
    all_rows = []
    for arm, b in NEWCELLS:
        rows = run_sweep(backend, facts, ks=[0, 1, 2, 4], conditions=[arm], budget=b,
                         load="light", filler=FILLER, seeds=[0], level="actionable",
                         m_facts=m_facts)
        for r in rows:
            r["label"] = f"{arm}@{b}"
        all_rows.extend(rows)
        with open(f"out/pilot/newstance_rows_{tag}.jsonl", "w") as fh:  # incremental
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
        s4 = mean([r["correct"] for r in rows if r["k"] == 4])
        obs[f"{arm}@{b}"] = s4
    os.makedirs("out/pilot", exist_ok=True)

    cache = backend._cache
    print(f"=== R0.1b new-stance prediction test (n={n}, M={m_facts}) ===")
    print(f"{'cell':>10} {'class':>12} {'w_pred':>7} {'w_obs':>6} {'S4_pred':>8} "
          f"{'S4_obs':>7} {'|err|':>6} {'lenonly':>8} {'|err_L|':>8}")
    errs, errs_len, rows_out = [], [], []
    for arm, b in NEWCELLS:
        key = f"{arm}@{b}"
        p = pred[key]
        wob = realized_w(cache, tag, facts, by_type, arm, b, m_facts)
        s4o = obs[key]
        lonly = min(1.0, max(0.0, L(wob)))
        e = abs(p["S4_pred"] - s4o); el = abs(lonly - s4o)
        errs.append(e); errs_len.append(el)
        rows_out.append({"cell": key, "cls": p["cls"], "w_pred": p["w_pred"],
                         "w_obs": round(wob, 1), "S4_pred": p["S4_pred"],
                         "S4_obs": round(s4o, 3), "abs_err": round(e, 3),
                         "S4_lenonly": round(lonly, 3),
                         "abs_err_lenonly": round(el, 3)})
        print(f"{key:>10} {p['cls']:>12} {p['w_pred']:>7.1f} {wob:>6.1f} "
              f"{p['S4_pred']:>8.2f} {s4o:>7.2f} {e:>6.2f} {lonly:>8.2f} {el:>8.2f}",
              flush=True)
    med = median(errs)
    mse_c = mean([e * e for e in errs]); mse_l = mean([e * e for e in errs_len])
    verdict = (med <= 0.10) and (mse_c < mse_l)
    result = {"model": model, "cells": rows_out, "median_abs_err": round(med, 3),
              "mse_class": round(mse_c, 4), "mse_lenonly": round(mse_l, 4),
              "pass": bool(verdict)}
    json.dump(result, open(f"out/pilot/newstance_{tag}.json", "w"), indent=2)
    print(f"\nmedian |S4 err| = {med:.3f} (<=0.10 needed)")
    print(f"held-out MSE: class-model={mse_c:.4f}  length-only={mse_l:.4f} "
          f"-> class-model {'WINS' if mse_c < mse_l else 'does NOT beat'} length-only")
    v_pass = ("PASS -> the bonus dial generalizes to UNSEEN stances via the class "
              "rule (stance-level out-of-sample; answers round-4 CRITICAL #1)")
    v_fail = ("FAIL -> the surface predicts only measured stances; re-scope Sec 8d "
              "to budget-transfer of profiled stances")
    print(f"GATE: {v_pass if verdict else v_fail}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
