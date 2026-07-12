# P0 frontier gate verdicts (prereg: out/pilot/frontier_p0_predictions.json,
# commit 4b239c7 BEFORE run). Reads framing_rows_{tag}.jsonl and reports the three
# pre-registered gates plus paired-over-facts bootstrap CIs for the two key
# contrasts (moat = handoff drop - longctx drop; floor lever = faithful-handoff
# S(4) gap), matching the round-6 S4-gap estimand.
import json, sys
from statistics import mean
import numpy as np

RNG = np.random.default_rng(0)


def load(tag):
    rows = [json.loads(l) for l in open(f"out/pilot/framing_rows_{tag}.jsonl")]
    byfact = {}
    for r in rows:
        byfact.setdefault(r["fact_id"], {})[(r["condition"], r["k"])] = r["correct"]
    return rows, byfact


def paired_ci(byfact, stat, n_boot=2000):
    fids = sorted(byfact)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(fids), len(fids))
        try:
            vals.append(stat([byfact[fids[i]] for i in idx]))
        except KeyError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def S(facts, cond, k):
    v = [f[(cond, k)] for f in facts if (cond, k) in f]
    return mean(v) if v else float("nan")


def main(tag):
    rows, byfact = load(tag)
    facts = list(byfact.values())
    ks = sorted({r["k"] for r in rows})
    kmax = max(ks)
    print(f"=== P0 gates: {tag} (n_facts={len(facts)}, ks={ks}) ===")
    for c in ("handoff", "longctx", "verbatim", "faithful"):
        print(f"  {c:9s} S = {[round(S(facts, c, k), 3) for k in ks]}")
    drop_h = S(facts, "handoff", 0) - S(facts, "handoff", kmax)
    drop_l = S(facts, "longctx", 0) - S(facts, "longctx", kmax)
    moat = drop_h - drop_l
    moat_ci = paired_ci(byfact, lambda fs: (S(fs, "handoff", 0) - S(fs, "handoff", kmax))
                        - (S(fs, "longctx", 0) - S(fs, "longctx", kmax)))
    gap4 = S(facts, "faithful", 4) - S(facts, "handoff", 4)
    gap4_ci = paired_ci(byfact, lambda fs: S(fs, "faithful", 4) - S(fs, "handoff", 4))
    gap8 = S(facts, "faithful", kmax) - S(facts, "handoff", kmax)
    gap8_ci = paired_ci(byfact, lambda fs: S(fs, "faithful", kmax) - S(fs, "handoff", kmax))
    verb8 = S(facts, "verbatim", kmax)
    g1 = drop_h >= 0.20
    g2 = moat >= 0.20 and verb8 >= 0.90
    g3 = gap4 >= 0.10
    print(f"  G1 decay_exists : drop_h={drop_h:.3f} >= 0.20            -> {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 moat         : moat={moat:.3f} CI[{moat_ci[0]:.3f},{moat_ci[1]:.3f}] "
          f">= 0.20, verbatim S({kmax})={verb8:.3f} >= 0.90 -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 floor_lever  : S4 gap={gap4:.3f} CI[{gap4_ci[0]:.3f},{gap4_ci[1]:.3f}] "
          f">= 0.10 (S{kmax} gap={gap8:.3f} CI[{gap8_ci[0]:.3f},{gap8_ci[1]:.3f}]) -> {'PASS' if g3 else 'FAIL'}")
    out = {"tag": tag, "drop_h": round(drop_h, 3), "moat": round(moat, 3),
           "moat_ci": [round(x, 3) for x in moat_ci], "verbatim_S_kmax": round(verb8, 3),
           "gap4": round(gap4, 3), "gap4_ci": [round(x, 3) for x in gap4_ci],
           "gap8": round(gap8, 3), "gap8_ci": [round(x, 3) for x in gap8_ci],
           "gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3)}}
    json.dump(out, open(f"out/pilot/p0_gates_{tag}.json", "w"), indent=2)
    print(f"wrote out/pilot/p0_gates_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1])
