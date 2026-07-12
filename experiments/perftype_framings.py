# Tier-1 #5: per-fact-type tau for each agent-boundary FRAMING (disambig arms).
# Question: is the framing effect (persona << self << neutral, and the length-matched
# neutral_long) uniform across fact types, or carried by particular types? Reuses the
# per-ftype bootstrap machinery in perftype.py; reads disambig_rows_<tag>.jsonl.
import json, sys, os
import numpy as np
from perftype import load_rows, survival, emp_chance, boot_tau
from analyze import fit_tau

FRAMINGS = ["neutral_long", "handoff", "selfsumm", "persona"]  # display order


def main(tag="Qwen2.5-7B-Instruct"):
    path = f"out/pilot/disambig_rows_{tag}.jsonl"
    rows = load_rows(path)
    present = [c for c in FRAMINGS if any(r["condition"] == c for r in rows)]
    ftypes = sorted({r["ftype"] for r in rows if r["condition"] != "nofact"})
    print(f"=== per-ftype tau by framing  ({tag}) ===")
    print(f"framings present: {present}")
    out = {"tag": tag, "framings": present, "by_ftype": {}}

    # aggregate row per framing (for reference)
    print("\n-- aggregate (all ftypes) --")
    agg = {}
    for c in present:
        ks, S = survival(rows, c)
        ch = emp_chance(rows)
        fit = fit_tau(ks, S, ch)
        agg[c] = fit["tau"]
        print(f"  {c:13s} tau={fit['tau']:.2f}  S={[round(x,2) for x in S]}")
    out["aggregate"] = agg

    hdr = "  ftype      " + "".join(f"{c:>16s}" for c in present)
    print("\n-- per-ftype tau [95% CI] --")
    print(hdr)
    for ft in ftypes:
        cells = []
        rec = {}
        for c in present:
            ks_t, S_t = survival(rows, c, ft)
            ch_t = emp_chance(rows, ft)
            fit = fit_tau(ks_t, S_t, ch_t)
            med, lo, hi = boot_tau(rows, c, ft, ch_t)
            rec[c] = {"tau": fit["tau"], "tau_ci": [lo, hi], "chance": ch_t, "S": S_t}
            cells.append(f"{fit['tau']:.2f}[{lo:.1f},{hi:.1f}]")
        out["by_ftype"][ft] = rec
        print(f"  {ft:10s} " + "".join(f"{x:>16s}" for x in cells))

    # verdict: does persona < neutral hold within EACH fact type?
    if "persona" in present and "handoff" in present:
        print("\n-- within-ftype: persona faster than neutral handoff? (CI-disjoint) --")
        uniform = True
        for ft in ftypes:
            p = out["by_ftype"][ft]["persona"]["tau_ci"]
            h = out["by_ftype"][ft]["handoff"]["tau_ci"]
            sep = p[1] < h[0]
            uniform &= sep
            print(f"  {ft:10s} persona_CI={[round(x,2) for x in p]} vs handoff_CI={[round(x,2) for x in h]}  -> {'SEPARATE' if sep else 'overlap'}")
        print(f"  => framing effect {'UNIFORM across all fact types' if uniform else 'NOT uniform (some types overlap) -- carried by subset'}")

    os.makedirs("out/pilot", exist_ok=True)
    json.dump(out, open(f"out/pilot/perftype_framings_{tag}.json", "w"), indent=2)
    print(f"\nwrote out/pilot/perftype_framings_{tag}.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
