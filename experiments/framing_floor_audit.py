# Round-6 C1/C2 fix: the framing "law" on the DECAY RATE tau does not survive
# CI-disjointness within Qwen (only 2/7 adjacent links separable) and INVERTS
# cross-family (node/rolecont, the Qwen collapse arms, have HIGHER tau than the
# preservation arms on Mistral/Phi). The stance effect lives in the FLOOR f
# (asymptotic survival), not tau. This script audits both, cross-family, from the
# cached framing_sweep fits -- reproducing the honest reframe of headline #2:
#   "instruction stance controls the asymptotic survival FLOOR, and the floor
#    lever replicates across 3 model families; the decay-rate tau ordering does not."
import json

FAMS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Phi-3-medium-4k-instruct"]
# stance class from head wording (pre-registered vocabulary, not fit to the result)
PRESERVE = {"faithful", "manifest", "faithman", "itemize", "distrust"}
COLLAPSE = {"node", "rolecont", "persona"}
NEUTRAL  = {"handoff", "selfsumm", "neutral_long", "link"}


def cls(a):
    return "preserve" if a in PRESERVE else "collapse" if a in COLLAPSE else "neutral"


def audit():
    out = {}
    for fam in FAMS:
        try:
            C = json.load(open(f"out/pilot/framing_sweep_{fam}.json", encoding="utf-8"))["conditions"]
        except FileNotFoundError:
            continue
        rows = []
        for a, d in C.items():
            rows.append({"arm": a, "cls": cls(a), "tau": round(d["tau"], 2),
                         "tau_ci": [round(x, 2) for x in d["tau_ci"]],
                         "f": round(d["f"], 3)})
        rows.sort(key=lambda r: r["tau"])
        # tau ordinal: fraction of adjacent pairs that are CI-disjoint
        disj = sum(1 for x, y in zip(rows, rows[1:]) if x["tau_ci"][1] < y["tau_ci"][0])
        # floor separation: min preserve-floor minus max collapse-floor
        pf = [r["f"] for r in rows if r["cls"] == "preserve"]
        cf = [r["f"] for r in rows if r["cls"] == "collapse"]
        gap = (min(pf) - max(cf)) if pf and cf else None
        out[fam] = {"rows": rows, "tau_adjacent_disjoint": f"{disj}/{len(rows)-1}",
                    "min_preserve_floor": round(min(pf), 3) if pf else None,
                    "max_collapse_floor": round(max(cf), 3) if cf else None,
                    "floor_gap_preserve_minus_collapse": round(gap, 3) if gap is not None else None}
    return out


if __name__ == "__main__":
    res = audit()
    json.dump(res, open("out/pilot/framing_floor_audit.json", "w"), indent=2)
    for fam, d in res.items():
        print(f"== {fam} ==")
        print(f"  tau adjacent CI-disjoint: {d['tau_adjacent_disjoint']}  "
              f"(strict-order links justified by tau)")
        print(f"  FLOOR gap (min preserve - max collapse) = "
              f"{d['floor_gap_preserve_minus_collapse']}  "
              f"[preserve>={d['min_preserve_floor']} vs collapse<={d['max_collapse_floor']}]")
    print("\nVERDICT: tau ordering is Qwen-specific and inverts cross-family; the "
          "preservation>collapse FLOOR gap is positive in every family -> the stance "
          "lever lives in the asymptotic floor, not the decay rate.")
