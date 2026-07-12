# Framing-prompt-variant sweep (Sec.8 caveat -> positive map). The relay persona
# ("you received a handoff from X, write a message for the next teammate") COLLAPSES
# retention (tau~0.26) while the LangGraph-native persona ("emit the state for the
# next node") does NOT -> the framing effect is prompt-dependent. This sweep
# decomposes the persona prompt into interpretable ingredients to map WHICH one
# drives the collapse, holding the compression op fixed:
#   handoff   = neutral baseline (no identity)
#   persona   = full multi-agent persona + received-from + write-for-next  [collapse anchor]
#   node      = functional pipeline node, structured state passing  (LangGraph-like)
#   rolecont  = named role but CONTINUITY (no received-from)  -> isolates role vs discontinuity
#   distrust  = discontinuity + "possibly-incomplete other-agent notes", NO named role
#   faithful  = neutral + explicit fidelity instruction  (prompt-level MITIGATION)
# Verdict per variant: COLLAPSE (tau CI below handoff) vs PRESERVE (>= handoff).
import json, os, sys
from facts import make_facts
from run import run_sweep, FILLER, _build_backend
from analyze import survival_curve, fit_tau, empirical_chance, bootstrap_tau_ci

def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100",
         ks="0,1,2,4", budget="25", m_facts="8", conds=""):
    n = int(n); budget = int(budget); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    facts = make_facts(n, seed=0)
    backend = _build_backend(provider, model, f"data/cache_pilot_{model.split('/')[-1]}.json")
    # handoff/persona/selfsumm/neutral_long/nofact reuse the disambig cache; the four
    # new variants are computed fresh. Same (budget, M, seed) -> anchors hit cache.
    # conds override (R1.1 cross-family replication uses a reduced arm set); default
    # empty -> the original list, byte-identical cache behavior.
    conds = ([c for c in str(conds).split(",") if c] or
             ["handoff", "persona", "selfsumm", "neutral_long",
              "node", "rolecont", "distrust", "faithful", "nofact"])
    if "nofact" not in conds:
        conds = conds + ["nofact"]
    rows = run_sweep(backend, facts, ks=ks_list, conditions=conds, budget=budget,
                     load="light", filler=FILLER, seeds=[0], level="actionable", m_facts=m_facts)
    chance = empirical_chance(rows)["overall"]
    tag = model.split('/')[-1]
    os.makedirs("out/pilot", exist_ok=True)
    with open(f"out/pilot/framing_rows_{tag}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    fit_conds = [c for c in conds if c != "nofact"]
    res = {"model": model, "n": n, "budget": budget, "m_facts": m_facts,
           "chance": chance, "conditions": {}}
    for c in fit_conds:
        k, S = survival_curve(rows, c)
        fit = fit_tau(k, S, chance)
        ci = bootstrap_tau_ci(rows, c, chance)
        res["conditions"][c] = {"tau": fit["tau"], "tau_ci": [ci["lo"], ci["hi"]],
                                "r2": fit["r2"], "f": fit["f"], "S": S}
    # ordered map (low tau = worst retention)
    order = sorted(fit_conds, key=lambda c: res["conditions"][c]["tau"])
    print(f"=== framing map ({tag}, ks={ks_list}, budget={budget}, M={m_facts}, chance={chance:.3f}) ===")
    for c in order:
        d = res["conditions"][c]
        print(f"  {c:12s} tau={d['tau']:.2f}  CI=[{d['tau_ci'][0]:.2f},{d['tau_ci'][1]:.2f}]  "
              f"r2={d['r2']:.2f}  S={[round(x,2) for x in d['S']]}", flush=True)
    # classify each variant vs the neutral (handoff) baseline via CI separation
    hlo, hhi = res["conditions"]["handoff"]["tau_ci"]
    print("--- verdict vs neutral(handoff) baseline ---")
    verdicts = {}
    for c in fit_conds:
        if c == "handoff":
            continue
        clo, chi = res["conditions"][c]["tau_ci"]
        if chi < hlo:
            v = "COLLAPSE (CI below handoff)"
        elif clo > hhi:
            v = "PRESERVE+ (CI above handoff)"
        else:
            v = "~neutral (CI overlaps handoff)"
        verdicts[c] = v
        print(f"  {c:12s} {v}")
    res["verdicts"] = verdicts
    # the interpretive question: does removing the received-from/discontinuity cue
    # (rolecont, node) rescue retention that persona/distrust lose?
    json.dump(res, open(f"out/pilot/framing_sweep_{tag}.json", "w"), indent=2)
    print(f"\nwrote out/pilot/framing_sweep_{tag}.json")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
