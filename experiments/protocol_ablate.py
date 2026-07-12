# Protocol-ingredient ablation (go/no-go gate for the protocol pivot).
#
# E1/wordlens established that per-hop loss is instruction-governed SELECTION, not
# capacity eviction. The proposed method is therefore a per-hop handoff PROTOCOL
# (steer the selection), not a re-injection schedule. This ablation asks whether the
# protocol's ingredients add measurable value over the generic `faithful` instruction
# (the reviewers' "folklore prompt" flank):
#   itemize  -- selection steering via output FORMAT (one distinct fact per item)
#   manifest -- selection steering via a keys-only CHECKLIST (what must stay answerable)
#   faithman -- fidelity stance + checklist (the full protocol candidate)
# vs anchors: handoff (neutral), faithful, neutral_long (all cache-shared).
#
# PRE-REGISTERED GATE (metric = S(k=4) with bootstrap CI over facts, NOT tau -- avoids
# top-end censoring): PASS iff manifest or faithman S(4) CI-separated ABOVE faithful
# S(4). Secondary: itemize vs handoff. FAIL -> protocol collapses to `faithful`;
# the pivot narrows to "framing law + its engineering" and we report that honestly.
import json, os, sys
import numpy as np
from facts import make_facts
from run import run_sweep, FILLER, _build_backend
from analyze import survival_curve, fit_tau, empirical_chance, bootstrap_tau_ci

def boot_S_at_k(rows, cond, k, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sub = [r for r in rows if r["condition"] == cond and r["k"] == k]
    fids = sorted({r["fact_id"] for r in sub})
    by = {f: [r["correct"] for r in sub if r["fact_id"] == f] for f in fids}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(fids, size=len(fids), replace=True)
        v = [c for f in draw for c in by[f]]
        vals.append(sum(v) / len(v))
    a = np.array(vals)
    point = sum(v for f in fids for v in by[f]) / sum(len(by[f]) for f in fids)
    return point, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100",
         ks="0,1,2,4", budget="25", m_facts="8"):
    n = int(n); budget = int(budget); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    k_gate = max(ks_list)
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_pilot_{tag}.json")
    conds = ["handoff", "faithful", "neutral_long", "itemize", "manifest", "faithman", "nofact"]
    rows = run_sweep(backend, facts, ks=ks_list, conditions=conds, budget=budget,
                     load="light", filler=FILLER, seeds=[0], level="actionable", m_facts=m_facts)
    chance = empirical_chance(rows)["overall"]
    os.makedirs("out/pilot", exist_ok=True)
    with open(f"out/pilot/protocol_rows_{tag}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    fit_conds = [c for c in conds if c != "nofact"]
    res = {"model": model, "n": n, "budget": budget, "m_facts": m_facts,
           "chance": chance, "k_gate": k_gate, "conditions": {}}
    print(f"=== protocol ablation ({tag}, ks={ks_list}, budget={budget}, M={m_facts}, "
          f"chance={chance:.3f}; gate metric = S({k_gate}) bootstrap CI) ===")
    for c in fit_conds:
        kk, S = survival_curve(rows, c)
        fit = fit_tau(kk, S, chance)
        tci = bootstrap_tau_ci(rows, c, chance)
        p, lo, hi = boot_S_at_k(rows, c, k_gate)
        res["conditions"][c] = {"S": S, "tau": fit["tau"], "tau_ci": [tci["lo"], tci["hi"]],
                                "S_gate": p, "S_gate_ci": [lo, hi]}
        print(f"  {c:12s} S={[round(x,2) for x in S]}  S({k_gate})={p:.2f} [{lo:.2f},{hi:.2f}]  "
              f"tau={fit['tau']:.2f} [{tci['lo']:.2f},{tci['hi']:.2f}]", flush=True)
    json.dump(res, open(f"out/pilot/protocol_ablate_{tag}.json", "w"), indent=2)

    C = res["conditions"]
    def sep_above(a, b):  # a's S-gate CI entirely above b's
        return C[a]["S_gate_ci"][0] > C[b]["S_gate_ci"][1]
    print("--- pre-registered gate ---")
    for arm in ("manifest", "faithman"):
        print(f"  {arm} S({k_gate}) CI above faithful: {sep_above(arm, 'faithful')}")
    print(f"  (secondary) itemize CI above handoff: {sep_above('itemize', 'handoff')}")
    ok = sep_above("manifest", "faithful") or sep_above("faithman", "faithful")
    verdict = ("PASS -> protocol adds value beyond generic fidelity; proceed to E3-prime"
               if ok else
               "FAIL -> protocol collapses to faithful; narrow the pivot, report honestly")
    print(f"GATE: {verdict}")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
