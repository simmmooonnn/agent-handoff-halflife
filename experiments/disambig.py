# Disambiguation experiment (reviewer C1, unanimous CRITICAL): is the multi-agent
# HANDOFF loss anything more than repeated lossy compression? Run the SAME compression
# chain under three agent-boundary framings -- neutral (handoff), single-agent
# self-notes (selfsumm), distinct multi-agent personas (persona) -- at matched budget.
#   tau coincide across all three  -> loss is pure iterated compression; agent-boundary
#                                     framing is irrelevant (reframe / controlled finding).
#   persona tau << handoff/selfsumm -> a genuine boundary effect exists.
import json, os, sys
from facts import make_facts
from run import run_sweep, FILLER, _build_backend
from analyze import survival_curve, fit_tau, empirical_chance, bootstrap_tau_ci

def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100",
         ks="0,1,2,4,8,16", budget="25", m_facts="8"):
    n = int(n); budget = int(budget); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    facts = make_facts(n, seed=0)
    backend = _build_backend(provider, model, f"data/cache_pilot_{model.split('/')[-1]}.json")
    # neutral_long = Tier-1 #1 length-matched control: persona-length neutral preamble,
    # no agent identity. Isolates framing CONTENT from prompt LENGTH.
    conds = ["handoff", "selfsumm", "persona", "neutral_long", "nofact"]
    rows = run_sweep(backend, facts, ks=ks_list, conditions=conds, budget=budget,
                     load="light", filler=FILLER, seeds=[0], level="actionable", m_facts=m_facts)
    chance = empirical_chance(rows)["overall"]
    tag = model.split('/')[-1]
    os.makedirs("out/pilot", exist_ok=True)
    with open(f"out/pilot/disambig_rows_{tag}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    res = {"model": model, "n": n, "budget": budget, "m_facts": m_facts, "chance": chance, "conditions": {}}
    for c in ["handoff", "selfsumm", "persona", "neutral_long"]:
        k, S = survival_curve(rows, c)
        fit = fit_tau(k, S, chance)
        ci = bootstrap_tau_ci(rows, c, chance)
        res["conditions"][c] = {"tau": fit["tau"], "tau_ci": [ci["lo"], ci["hi"]],
                                "r2": fit["r2"], "f": fit["f"], "S": S}
        print(f"{c:9s} tau={fit['tau']:.2f}  CI=[{ci['lo']:.2f},{ci['hi']:.2f}]  "
              f"r2={fit['r2']:.2f}  S={[round(x,2) for x in S]}", flush=True)
    json.dump(res, open(f"out/pilot/disambig_{tag}.json", "w"), indent=2)
    # quick verdict
    t = {c: res["conditions"][c]["tau"] for c in res["conditions"]}
    ci = {c: res["conditions"][c]["tau_ci"] for c in res["conditions"]}
    overlap = (ci["handoff"][0] <= ci["persona"][1] and ci["persona"][0] <= ci["handoff"][1])
    print(f"chance={chance:.3f}")
    print(f"VERDICT: handoff vs persona CI {'OVERLAP -> boundary framing irrelevant (pure compression)' if overlap else 'SEPARATE -> genuine boundary effect'}")
    # Tier-1 #1 length control: does the persona collapse survive a length-matched
    # NEUTRAL preamble? nlong ~ handoff (>> persona) => framing content, not length.
    nl_vs_persona_sep = ci["neutral_long"][0] > ci["persona"][1]
    nl_vs_handoff_overlap = (ci["handoff"][0] <= ci["neutral_long"][1] and
                             ci["neutral_long"][0] <= ci["handoff"][1])
    print(f"LENGTH-CONTROL: neutral_long tau={t['neutral_long']:.2f} "
          f"(persona={t['persona']:.2f}, handoff={t['handoff']:.2f})")
    if nl_vs_persona_sep and nl_vs_handoff_overlap:
        print("  -> nlong SEPARATE from persona AND ~ handoff: framing CONTENT is causal, not prompt length. FRAMING LAW LOCKED.")
    elif nl_vs_persona_sep:
        print("  -> nlong SEPARATE from persona but not ~ handoff: framing content matters, some length effect too.")
    else:
        print("  -> nlong ~ persona: collapse is (partly) a PROMPT-LENGTH effect, not agent identity. FRAMING LAW WEAKENED.")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
