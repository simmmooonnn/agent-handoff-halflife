# pilot_gate.py — Task 9 go/no-go gate.
# GO iff: handoff survival decays monotonically in k AND fits tau with r2>0.8,
#         AND moat_verdict passes (handoff decays faster than long-context, with
#         the long-context fit itself being well-behaved).
import json, os, sys
from facts import make_facts, chance_level
from run import run_sweep, FILLER, _build_backend
from analyze import (survival_curve, fit_tau, moat_verdict, empirical_chance,
                     compare_decay_models, bootstrap_tau_ci)

def main(provider="openai", model="gpt-4o", n="40", ks="0,1,2,4,8,16",
         budget="120", m_facts="8", load="light"):
    n = int(n); budget = int(budget); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    facts = make_facts(n, seed=0)
    backend = _build_backend(provider, model, f"data/cache_pilot_{model.split('/')[-1]}.json")
    rows = run_sweep(backend, facts, ks=ks_list,
                     conditions=["handoff", "longctx", "verbatim", "nofact"],
                     budget=budget, load=load, filler=FILLER, seeds=[0],
                     level="actionable", m_facts=m_facts)
    emp = empirical_chance(rows)
    # use empirical (no-fact) guess rate as chance; fall back to hand-coded if unavailable
    chance = emp["overall"] if emp["overall"] is not None else \
        sum(chance_level(f.ftype) for f in facts) / len(facts)
    kh, Sh = survival_curve(rows, "handoff")
    kl, Sl = survival_curve(rows, "longctx")
    kv, Sv = survival_curve(rows, "verbatim")
    fh = fit_tau(kh, Sh, chance)
    moat = moat_verdict(rows, chance)
    shape = compare_decay_models(kh, Sh, chance)
    mono = all(x >= y - 0.05 for x, y in zip(Sh, Sh[1:]))
    tau_ci = bootstrap_tau_ci(rows, "handoff", chance)
    verdict = {
        "provider": provider, "model": model, "n": n, "ks": ks_list,
        "budget": budget, "m_facts": m_facts, "load": load,
        "chance": chance, "empirical_chance": emp,
        "handoff": {"ks": kh, "S": Sh, "tau": fh["tau"], "f": fh["f"], "S0": fh["S0"], "r2": fh["r2"],
                    "tau_ci": [tau_ci["lo"], tau_ci["hi"]], "tau_boot_median": tau_ci["tau"]},
        "longctx": {"ks": kl, "S": Sl},
        "verbatim": {"ks": kv, "S": Sv},
        "decay_shape": {"best": shape["best"],
                        "aic": {m: shape["models"][m]["aic"] for m in shape["models"]}},
        "monotonic": mono,
        "moat": moat,
        "GO": bool(mono and fh["r2"] > 0.8 and moat["passes"]),
    }
    os.makedirs("out/pilot", exist_ok=True)
    tag = model.split('/')[-1]
    # dump raw per-fact rows so bootstrap CIs can be recomputed/aggregated offline
    with open(f"out/pilot/rows_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(f"out/pilot/verdict_{tag}.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    return verdict

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
