# Compression-budget sweep at a FIXED model: does the handoff half-life tau scale with
# the per-hop word budget (the channel capacity) rather than with model size?
import json, os, sys
from facts import make_facts
from run import run_sweep, FILLER, _build_backend
from analyze import survival_curve, fit_tau, empirical_chance, bootstrap_tau_ci

def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="40",
         budgets="8,16,25,50,100", ks="0,1,2,4,8,16", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    budget_list = [int(x) for x in str(budgets).split(",")]
    facts = make_facts(n, seed=0)
    backend = _build_backend(provider, model, f"data/cache_pilot_{model.split('/')[-1]}.json")
    nf = run_sweep(backend, facts, ks=[0], conditions=["nofact"], budget=8, load="light",
                   filler=FILLER, seeds=[0], level="actionable", m_facts=m_facts)
    chance = empirical_chance(nf)["overall"]
    res = {"model": model, "n": n, "m_facts": m_facts, "chance": chance, "budgets": []}
    os.makedirs("out/pilot", exist_ok=True)
    tag = model.split('/')[-1]
    rowf = open(f"out/pilot/budget_rows_{tag}.jsonl", "w")
    for b in budget_list:
        rows = run_sweep(backend, facts, ks=ks_list, conditions=["handoff"], budget=b,
                         load="light", filler=FILLER, seeds=[0], level="actionable", m_facts=m_facts)
        for r in rows:
            r = dict(r); r["budget"] = b
            rowf.write(json.dumps(r) + "\n")
        k, S = survival_curve(rows, "handoff")
        fit = fit_tau(k, S, chance)
        ci = bootstrap_tau_ci(rows, "handoff", chance)
        res["budgets"].append({"budget": b, "tau": fit["tau"], "r2": fit["r2"],
                               "S0": fit["S0"], "f": fit["f"], "S": S,
                               "tau_ci": [ci["lo"], ci["hi"]], "tau_boot_median": ci["tau"]})
        # persist incrementally so a walltime timeout still leaves a usable JSON
        json.dump(res, open(f"out/pilot/budget_sweep_{tag}.json", "w"), indent=2)
        print(f"budget={b:4d}  tau={fit['tau']:6.2f}  r2={fit['r2']:.2f}  "
              f"ci=[{ci['lo']:.2f},{ci['hi']:.2f}]  S={[round(x,2) for x in S]}", flush=True)
    rowf.close()
    print("chance(empirical)=", round(chance, 3))

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
