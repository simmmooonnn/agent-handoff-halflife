# E1 — coupling / externality measurement (path-A go/no-go gate).
#
# Spaced repetition schedules INDEPENDENT memories: reviewing card A does nothing to
# card B. In a budgeted handoff channel that independence should fail: re-injecting
# fact A restates its text into the carry, but the per-hop output budget is FIXED,
# so A crowds the compressed carry and the bystander facts lose effective budget.
# By our own budget law (tau ~ B^beta), their hazard must rise.
#
# Design: chain carries 1 probe + (M-1) same-type distractors. At EVERY hop we
# restate m of the DISTRACTORS verbatim (m = injection load) and probe the BYSTANDER.
#   m=0   -> plain handoff (byte-identical cache items with the pilot runs)
#   m>0   -> if the externality is real, tau_probe(m) falls monotonically in m;
#            quantitatively, B_eff ~ B - m*w (w = words/statement) and the budget law
#            predicts tau(m) ~ tau(0) * ((B - m*w)/B)^beta with the MEASURED beta.
# PASS gate: tau(inj2) CI-separated below tau(inj0)  -> the coupled scheduling
# problem is real; proceed with theory T1-T3 + EA-scheduler.
# FAIL gate: tau flat in m -> externality ~ 0; abandon path A, report as a finding.
import json, os, sys
from facts import make_facts
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend
from analyze import survival_curve, fit_tau, empirical_chance, bootstrap_tau_ci

BETA = 0.82  # measured budget-law exponent (H1), used only for the prediction overlay


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", ks="0,1,2,4",
         budgets="25,16", m_facts="8", loads="0,1,2,4"):
    n = int(n); m_facts = int(m_facts)
    ks_list = [int(x) for x in str(ks).split(",")]
    budget_list = [int(x) for x in str(budgets).split(",")]
    load_list = [int(x) for x in str(loads).split(",")]
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_pilot_{tag}.json")
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    os.makedirs("out/pilot", exist_ok=True)
    out = {"model": model, "n": n, "m_facts": m_facts, "beta_used": BETA, "budgets": {}}

    for budget in budget_list:
        rows = []
        # nofact chance (cached from pilot at the same budget where available)
        for f in facts:
            resp = run_chain(backend, f, 0, "nofact", budget=budget, load="light",
                             filler=FILLER, seed=0)
            rows.append({"condition": "nofact", "k": 0, "fact_id": f.fact_id,
                         "ftype": f.ftype, "seed": 0, "correct": grade(f, resp, "actionable")})
        chance = empirical_chance(rows)["overall"]
        for m in load_list:
            for f in facts:
                pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
                distractors = tuple(pool[:max(0, m_facts - 1)])
                inj = [d.statement for d in distractors[:m]] if m > 0 else None
                for k in ks_list:
                    if m > 0 and k == 0:
                        continue  # identical to m=0, k=0; copied below
                    hops = tuple(range(1, k + 1)) if (m > 0 and k > 0) else ()
                    resp = run_chain(backend, f, k, "handoff", budget=budget, load="light",
                                     filler=FILLER, seed=0, distractors=distractors,
                                     reinject_hops=hops, reinject_stmts=inj)
                    rows.append({"condition": f"inj{m}", "k": k, "fact_id": f.fact_id,
                                 "ftype": f.ftype, "seed": 0,
                                 "correct": grade(f, resp, "actionable")})
            # m>0 shares its k=0 point with m=0 (no hops -> no injection possible)
            if m > 0:
                for r in [r for r in rows if r["condition"] == "inj0" and r["k"] == 0]:
                    rows.append({**r, "condition": f"inj{m}"})
            # incremental dump after each load level
            with open(f"out/pilot/coupling_rows_{tag}_b{budget}.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

        res = {"chance": chance, "loads": {}}
        print(f"=== E1 coupling (budget={budget}, M={m_facts}, n={n}, chance={chance:.3f}) ===")
        # words/statement -> effective-budget prediction from the budget law
        w = sum(len(f.statement.split()) for f in facts) / len(facts)
        tau0 = None
        for m in load_list:
            cond = f"inj{m}"
            kk, S = survival_curve(rows, cond)
            fit = fit_tau(kk, S, chance)
            ci = bootstrap_tau_ci(rows, cond, chance)
            if m == 0:
                tau0 = fit["tau"]
            pred = None
            if m > 0 and tau0:
                b_eff = max(budget - m * w, 1.0)
                pred = tau0 * (b_eff / budget) ** BETA
            res["loads"][m] = {"tau": fit["tau"], "tau_ci": [ci["lo"], ci["hi"]],
                               "r2": fit["r2"], "f": fit["f"], "S": S,
                               "pred_tau_budgetlaw": pred}
            ptxt = f"  pred(budget-law)={pred:.2f}" if pred else ""
            print(f"  inj{m}: tau={fit['tau']:.2f} CI=[{ci['lo']:.2f},{ci['hi']:.2f}] "
                  f"r2={fit['r2']:.2f} S={[round(x,2) for x in S]}{ptxt}", flush=True)
        out["budgets"][budget] = res
        json.dump(out, open(f"out/pilot/coupling_{tag}.json", "w"), indent=2)

        # verdict per budget
        t = {m: res["loads"][m]["tau"] for m in load_list}
        ci = {m: res["loads"][m]["tau_ci"] for m in load_list}
        mono = all(t[a] >= t[b] - 1e-9 for a, b in zip(load_list, load_list[1:]))
        m_test = 2 if 2 in load_list else load_list[-1]
        sep = ci[m_test][1] < ci[0][0]
        print(f"  monotone tau decrease in load: {mono}")
        print(f"  inj{m_test} CI below inj0 CI: {sep} -> "
              f"{'EXTERNALITY CONFIRMED (coupled scheduling is real)' if sep else 'not separated'}")

    print("\nGATE:", "PASS -> proceed with T1-T3 + EA-scheduler"
          if any(out["budgets"][b]["loads"].get(2, out["budgets"][b]["loads"][load_list[-1]])["tau_ci"][1]
                 < out["budgets"][b]["loads"][0]["tau_ci"][0] for b in out["budgets"])
          else "FAIL -> externality ~0; abandon path A, report as finding")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
