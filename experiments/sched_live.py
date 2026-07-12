# sched_live.py — LIVE head-to-head: the tau-aware NON-UNIFORM re-injection schedule
# vs a UNIFORM schedule at matched cost, executed on a real model. This closes the
# last gap in the method story: scheduler.py's -24%/Pareto result was computed on the
# FITTED survival curves; here we actually RUN a tau-informed schedule and a uniform
# one live and compare worst-case (min-over-hops) fact survival.
#
# Idea: fact types decay at different measured rates (numeric tau=0.74 fastest ...
# preference/negation ~2.1 slowest; negation is chance-bound at ~0.52 >= theta=0.5 so
# it needs NO re-injection). tau-aware spends a fixed re-injection budget where it is
# needed (fast types), skipping the chance-bound one; uniform spreads it evenly and
# wastes restatements on facts that were fine. Metric = worst-case retention through
# the pipeline (min over probe hops k=1..K), the live analog of the sim's min_i S_i.
import json, os, sys
from facts import make_facts
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend

# measured per-ftype handoff fit (perftype_Qwen2.5-7B-Instruct_handoff.json):
# tau, floor f, and S0 (=S[0]). The floor MATTERS: preference has a slow tau but a
# near-zero floor, so it still dies by K -- a 1/tau heuristic underserves it; the
# a_star rule below (the actual Sec.8b method) uses BOTH tau and f.
TAU = {"numeric": 0.74, "entity": 1.24, "preference": 2.14, "negation": 2.09}
F = {"numeric": 0.077, "entity": 0.229, "preference": 0.045, "negation": 0.495}
S0 = {"numeric": 0.84, "entity": 1.0, "preference": 0.84, "negation": 0.96}
CLEAN = ["numeric", "entity", "preference"]   # negation excluded: chance-bound >= theta


def _hops_period(period, K):
    # re-inject every `period` hops in 1..K; period<=0/inf -> none
    if period is None or period <= 0 or period > K:
        return []
    return list(range(period, K + 1, period))


def a_star_period(t, theta):
    # Sec.8b method: a fact may age at most a* = tau*ln((S0-f)/(theta-f)) before
    # dropping to theta -> re-inject every floor(a*) hops. theta<=f -> never drops
    # below theta -> no re-injection needed (period = infinity).
    import math
    f = F[t]
    if theta <= f:
        return None
    a = TAU[t] * math.log((S0[t] - f) / (theta - f))
    return max(1, int(a))


def _run(backend, facts, K, sched_of_type, budget, m_facts, seed=0):
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    per_hop = {t: {k: [] for k in range(1, K + 1)} for t in by_type}
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:max(0, m_facts - 1)])
        hops = tuple(sched_of_type.get(f.ftype, []))
        for k in range(1, K + 1):
            resp = run_chain(backend, f, k, "handoff", budget=budget, load="light",
                             filler=FILLER, seed=seed, distractors=distractors,
                             reinject_hops=hops)
            per_hop[f.ftype][k].append(grade(f, resp, "actionable"))
    worst = {}
    for t in by_type:
        mins = [(lambda v: sum(v) / len(v) if v else 0.0)(per_hop[t][k]) for k in range(1, K + 1)]
        worst[t] = {"per_hop": mins, "worst": min(mins)}
    return worst


def _wc(res, types):
    return min(res[t]["worst"] for t in types)


def _mn(res, types):
    return sum(res[t]["worst"] for t in types) / len(types)


def _fmt(res):
    return "{" + ", ".join(f"{t}:{res[t]['worst']:.2f}" for t in sorted(res)) + "}"


def _uniform_match(target_cost, K):
    # pick the uniform period whose per-fact cost is closest to target (ties -> the
    # cheaper period is fine; we prefer uniform to have >= cost so any tau-aware win
    # is not just a cost advantage).
    best = None
    for p in range(1, K + 1):
        c = len(_hops_period(p, K))
        if best is None or abs(c - target_cost) < abs(best[1] - target_cost) or \
           (abs(c - target_cost) == abs(best[1] - target_cost) and c > best[1]):
            best = (p, c)
    return best  # (period, cost)


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="40", K="8",
         budget="25", m_facts="8", thetas="0.5,0.3,0.2"):
    n = int(n); K = int(K); budget = int(budget); m_facts = int(m_facts)
    theta_list = [float(x) for x in str(thetas).split(",")]
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_schedlive_{tag}.json")
    out = {"model": model, "K": K, "budget": budget, "m_facts": m_facts,
           "tau": TAU, "f": F, "S0": S0, "clean": CLEAN, "runs": []}
    os.makedirs("out/pilot", exist_ok=True)
    print(f"=== live schedule head-to-head, a_star method (K={K}, budget={budget}, M={m_facts}, n={n}) ===")
    print("metric = worst-case retention (min over probe hops k=1..K); clean types only\n")
    for theta in theta_list:
        ta_period = {t: a_star_period(t, theta) for t in TAU}
        ta_sched = {t: _hops_period(ta_period[t], K) for t in TAU}
        ta_cost = sum(len(ta_sched[t]) for t in TAU) / len(TAU)
        up, uc = _uniform_match(ta_cost, K)
        un_sched = {t: _hops_period(up, K) for t in TAU}
        ta = _run(backend, facts, K, ta_sched, budget, m_facts)
        un = _run(backend, facts, K, un_sched, budget, m_facts)
        row = {"theta": theta,
               "tau_aware": {"cost_per_fact": ta_cost, "period": ta_period, "sched": ta_sched,
                             "worst_clean": _wc(ta, CLEAN), "mean_clean": _mn(ta, CLEAN),
                             "by_type_worst": {t: ta[t]["worst"] for t in ta},
                             "by_type_perhop": {t: ta[t]["per_hop"] for t in ta}},
               "uniform": {"cost_per_fact": float(uc), "period": up, "sched": un_sched,
                           "worst_clean": _wc(un, CLEAN), "mean_clean": _mn(un, CLEAN),
                           "by_type_worst": {t: un[t]["worst"] for t in un},
                           "by_type_perhop": {t: un[t]["per_hop"] for t in un}}}
        out["runs"].append(row)
        json.dump(out, open(f"out/pilot/sched_live_{tag}.json", "w"), indent=2)
        print(f"-- theta={theta} --")
        print(f"  tau-aware period={ta_period}  sched={ta_sched}  (cost {ta_cost:.2f}/fact)")
        print(f"  uniform   every-{up} sched={un_sched[CLEAN[0]]}  (cost {uc:.2f}/fact)")
        print(f"  WORST-CASE (min clean type):  tau-aware={_wc(ta,CLEAN):.2f}  uniform={_wc(un,CLEAN):.2f}")
        print(f"  MEAN over clean types:        tau-aware={_mn(ta,CLEAN):.2f}  uniform={_mn(un,CLEAN):.2f}")
        print(f"  per-type worst  tau-aware={_fmt(ta)}")
        print(f"  per-type worst  uniform  ={_fmt(un)}\n", flush=True)
    wins = sum(1 for r in out["runs"]
               if r["tau_aware"]["worst_clean"] >= r["uniform"]["worst_clean"] - 1e-9
               and r["tau_aware"]["cost_per_fact"] <= r["uniform"]["cost_per_fact"] + 1e-9)
    print(f"VERDICT: tau-aware >= uniform worst-case at <= matched cost in {wins}/{len(out['runs'])} thetas")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
