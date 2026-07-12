# R0.2 / E3 -- the FAIR scheduler head-to-head the round-3/4 panels demanded.
#
# sched_live.py compared a_star only to plain uniform, n=40, no CIs -- and the
# round-4 panel showed the -25% "win" reduces to skipping the chance-bound
# negation type ("don't water a dead plant"), a heuristic any baseline may use.
# E3 gives every baseline that heuristic and adds the missing statistics.
#
# Arms (per Sec.8c spec):
#   a_star        per-type period floor(tau*ln((S0-f)/(theta-f)))  (the method)
#   uniform       one period for ALL types, cost-matched to a_star
#   uniform_skip  one period for exactly the types a_star protects (a_star_period
#                 <= K), cost-matched -- THE fair baseline. ROUND-5 FIX: the old
#                 predicate {t: F[t]<theta} wrongly kept negation (F=0.495) in the
#                 schedule at theta=0.5 while a_star skipped it, so the "-25%" was
#                 a denied skip, not a scheduling win. Now both skip identically and
#                 a_star's ONLY residual edge is per-type period differentiation.
#   sm2           expanding intervals (Leitner/SM-2-style doubling p0,2p0,4p0..),
#                 same for all clean types, p0 cost-matched; static because our
#                 chains give no online recall feedback (noted honestly)
#   pin           re-inject every fact every hop (retention ceiling, cost K)
#   never         cost 0 floor
#   faithful      prompt-only mitigation (faithful framing, no re-injection, cost 0)
#
# Metric: STRICT worst-case = min over probed depths k in {2,4,8} of S(k), per
# type; report min and mean over the clean types. Statistics: bootstrap CIs over
# facts (resampled within type) and PAIRED bootstrap for a_star - uniform_skip
# (same resample indices across arms; facts are identical across arms).
#
# PRE-REGISTERED GATE: a_star beats uniform_skip on strict-min with a paired
# 95% CI excluding 0 at >=1 theta at matched cost, OR matches it (paired CI
# lower bound > -0.05) at >=15% lower cost. Otherwise the scheduler is demoted
# to a consumer of the Sec.8d surface, per the demotion text already in Sec.8c.
import json, math, os, sys
import numpy as np
from facts import make_facts
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend

TAU = {"numeric": 0.74, "entity": 1.24, "preference": 2.14, "negation": 2.09}
F = {"numeric": 0.077, "entity": 0.229, "preference": 0.045, "negation": 0.495}
S0 = {"numeric": 0.84, "entity": 1.0, "preference": 0.84, "negation": 0.96}
CLEAN = ["numeric", "entity", "preference"]
PROBE_KS = [2, 4, 8]


def _hops_period(period, K):
    if period is None or period <= 0 or period > K:
        return []
    return list(range(period, K + 1, period))


def _hops_expanding(p0, K):
    # Leitner/SM-2-style doubling intervals: p0, 2*p0, 4*p0, ...
    hops, h, step = [], 0, p0
    while True:
        h += step
        if h > K:
            break
        hops.append(h)
        step *= 2
    return hops


def a_star_period(t, theta):
    f = F[t]
    if theta <= f:
        return None
    a = TAU[t] * math.log((S0[t] - f) / (theta - f))
    return max(1, int(a))


def _cost(sched):
    # restatements per fact, averaged over the 4 types
    return sum(len(v) for v in sched.values()) / len(sched)


def _match_uniform(target, K, types):
    # uniform period over `types` whose avg cost is nearest target (ties -> higher)
    best = None
    for p in range(1, K + 1):
        sched = {t: (_hops_period(p, K) if t in types else []) for t in TAU}
        c = _cost(sched)
        key = (abs(c - target), -c)
        if best is None or key < best[0]:
            best = (key, p, sched, c)
    return best[1], best[2], best[3]


def _match_sm2(target, K, types):
    best = None
    for p0 in range(1, K + 1):
        sched = {t: (_hops_expanding(p0, K) if t in types else []) for t in TAU}
        c = _cost(sched)
        key = (abs(c - target), -c)
        if best is None or key < best[0]:
            best = (key, p0, sched, c)
    return best[1], best[2], best[3]


def _run_arm(backend, facts, by_type, K, sched_of_type, budget, m_facts,
             condition="handoff", seed=0):
    # returns per-type matrix: mat[t] = {fact_id: {k: 0/1}}
    mat = {t: {} for t in by_type}
    for f in facts:
        pool = [g for g in by_type[f.ftype] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:max(0, m_facts - 1)])
        hops = tuple(sched_of_type.get(f.ftype, []))
        row = {}
        for k in PROBE_KS:
            hk = tuple(h for h in hops if h <= k)
            resp = run_chain(backend, f, k, condition, budget=budget, load="light",
                             filler=FILLER, seed=seed, distractors=distractors,
                             reinject_hops=hk)
            row[k] = grade(f, resp, "actionable")
        mat[f.ftype][f.fact_id] = row
    return mat


def _strictmin(mat, types, idx=None):
    # min over probed ks of mean-over-facts, per type; then min/mean over types.
    per_type = {}
    for t in types:
        fids = sorted(mat[t])
        sel = fids if idx is None else [fids[i] for i in idx[t]]
        per_hop = [np.mean([mat[t][f][k] for f in sel]) for k in PROBE_KS]
        per_type[t] = min(per_hop)
    vals = list(per_type.values())
    return min(vals), float(np.mean(vals)), per_type


def _boot(mats, types, n_boot=2000, seed=0):
    # paired bootstrap: same fact-resample indices applied to every arm.
    rng = np.random.default_rng(seed)
    sizes = {t: len(next(iter(mats.values()))[t]) for t in types}
    out = {a: {"min": [], "mean": []} for a in mats}
    diffs = []  # a_star - uniform_skip on strict-min
    for _ in range(n_boot):
        idx = {t: rng.integers(0, sizes[t], size=sizes[t]) for t in types}
        vals = {}
        for a, m in mats.items():
            mn, me, _ = _strictmin(m, types, idx)
            out[a]["min"].append(mn)
            out[a]["mean"].append(me)
            vals[a] = mn
        if "a_star" in vals and "uniform_skip" in vals:
            diffs.append(vals["a_star"] - vals["uniform_skip"])
    ci = {a: {m: [float(np.percentile(v[m], 2.5)), float(np.percentile(v[m], 97.5))]
              for m in v} for a, v in out.items()}
    dci = ([float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
           if diffs else None)
    return ci, dci


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", K="8",
         budget="25", m_facts="8", thetas="0.5,0.35,0.2"):
    n = int(n); K = int(K); budget = int(budget); m_facts = int(m_facts)
    theta_list = [float(x) for x in str(thetas).split(",")]
    facts = make_facts(n, seed=0)
    tag = model.split('/')[-1]
    backend = _build_backend(provider, model, f"data/cache_e3_{tag}.json")
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    os.makedirs("out/pilot", exist_ok=True)
    out = {"model": model, "n": n, "K": K, "budget": budget, "m_facts": m_facts,
           "probe_ks": PROBE_KS, "clean": CLEAN, "runs": [], "static_arms": {}}

    # theta-independent arms (run once; reused across thetas)
    static = {
        "never": ({t: [] for t in TAU}, "handoff"),
        "pin": ({t: list(range(1, K + 1)) for t in TAU}, "handoff"),
        "faithful": ({t: [] for t in TAU}, "faithful"),
    }
    static_mats = {}
    print(f"=== E3 fair head-to-head (n={n}, K={K}, budget={budget}, M={m_facts}, "
          f"probe ks={PROBE_KS}) ===", flush=True)
    for name, (sched, cond) in static.items():
        static_mats[name] = _run_arm(backend, facts, by_type, K, sched, budget,
                                     m_facts, condition=cond)
        mn, me, pt = _strictmin(static_mats[name], CLEAN)
        out["static_arms"][name] = {"cost": _cost(sched), "min": mn, "mean": me,
                                    "per_type": pt}
        print(f"[static] {name:>9} cost={_cost(sched):.1f}  strict-min={mn:.2f}  "
              f"mean={me:.2f}  {pt}", flush=True)

    gate_pass = False
    for theta in theta_list:
        ta_period = {t: a_star_period(t, theta) for t in TAU}
        ta_sched = {t: _hops_period(ta_period[t], K) for t in TAU}
        target = _cost(ta_sched)
        up, u_sched, uc = _match_uniform(target, K, set(TAU))
        # FAIR baseline (round-5 fix): uniform_skip must skip EXACTLY the types
        # a_star declines to protect (a_star_period None or > K), so the two arms
        # make identical skip decisions and a_star's only residual edge is per-type
        # period differentiation among the protected types. The old predicate
        # {t: F[t] < theta} let negation (F=0.495) slip INTO the schedule at
        # theta=0.5 while a_star skipped it -> the "-25%" was the baseline being
        # denied a skip a_star took, not a scheduling win. (See out/review_round5.)
        protected = {t for t in TAU
                     if (ap := a_star_period(t, theta)) is not None and ap <= K}
        skip_types = protected
        usp, us_sched, usc = _match_uniform(target, K, skip_types)
        sp0, s_sched, sc = _match_sm2(target, K, skip_types)
        arms = {"a_star": ta_sched, "uniform": u_sched, "uniform_skip": us_sched,
                "sm2": s_sched}
        mats = {}
        for a, sched in arms.items():
            mats[a] = _run_arm(backend, facts, by_type, K, sched, budget, m_facts)
        mats.update(static_mats)
        ci, dci = _boot(mats, CLEAN)
        row = {"theta": theta, "arms": {}}
        print(f"\n-- theta={theta} --  (a_star target cost {target:.2f}/fact; "
              f"uniform every-{up}, uniform_skip every-{usp} on {sorted(skip_types)}, "
              f"sm2 p0={sp0})", flush=True)
        costs = {"a_star": target, "uniform": uc, "uniform_skip": usc, "sm2": sc,
                 "never": 0.0, "pin": float(K), "faithful": 0.0}
        for a in ("a_star", "uniform_skip", "uniform", "sm2", "pin", "faithful", "never"):
            mn, me, pt = _strictmin(mats[a], CLEAN)
            row["arms"][a] = {"cost": costs[a], "min": mn, "min_ci": ci[a]["min"],
                              "mean": me, "mean_ci": ci[a]["mean"], "per_type": pt,
                              "sched": arms.get(a)}
            print(f"  {a:>12} cost={costs[a]:>4.1f}  strict-min={mn:.2f} "
                  f"[{ci[a]['min'][0]:.2f},{ci[a]['min'][1]:.2f}]  mean={me:.2f} "
                  f"[{ci[a]['mean'][0]:.2f},{ci[a]['mean'][1]:.2f}]", flush=True)
        d = row["arms"]["a_star"]["min"] - row["arms"]["uniform_skip"]["min"]
        row["paired_diff_min_astar_minus_uskip"] = {"point": d, "ci": dci}
        cheaper = target <= 0.85 * usc
        win = (dci[0] > 0) or (dci[0] > -0.05 and cheaper)
        row["gate_this_theta"] = {"ci_beats": dci[0] > 0,
                                  "noninferior_and_cheaper": bool(dci[0] > -0.05 and cheaper),
                                  "win": bool(win)}
        gate_pass = gate_pass or win
        print(f"  PAIRED a_star - uniform_skip strict-min: {d:+.3f} "
              f"[{dci[0]:+.3f},{dci[1]:+.3f}]  costs {target:.2f} vs {usc:.2f} "
              f"-> {'WIN' if win else 'no win'} at this theta", flush=True)
        out["runs"].append(row)
        json.dump(out, open(f"out/pilot/e3_{tag}.json", "w"), indent=2)

    out["gate_pass"] = bool(gate_pass)
    json.dump(out, open(f"out/pilot/e3_{tag}.json", "w"), indent=2)
    v_pass = ("PASS -> a_star CI-separably beats (or matches >=15% cheaper than) "
              "the fair uniform+skip baseline on strict-min: the scheduler's "
              "per-type differentiation earns its keep")
    v_fail = ("FAIL -> scheduler demoted to a consumer of the Sec.8d surface "
              "(demotion text already in Sec.8c); report pin/faithful/never "
              "positions honestly")
    print(f"\nGATE: {v_pass if gate_pass else v_fail}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
