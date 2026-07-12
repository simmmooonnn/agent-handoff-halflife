# E3' -- the scheduler-SAVE experiment (round-5). The M=8 E3 regime cannot reward
# a_star: E1 made re-injection externality-free (pin-everything is retention-safe)
# and at theta=0.5 every clean type needs period 1 (= pin), so there is no
# heterogeneity to exploit. E3' creates the regime where a scheduler MUST be
# selective and tau-heterogeneity pays:
#   * M=24 competing MIXED-type facts per chain, per-hop word budget 25 -> you
#     cannot restate all facts; a per-hop RE-INJECTION BUDGET B_ri=4 forces choice.
#   * K=16 so period differences compound.
# Because re-injection is externality-free (E1), a probe's survival depends only on
# WHICH hops IT is restated; a policy over the 24 facts determines the probe's
# restatement hops under the shared B_ri budget. All policies get the SAME B_ri
# (matched cost), so this is a pure retention-per-token comparison.
#
# Policies (all cost = B_ri restatements/hop):
#   a_star_edf   restate the B_ri facts with least slack (deadline - h); deadline =
#                last_restated + a_star_period(type). tau-aware EDF (the method).
#   round_robin  cycle through all 24 facts, B_ri/hop (tau-agnostic, fair).
#   random       B_ri random facts/hop (seeded).
#   pin_fixed    always the same first B_ri facts (can't protect the rest).
#
# PRE-REGISTERED GATE (predictions/gate committed before run):
#   a_star_edf beats the BEST tau-agnostic baseline (max of round_robin/random/
#   pin_fixed) on strict-min-over-probe_ks survival, paired bootstrap 95% CI > 0,
#   on >= 1 family. PASS -> the scheduler earns its keep once selection is forced
#   and tau is heterogeneous (the honest, non-artifact scheduler claim). FAIL ->
#   demote the scheduler to a surface-consumer, as round-4/5 already prepared.
import json, math, os, random, sys
import numpy as np
from facts import make_facts
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend

TAU = {"numeric": 0.74, "entity": 1.24, "preference": 2.14, "negation": 2.09}
F = {"numeric": 0.077, "entity": 0.229, "preference": 0.045, "negation": 0.495}
S0 = {"numeric": 0.84, "entity": 1.0, "preference": 0.84, "negation": 0.96}
PROBE_KS = [4, 8, 16]
POLICIES = ["a_star_edf", "round_robin", "random", "pin_fixed"]
BASELINES = ["round_robin", "random", "pin_fixed"]


def a_star_period(t, theta):
    f = F[t]
    if theta <= f:
        return None
    return max(1, int(TAU[t] * math.log((S0[t] - f) / (theta - f))))


def win_hops(types, K, B_ri, policy, theta, seed):
    # simulate the policy over M facts (index 0 = probe); return hops where the
    # probe wins a re-injection slot.
    M = len(types)
    last = [0] * M
    rng = random.Random(seed)
    rr = 0
    wh = []
    per = [a_star_period(t, theta) or 10 ** 9 for t in types]
    for h in range(1, K + 1):
        if policy == "random":
            sel = rng.sample(range(M), min(B_ri, M))
        elif policy == "round_robin":
            sel = [(rr + i) % M for i in range(B_ri)]; rr = (rr + B_ri) % M
        elif policy == "pin_fixed":
            sel = list(range(min(B_ri, M)))
        else:  # a_star_edf: least slack = most urgent
            slack = [(last[i] + per[i]) - h for i in range(M)]
            sel = sorted(range(M), key=lambda i: slack[i])[:B_ri]
        for i in sel:
            last[i] = h
        if 0 in sel:
            wh.append(h)
    return wh


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", K="16",
         budget="25", m_facts="24", b_ri="4", theta="0.35"):
    n = int(n); K = int(K); budget = int(budget); M = int(m_facts)
    B_ri = int(b_ri); theta = float(theta)
    facts = make_facts(n, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_e3p_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)

    def mixed_distractors(probe):
        # M-1 distractors spanning all types (round-robin over types), excluding probe
        out = []
        pools = {t: [g for g in by_type[t] if g.fact_id != probe.fact_id] for t in types_cycle}
        ptr = {t: 0 for t in types_cycle}
        ti = 0
        while len(out) < M - 1:
            t = types_cycle[ti % len(types_cycle)]
            if ptr[t] < len(pools[t]):
                out.append(pools[t][ptr[t]]); ptr[t] += 1
            ti += 1
            if all(ptr[t] >= len(pools[t]) for t in types_cycle):
                break
        return tuple(out[:M - 1])

    # survival matrix: mat[policy][ftype][fact_id][k] = 0/1
    mat = {p: {t: {} for t in types_cycle} for p in POLICIES}
    for fi, f in enumerate(facts):
        distractors = mixed_distractors(f)
        dtypes = [f.ftype] + [d.ftype for d in distractors]
        for policy in POLICIES:
            wh = win_hops(dtypes, K, B_ri, policy, theta, seed=fi)  # deterministic
            row = {}
            for k in PROBE_KS:
                hk = tuple(h for h in wh if h <= k)
                resp = run_chain(backend, f, k, "handoff", budget=budget, load="light",
                                 filler=FILLER, seed=0, distractors=distractors,
                                 reinject_hops=hk)
                row[k] = grade(f, resp, "actionable")
            mat[policy][f.ftype][f.fact_id] = row
        with open(f"out/pilot/e3prime_progress_{tag}.txt", "w") as fh:
            fh.write(f"done fact {f.fact_id}\n")

    def strictmin(m, idx=None):
        per_type = {}
        for t in types_cycle:
            fids = sorted(m[t])
            sel = fids if idx is None else [fids[i] for i in idx[t]]
            if not sel:
                continue
            per_hop = [np.mean([m[t][fid][k] for fid in sel]) for k in PROBE_KS]
            per_type[t] = float(min(per_hop))
        return (min(per_type.values()), float(np.mean(list(per_type.values()))), per_type)

    # paired bootstrap: a_star_edf - best baseline on strict-min
    rng = np.random.default_rng(0)
    sizes = {t: len(mat["a_star_edf"][t]) for t in types_cycle}
    boot = {p: [] for p in POLICIES}
    diff_vs_best = []
    for _ in range(2000):
        idx = {t: rng.integers(0, sizes[t], size=sizes[t]) for t in types_cycle if sizes[t]}
        vals = {p: strictmin(mat[p], idx)[0] for p in POLICIES}
        for p in POLICIES:
            boot[p].append(vals[p])
        diff_vs_best.append(vals["a_star_edf"] - max(vals[b] for b in BASELINES))
    res = {}
    for p in POLICIES:
        mn, me, pt = strictmin(mat[p])
        res[p] = {"strict_min": round(mn, 3), "mean": round(me, 3),
                  "min_ci": [round(float(np.percentile(boot[p], 2.5)), 3),
                             round(float(np.percentile(boot[p], 97.5)), 3)],
                  "per_type": {t: round(v, 3) for t, v in pt.items()}}
    dci = [round(float(np.percentile(diff_vs_best, 2.5)), 3),
           round(float(np.percentile(diff_vs_best, 97.5)), 3)]
    dpt = float(np.mean(diff_vs_best))
    gate = dci[0] > 0
    out = {"model": model, "n": n, "K": K, "M": M, "B_ri": B_ri, "theta": theta,
           "budget": budget, "probe_ks": PROBE_KS, "arms": res,
           "a_star_minus_best_baseline": {"point": round(dpt, 3), "ci": dci},
           "gate_pass": bool(gate)}
    json.dump(out, open(f"out/pilot/e3prime_{tag}.json", "w"), indent=2)
    print(f"=== E3' scheduler-save (M={M}, B_ri={B_ri}, K={K}, theta={theta}) ===")
    for p in POLICIES:
        print(f"  {p:>12} strict-min={res[p]['strict_min']:.3f} "
              f"[{res[p]['min_ci'][0]:.3f},{res[p]['min_ci'][1]:.3f}]  "
              f"mean={res[p]['mean']:.3f}  per_type={res[p]['per_type']}")
    print(f"\na_star_edf - best baseline: {dpt:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]")
    print("GATE:", "PASS -> tau-aware scheduling earns its keep once selection is "
          "forced (bounded budget) and tau is heterogeneous" if gate else
          "FAIL -> demote scheduler to a surface-consumer")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
