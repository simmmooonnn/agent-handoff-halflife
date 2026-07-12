# E4 -- DROPWATCH: closed-loop drop-detection + adaptive-pinning controller.
#
# WHY (round-6): the tau-aware timing scheduler died 3x (E1/E3/E3') because TIMING is
# the wrong control axis in this channel: loss is front-loaded (saturates by k~4),
# survival has stance-controlled FLOORS, and re-injection is externality-free (E1).
# E3' showed pin_fixed wins under scarcity -- but pin_fixed needs to know WHO is
# vulnerable a priori, and the paper's own newstance FAIL proves vulnerability is not
# text-predictable. DROPWATCH closes that gap: it OBSERVES drops in the carry text
# (deterministic, zero model calls -- R1.4 showed loss is visible in the carry) and
# adaptively repairs/pins the facts that demonstrably need it.
#
# The controller's correctness preconditions are the paper's measured laws:
#   front-loading  -> drops cluster early; watchdog cost is bounded, then quiesces
#   E1 null        -> repairs are safe (no eviction of co-carried facts)
#   slack (~70%)   -> repaired facts fit (channel below capacity)
#   R1.4           -> the detector is valid (death is visible in the carry text)
#
# Regime = E3' (M=24 mixed-type, B_ri=4 slots/hop, K=16, budget=25, neutral handoff).
# All policies run through the SAME closed-loop runner; only the per-hop selection
# rule differs, and every selected fact's statement is EXECUTED (prepended before
# that hop's compression). Slot budget B_ri is a hard per-hop cap for all policies;
# dropwatch may use FEWER slots (event-driven) -- executed cost is reported.
#
# Policies:
#   pin_oracle    always facts {0..B_ri-1}; index 0 IS the probe (the inflated E3'
#                 winner, kept for continuity/upper anchor).
#   pin_fair      fixed seeded-random B_ri subset per chain (probe pinned only by
#                 chance B_ri/M) -- the honest pin baseline dropwatch must beat.
#   random        fresh random B_ri facts per hop.
#   a_star_edf    tau-aware EDF on offline (tau,f) periods (E3' method, continuity).
#   dropwatch     repair currently-ABSENT watchable facts (word-boundary answer match
#                 on the carry, negation facts unwatchable -- honest scope), priority
#                 to recidivists, probe-neutral rotating tiebreak, cap B_ri.
#   dropwatch_esc dropwatch + escalation: a fact needing a 2nd repair is PINNED
#                 (occupies a slot every hop thereafter, pinned set capped at B_ri);
#                 free slots repair fresh drops. Learns pin_fixed's subset online.
#
# PRE-REGISTERED GATE: out/pilot/e4dropwatch_predictions.json (committed before run).
import hashlib, json, math, os, random, sys
import numpy as np
from facts import make_facts
from grade import grade, _wb
from relay import _planted, _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend

TAU = {"numeric": 0.74, "entity": 1.24, "preference": 2.14, "negation": 2.09}
F = {"numeric": 0.077, "entity": 0.229, "preference": 0.045, "negation": 0.495}
S0 = {"numeric": 0.84, "entity": 1.0, "preference": 0.84, "negation": 0.96}
PROBE_KS = [4, 8, 16]
POLICIES = ["pin_oracle", "pin_fair", "random", "a_star_edf", "dropwatch", "dropwatch_esc"]
GATE_BASELINE = "pin_fair"


def a_star_period(t, theta):
    f = F[t]
    if theta <= f:
        return None
    return max(1, int(TAU[t] * math.log((S0[t] - f) / (theta - f))))


class WatchState:
    """Pure controller state -- testable without a model."""
    def __init__(self, M, B_ri, esc):
        self.M, self.B_ri, self.esc = M, B_ri, esc
        self.repairs = [0] * M          # times each fact has been repaired
        self.pinned = []                # escalated facts (order of escalation)

    def select(self, absent, hop):
        """absent: set of currently-absent watchable fact indices. Returns sel list."""
        sel = list(self.pinned)[: self.B_ri]
        free = self.B_ri - len(sel)
        cand = [j for j in sorted(absent) if j not in sel]
        # recidivists first; probe-neutral rotating tiebreak (no index-0 favoritism)
        cand.sort(key=lambda j: (-self.repairs[j], (j + hop) % self.M))
        chosen = cand[:free]
        for j in chosen:
            self.repairs[j] += 1
            if self.esc and self.repairs[j] >= 2 and j not in self.pinned \
                    and len(self.pinned) < self.B_ri:
                self.pinned.append(j)
        return sel + chosen


def policy_select(policy, state, absent, hop, M, B_ri, rng, per, last):
    if policy == "pin_oracle":
        return list(range(B_ri))
    if policy == "pin_fair":
        return state["pin_set"]
    if policy == "random":
        return rng.sample(range(M), B_ri)
    if policy == "a_star_edf":
        slack = [(last[i] + per[i]) - hop for i in range(M)]
        return sorted(range(M), key=lambda i: slack[i])[:B_ri]
    return state["watch"].select(absent, hop)   # dropwatch / dropwatch_esc


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", K="16",
         budget="25", m_facts="24", b_ri="4", theta="0.35"):
    n = int(n); K = int(K); budget = int(budget); M = int(m_facts)
    B_ri = int(b_ri); theta = float(theta)
    probe_ks = [k for k in PROBE_KS if k <= K]   # smoke tests may use K < 16
    facts = make_facts(n, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_e4_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    work = " " + FILLER * _LOAD_REPS["light"]

    def mixed_distractors(probe):
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
        return tuple(out[: M - 1])

    def gen(prompt, label, max_tokens):
        # cache key folds the prompt hash: dynamic histories can never collide, and
        # identical prompts across policies share the cache automatically.
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"e4:{label}:{h}", max_tokens=max_tokens)

    mat = {p: {t: {} for t in types_cycle} for p in POLICIES}
    cost = {p: [] for p in POLICIES}          # executed restatements per hop
    for fi, f in enumerate(facts):
        distractors = mixed_distractors(f)
        chain = [f] + list(distractors)       # index 0 = probe
        stmts = [g.statement for g in chain]
        answers = [_wb(g.answer) for g in chain]
        watchable = [g.ftype != "negation" for g in chain]
        per = [a_star_period(g.ftype, theta) or 10 ** 9 for g in chain]
        for policy in POLICIES:
            rng = random.Random(f"{policy}:{fi}")
            state = {"watch": WatchState(M, B_ri, esc=(policy == "dropwatch_esc")),
                     "pin_set": sorted(rng.sample(range(M), B_ri))}
            last = [0] * M
            absent = set()                    # nothing observed absent before hop 1
            carry = _planted(f, distractors, seed=0)
            row, ncalls = {}, 0
            for h in range(1, K + 1):
                sel = policy_select(policy, state, absent, h, M, B_ri, rng, per, last)
                for i in sel:
                    last[i] = h
                ncalls += len(sel)
                if sel:
                    carry = " ".join(stmts[i] for i in sel) + " " + carry
                p = _agent_prompt(carry, work, None, budget=budget)
                msg = gen(p, f"{policy}:{f.fact_id}:h{h}", budget * 3 + 16)
                carry = _truncate(msg, budget)
                wc = _wb(carry)
                absent = {j for j in range(M) if watchable[j] and answers[j] not in wc}
                if h in probe_ks:
                    q = _agent_prompt(carry, "", f.query)
                    resp = gen(q, f"{policy}:{f.fact_id}:q{h}", 64)
                    row[h] = grade(f, resp, "actionable")
            mat[policy][f.ftype][f.fact_id] = row
            cost[policy].append(ncalls / K)
        with open(f"out/pilot/e4_progress_{tag}.txt", "w") as fh:
            fh.write(f"done fact {fi + 1}/{len(facts)}\n")

    def strictmin(m, idx=None):
        per_type = {}
        for t in types_cycle:
            fids = sorted(m[t])
            sel = fids if idx is None else [fids[i] for i in idx[t]]
            if not sel:
                continue
            per_hop = [np.mean([m[t][fid][k] for fid in sel]) for k in probe_ks]
            per_type[t] = float(min(per_hop))
        return (min(per_type.values()), float(np.mean(list(per_type.values()))), per_type)

    rng = np.random.default_rng(0)
    sizes = {t: len(mat[POLICIES[0]][t]) for t in types_cycle}
    boot = {p: [] for p in POLICIES}
    diff_gate, diff_oracle = [], []
    for _ in range(2000):
        idx = {t: rng.integers(0, sizes[t], size=sizes[t]) for t in types_cycle if sizes[t]}
        vals = {p: strictmin(mat[p], idx)[0] for p in POLICIES}
        for p in POLICIES:
            boot[p].append(vals[p])
        diff_gate.append(vals["dropwatch_esc"] - vals[GATE_BASELINE])
        diff_oracle.append(vals["dropwatch_esc"] - vals["pin_oracle"])
    res = {}
    for p in POLICIES:
        mn, me, pt = strictmin(mat[p])
        res[p] = {"strict_min": round(mn, 3), "mean": round(me, 3),
                  "min_ci": [round(float(np.percentile(boot[p], 2.5)), 3),
                             round(float(np.percentile(boot[p], 97.5)), 3)],
                  "cost_per_hop": round(float(np.mean(cost[p])), 2),
                  "per_type": {t: round(v, 3) for t, v in pt.items()}}
    g = [round(float(np.percentile(diff_gate, q)), 3) for q in (2.5, 97.5)]
    o = [round(float(np.percentile(diff_oracle, q)), 3) for q in (2.5, 97.5)]
    gate = g[0] > 0 and res["dropwatch_esc"]["cost_per_hop"] <= res[GATE_BASELINE]["cost_per_hop"] + 1e-9
    out = {"model": model, "n": n, "K": K, "M": M, "B_ri": B_ri, "theta": theta,
           "budget": budget, "probe_ks": probe_ks, "arms": res,
           "dropwatch_esc_minus_pin_fair": {"point": round(float(np.mean(diff_gate)), 3), "ci": g},
           "dropwatch_esc_minus_pin_oracle": {"point": round(float(np.mean(diff_oracle)), 3), "ci": o},
           "gate_pass": bool(gate)}
    json.dump(out, open(f"out/pilot/e4dropwatch_{tag}.json", "w"), indent=2)
    print(f"=== E4 DROPWATCH (M={M}, B_ri={B_ri}, K={K}) ===")
    for p in POLICIES:
        r = res[p]
        print(f"  {p:>13} strict-min={r['strict_min']:.3f} [{r['min_ci'][0]:.3f},"
              f"{r['min_ci'][1]:.3f}] mean={r['mean']:.3f} cost/hop={r['cost_per_hop']}")
    print(f"\ndropwatch_esc - pin_fair:   {out['dropwatch_esc_minus_pin_fair']['point']:+.3f} {g}")
    print(f"dropwatch_esc - pin_oracle: {out['dropwatch_esc_minus_pin_oracle']['point']:+.3f} {o}")
    print("GATE:", "PASS -> observed-drop adaptive pinning beats a priori pinning at <= cost"
          if gate else "FAIL -> report honestly; pin_fair remains the recommendation")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
