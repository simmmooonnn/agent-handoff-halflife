# UEP-LIVE (prereg out/pilot/uep_live_predictions.json, commit e472c01). End-to-end
# confirmation on a REAL Sonnet pipeline of measurement-driven out-of-band budget
# allocation. C chains each carry M mixed facts through K neutral handoff hops; at
# depth K all M facts are queried -> real per-fact in-band survival. Out-of-band store
# holds protected facts' values (answered correctly by construction). Allocation
# policies chosen with LEAVE-ONE-CHAIN-OUT calibrated per-type survival.
import hashlib, json, os, sys
from statistics import mean
import numpy as np
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import _planted, _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend

M = int(os.environ.get("UEP_M", "24"))
K = int(os.environ.get("UEP_K", "8"))
C = int(os.environ.get("UEP_C", "12"))         # chains
budget = int(os.environ.get("UEP_BUDGET", "25"))
REGIME = os.environ.get("UEP_REGIME", "")      # tag suffix so regimes don't collide
PER_TYPE = M // 4                               # facts of each of the 4 types per chain
PHIS = [0.0, 0.125, 0.25, 0.375, 0.5]
RNG = np.random.default_rng(0)


def build_chains():
    facts = make_facts(400, seed=1)    # plenty; partition into C chains of M
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    chains = []
    ptr = {t: 0 for t in FACT_TYPES}
    for c in range(C):
        chain = []
        for t in FACT_TYPES:
            for _ in range(PER_TYPE):
                chain.append(bt[t][ptr[t]]); ptr[t] += 1
        chains.append(chain)
    return chains


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1] + REGIME
    backend = _build_backend(provider, model, f"data/cache_uep_{tag}.json")
    work = " " + FILLER * _LOAD_REPS["light"]
    os.makedirs("out/pilot", exist_ok=True)
    chains = build_chains()

    def gen(prompt, label, mt):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"uep:{label}:{h}", max_tokens=mt)

    # run each chain, record per-fact in-band survival at depth K
    chain_surv = []          # list over chains of {fid: (ftype, surv 0/1)}
    for ci, chain in enumerate(chains):
        probe = chain[0]
        distract = tuple(chain[1:])
        carry = _planted(probe, distract, seed=0)
        for h in range(1, K + 1):
            p = _agent_prompt(carry, work, None, budget=budget)
            msg = gen(p, f"ch{ci}:h{h}", budget * 3 + 16)
            carry = _truncate(msg, budget)
        surv = {}
        for f in chain:
            resp = gen(_agent_prompt(carry, "", f.query), f"ch{ci}:q:{f.fact_id}", 64)
            surv[f.fact_id] = (f.ftype, 1 if grade(f, resp, "actionable") else 0)
        chain_surv.append(surv)
        with open(f"out/pilot/uep_live_progress_{tag}.txt", "w") as fh:
            fh.write(f"chain {ci + 1}/{C}\n")

    # allocation evaluation with leave-one-chain-out per-type calibration
    def cal_type_surv(exclude_ci):
        agg = {}
        for ci, surv in enumerate(chain_surv):
            if ci == exclude_ci:
                continue
            for fid, (t, s) in surv.items():
                agg.setdefault(t, []).append(s)
        return {t: mean(v) for t, v in agg.items()}

    def eval_policy(policy, phi):
        vals = []
        for ci, surv in enumerate(chain_surv):
            items = list(surv.items())                       # [(fid,(t,s))]
            b = round(phi * len(items))
            if policy == "none":
                order = []
            elif policy == "random":
                # handled separately (averaged)
                order = None
            elif policy == "folk_numbers":
                order = sorted(range(len(items)),
                               key=lambda i: (0 if items[i][1][0] == "numeric" else 1, items[i][0]))
            elif policy == "measured_greedy":
                cs = cal_type_surv(ci)
                order = sorted(range(len(items)),
                               key=lambda i: (cs.get(items[i][1][0], 1.0), items[i][0]))
            else:  # oracle
                order = sorted(range(len(items)), key=lambda i: (items[i][1][1], items[i][0]))
            if policy == "random":
                sub = []
                for _ in range(200):
                    pr = set(RNG.choice(len(items), size=b, replace=False).tolist()) if b else set()
                    sub.append(mean(1 if i in pr else items[i][1][1] for i in range(len(items))))
                vals.append(mean(sub))
            else:
                pr = set(order[:b])
                vals.append(mean(1 if i in pr else items[i][1][1] for i in range(len(items))))
        return mean(vals)

    policies = ["none", "random", "folk_numbers", "measured_greedy", "oracle"]
    res = {"model": model, "M": M, "K": K, "C": C, "phis": PHIS, "curves": {}}
    # overall per-type survival (the load-bearing ranking)
    agg = {}
    for surv in chain_surv:
        for fid, (t, s) in surv.items():
            agg.setdefault(t, []).append(s)
    res["per_type_survival"] = {t: round(mean(v), 3) for t, v in sorted(agg.items())}
    for phi in PHIS:
        res["curves"][f"{phi:.3f}"] = {p: round(eval_policy(p, phi), 3) for p in policies}
    phis = [f"{p:.3f}" for p in PHIS]
    res["measured_minus_folk_avg"] = round(mean(res["curves"][p]["measured_greedy"]
                                                - res["curves"][p]["folk_numbers"] for p in phis), 3)
    res["measured_minus_random_avg"] = round(mean(res["curves"][p]["measured_greedy"]
                                                  - res["curves"][p]["random"] for p in phis), 3)
    res["measured_gap_to_oracle_avg"] = round(mean(res["curves"][p]["oracle"]
                                                   - res["curves"][p]["measured_greedy"] for p in phis), 3)
    json.dump(res, open(f"out/pilot/e7_uep_live_{tag}.json", "w"), indent=2)

    print(f"=== UEP-LIVE ({tag}, M={M}, K={K}, C={C}) ===")
    print(f"per-type survival (in-band, no protection): {res['per_type_survival']}")
    print(f"  {'phi':>5} | none  rand  folk  MEAS  orac")
    for p in phis:
        c = res["curves"][p]
        print(f"  {p:>5} | {c['none']:.2f}  {c['random']:.2f}  {c['folk_numbers']:.2f}  "
              f"{c['measured_greedy']:.2f}  {c['oracle']:.2f}")
    print(f"avg: MEAS-folk={res['measured_minus_folk_avg']:+.3f}  "
          f"MEAS-rand={res['measured_minus_random_avg']:+.3f}  "
          f"oracle-MEAS={res['measured_gap_to_oracle_avg']:+.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
