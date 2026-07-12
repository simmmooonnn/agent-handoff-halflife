# TRUE-SCARCITY FINAL SHOWDOWN (prereg out/pilot/true_scarcity_predictions.json). e10
# lost on REPRESENTATION EFFICIENCY, not allocation: our verbatim-sentence slots cost ~5
# words/fact vs Mem0's ~2-word key:value entries -- and at M=12, B=25 Mem0 could fit ALL
# facts (12x2=24<=25), so there was no real scarcity and allocation never mattered. Fixes,
# both licensed by our own laws: (1) e8b says budget-EXEMPTION is the lever and verbatim
# copy is NOT required -> the exempt region can use COMPACT key:value encoding (parity
# with Mem0's representation); (2) true scarcity requires M >> B/2 -> M=28 (every method
# must drop most facts; WHICH ones is now the whole game). Four arms, same per-hop budget:
#   free_text     = compress all M facts into B words of prose                 [naive]
#   mem0          = B-word LLM extract+merge fact list (Mem0 salience)         [SOTA]
#   ledger_folk   = compact exempt ledger, P entries chosen numbers-first      [folk ctrl]
#   ledger_meas   = compact exempt ledger, P entries chosen by MEASURED        [OURS]
#                   per-type fragility (leave-one-chain-out free_text survival)
# ledger_folk isolates allocation from representation: if meas > folk > mem0 the win
# decomposes into (exempt compact representation) + (measurement-driven allocation).
import hashlib, json, os, sys
from statistics import mean
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend
from e9_sota_showdown import _mem0_extract_prompt, _mem0_merge_prompt

M = int(os.environ.get("SC2_M", "28"))
K = int(os.environ.get("SC2_K", "4"))
C = int(os.environ.get("SC2_C", "10"))
budget = int(os.environ.get("SC2_BUDGET", "25"))
P = int(os.environ.get("SC2_P", "10"))         # ledger capacity (entries)
PER_TYPE = M // 4
LOAD = "light"


def to_kv(f):
    # deterministic compact transcoding (no LLM): statement -> "key: value"
    s = f.statement.strip().rstrip(".")
    try:
        if f.ftype in ("numeric", "entity"):
            subj, _, val = s[len("The "):].rpartition(" is ")
            return f"{subj}: {val}"
        if f.ftype == "negation":
            if s.startswith("Under no circumstances modify "):
                return "never-modify: " + s[len("Under no circumstances modify "):]
            if s.startswith("You may freely modify "):
                return "may-modify: " + s[len("You may freely modify "):]
            return s
        aspect, _, choice = s[len("For "):].partition(", the user prefers ")
        return f"{aspect} preference: {choice}"
    except Exception:
        return s


def build_chains():
    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    chains, ptr = [], {t: 0 for t in FACT_TYPES}
    for c in range(C):
        chain = []
        for t in FACT_TYPES:
            for _ in range(PER_TYPE):
                chain.append(bt[t][ptr[t]]); ptr[t] += 1
        chains.append(chain)
    return chains


def _plant(chain, seed):
    import random
    stmts = [f.statement for f in chain]
    random.Random(f"sc2:{seed}").shuffle(stmts)
    return " ".join(stmts)


def _query_all(backend, carry, chain, iid):
    surv = {}
    for f in chain:
        r = backend.generate(_agent_prompt(carry, "", f.query),
                             item_id=f"{iid}:q:{f.fact_id}", max_tokens=64)
        surv[f.fact_id] = (f.ftype, 1 if grade(f, r, "actionable") else 0)
    return surv


def run_free_text(backend, chain, ci):
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    iid = f"s2_ft:{ci}:{hashlib.sha1(_plant(chain,0).encode()).hexdigest()[:10]}"
    carry = _plant(chain, 0)
    for i in range(K):
        p = _agent_prompt(carry, work, None, budget=budget)
        carry = _truncate(backend.generate(p, item_id=f"{iid}:h{i}", max_tokens=budget * 3 + 16), budget)
    return _query_all(backend, carry, chain, iid)


def run_mem0(backend, chain, ci):
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    iid = f"s2_m0:{ci}:{hashlib.sha1(_plant(chain,0).encode()).hexdigest()[:10]}"
    mem = _truncate(backend.generate(_mem0_extract_prompt(_plant(chain, 0), budget),
                                     item_id=f"{iid}:e", max_tokens=budget * 3 + 16), budget * 2)
    for i in range(1, K):
        mem = _truncate(backend.generate(_mem0_merge_prompt(mem, work.strip(), budget),
                                         item_id=f"{iid}:m{i}", max_tokens=budget * 3 + 16), budget * 2)
    return _query_all(backend, mem, chain, iid)


def run_ledger(backend, chain, ci, order_key, label):
    # order_key: callable fact -> sort key ascending; first P get ledger entries
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    ranked = sorted(chain, key=order_key)
    protected, rest = ranked[:P], ranked[P:]
    nb = max(5, budget - 2 * P)  # notes budget after ~2-word entries
    import random
    ns = [f.statement for f in rest]
    random.Random("s2led").shuffle(ns)
    carry = "[LEDGER]\n" + "\n".join(to_kv(f) for f in protected) + "\n[NOTES]\n" + " ".join(ns)
    iid = f"s2_{label}:{ci}:P{P}:{hashlib.sha1(_plant(chain,0).encode()).hexdigest()[:10]}"
    head = ("Compress this running context for the next step. The [LEDGER] section is a "
            "protected record: reproduce EVERY ledger entry exactly as written -- do not "
            "drop, merge, round, or rephrase any entry. Then compress the [NOTES] section "
            f"to at most {nb} words. Output:\n[LEDGER]\n<all entries unchanged>\n[NOTES]\n"
            "<compressed>")
    for i in range(K):
        p = head + "\n" + carry + (f"\nWORK: {work}" if work else "")
        carry = _truncate(backend.generate(p, item_id=f"{iid}:h{i}",
                                           max_tokens=(2 * P + budget) * 3 + 48), budget * 4)
    return _query_all(backend, carry, chain, iid)


def total(surv):
    return mean(s for _, (_, s) in surv.items())


def per_type(survs):
    agg = {}
    for s in survs:
        for _, (t, v) in s.items():
            agg.setdefault(t, []).append(v)
    return {t: round(mean(v), 3) for t, v in sorted(agg.items())}


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_scarcity2_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    chains = build_chains()

    def prog(msg):
        with open(f"out/pilot/true_scarcity_progress_{tag}.txt", "w") as fh:
            fh.write(msg + "\n")

    ft, m0 = [], []
    for ci, ch in enumerate(chains):
        ft.append(run_free_text(backend, ch, ci)); m0.append(run_mem0(backend, ch, ci))
        prog(f"pass1 chain {ci + 1}/{C}")

    def cal_surv(exclude_ci):
        agg = {}
        for ci, s in enumerate(ft):
            if ci == exclude_ci:
                continue
            for _, (t, v) in s.items():
                agg.setdefault(t, []).append(v)
        return {t: mean(v) for t, v in agg.items()}

    lm, lf = [], []
    for ci, ch in enumerate(chains):
        cs = cal_surv(ci)
        lm.append(run_ledger(backend, ch, ci, lambda f: (cs.get(f.ftype, 1.0), f.fact_id), "meas"))
        lf.append(run_ledger(backend, ch, ci,
                             lambda f: (0 if f.ftype == "numeric" else 1, f.fact_id), "folk"))
        prog(f"pass2 chain {ci + 1}/{C}")

    res = {"model": model, "M": M, "K": K, "C": C, "budget": budget, "P": P,
           "total": {"free_text": round(mean(total(s) for s in ft), 3),
                     "mem0": round(mean(total(s) for s in m0), 3),
                     "ledger_folk": round(mean(total(s) for s in lf), 3),
                     "ledger_measured": round(mean(total(s) for s in lm), 3)},
           "per_type": {"free_text": per_type(ft), "mem0": per_type(m0),
                        "ledger_folk": per_type(lf), "ledger_measured": per_type(lm)}}
    res["measured_minus_mem0"] = round(res["total"]["ledger_measured"] - res["total"]["mem0"], 3)
    res["measured_minus_folk"] = round(res["total"]["ledger_measured"] - res["total"]["ledger_folk"], 3)
    json.dump(res, open(f"out/pilot/e11_true_scarcity_{tag}.json", "w"), indent=2)

    print(f"=== TRUE SCARCITY ({tag}, M={M}, K={K}, C={C}, budget={budget}, P={P}) ===")
    print(f"free_text per-type (calibration signal): {res['per_type']['free_text']}")
    print(f"  TOTAL survival over {M} facts:")
    for m in ("free_text", "mem0", "ledger_folk", "ledger_measured"):
        print(f"    {m:>15}: {res['total'][m]:.3f}   {res['per_type'][m]}")
    print(f"  MEASURED - Mem0 = {res['measured_minus_mem0']:+.3f}   "
          f"MEASURED - folk = {res['measured_minus_folk']:+.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
