# E6 -- DIVERSITY-CODED HANDOFFS (prereg out/pilot/diversity_coding_predictions.json,
# commit db82812). Run R INDEPENDENT parallel summarization chains at GEN_TEMP>0, then
# fact-level UNION / MAJORITY decode at depth K. Tests whether cross-chain redundancy
# recovers NUMBERS where single-chain in-band methods plateau. The make-or-break is
# whether cross-chain erasures are INDEPENDENT: observed union vs the independence
# prediction 1-(1-p)^R.
#
# Each chain uses pin_oracle-style re-injection (facts 0..B_ri-1 every hop) + NEUTRAL
# compression. Chains are made independent by folding the chain index into the cache
# key, so identical hop-1 prompts still draw distinct temp>0 samples.
import hashlib, json, os, re, sys
from collections import Counter
import numpy as np
from facts import make_facts
from grade import grade
from relay import _planted, _agent_prompt, _truncate, _LOAD_REPS
from run import FILLER, _build_backend

M, B_ri, K, budget = 24, 4, 16, 25
PROBE_KS = [8, 16]
R_VALUES = [1, 2, 3, 5, 8]
R_MAX = 8
TYPES = ["numeric", "entity"]
N_PER_TYPE = {"numeric": 25, "entity": 12}


def norm(resp, ftype):
    if resp is None:
        return ""
    if ftype == "numeric":
        m = re.search(r"\d+", resp)
        return m.group(0) if m else ""
    # entity: first alphabetic token, lowercased
    m = re.search(r"[A-Za-z]+", resp)
    return m.group(0).lower() if m else ""


def truth(fact):
    return fact.answer if fact.ftype == "numeric" else fact.answer.lower()


def main(provider="anthropic", model="claude-sonnet-4-6"):
    assert os.environ.get("GEN_TEMP", "0") not in ("0", ""), "set GEN_TEMP>0 for diversity"
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_dc_{tag}.json")
    facts = make_facts(100, seed=0)
    by_type = {}
    for f in facts:
        by_type.setdefault(f.ftype, []).append(f)
    types_cycle = sorted(by_type)
    work = " " + FILLER * _LOAD_REPS["light"]
    os.makedirs("out/pilot", exist_ok=True)

    def mixed_distractors(probe):
        out = []
        pools = {t: [g for g in by_type[t] if g.fact_id != probe.fact_id] for t in types_cycle}
        ptr = {t: 0 for t in types_cycle}; ti = 0
        while len(out) < M - 1:
            t = types_cycle[ti % len(types_cycle)]
            if ptr[t] < len(pools[t]):
                out.append(pools[t][ptr[t]]); ptr[t] += 1
            ti += 1
            if all(ptr[t] >= len(pools[t]) for t in types_cycle):
                break
        return tuple(out[: M - 1])

    def gen(prompt, label, mt):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"dc:{label}:{h}", max_tokens=mt)

    # answers[ftype][fid][k] = list over R_MAX chains of normalized answer string
    answers = {t: {} for t in TYPES}
    done = 0
    total = sum(N_PER_TYPE.values())
    for t in TYPES:
        for probe in by_type[t][: N_PER_TYPE[t]]:
            distractors = mixed_distractors(probe)
            chain0 = [probe] + list(distractors)
            stmts = [g.statement for g in chain0]
            per_k = {k: [] for k in PROBE_KS}
            for c in range(R_MAX):
                carry = _planted(probe, distractors, seed=0)
                for h in range(1, K + 1):
                    carry = " ".join(stmts[i] for i in range(B_ri)) + " " + carry
                    p = _agent_prompt(carry, work, None, budget=budget)
                    # chain index c folded into label -> independent temp>0 draw even
                    # when the hop-1 prompt is identical across chains.
                    msg = gen(p, f"c{c}:{probe.fact_id}:h{h}", budget * 3 + 16)
                    carry = _truncate(msg, budget)
                    if h in PROBE_KS:
                        q = _agent_prompt(carry, "", probe.query)
                        resp = gen(q, f"c{c}:{probe.fact_id}:q{h}", 64)
                        per_k[h].append(norm(resp, t))
            answers[t][probe.fact_id] = per_k
            done += 1
            with open(f"out/pilot/dc_progress_{tag}.txt", "w") as fh:
                fh.write(f"done {done}/{total}\n")

    # decode metrics
    def metrics_for(t, k, R):
        singles, unions, majs = [], [], []
        for fid, per_k in answers[t].items():
            fact = next(f for f in by_type[t] if f.fact_id == fid)
            tr = truth(fact)
            chains = per_k[k][:R]
            singles.append(np.mean([1 if a == tr else 0 for a in chains]))
            unions.append(1 if any(a == tr for a in chains) else 0)
            nonempty = [a for a in chains if a]
            if nonempty:
                win, _ = Counter(nonempty).most_common(1)[0]
                majs.append(1 if win == tr else 0)
            else:
                majs.append(0)
        return float(np.mean(singles)), float(np.mean(unions)), float(np.mean(majs))

    res = {"model": model, "temp": os.environ.get("GEN_TEMP"), "K": K, "M": M, "B_ri": B_ri,
           "R_values": R_VALUES, "probe_ks": PROBE_KS, "n_per_type": N_PER_TYPE, "by_type": {}}
    for t in TYPES:
        res["by_type"][t] = {}
        for k in PROBE_KS:
            p1 = metrics_for(t, k, 1)[0]     # single-chain survival p
            row = {"single_p": round(p1, 3), "R": {}}
            for R in R_VALUES:
                s, u, mj = metrics_for(t, k, R)
                indep_pred = 1 - (1 - p1) ** R
                row["R"][R] = {"single_mean": round(s, 3), "union": round(u, 3),
                               "majority": round(mj, 3),
                               "indep_pred_union": round(indep_pred, 3),
                               "union_minus_pred": round(u - indep_pred, 3)}
            res["by_type"][t][f"k{k}"] = row
    json.dump(res, open(f"out/pilot/e6_diversity_{tag}.json", "w"), indent=2)

    # verdict on numeric @ deepest K, top R (the decisive case)
    Rtop = max(R_VALUES)
    num = res["by_type"]["numeric"][f"k{max(PROBE_KS)}"]
    u8 = num["R"][Rtop]["union"]; pred8 = num["R"][Rtop]["indep_pred_union"]; p1 = num["single_p"]
    gap = u8 - pred8
    if u8 >= 0.80 and gap >= -0.10:
        verdict = f"METHOD_WORKS: numeric union@R{Rtop}={u8:.2f} >=0.80, tracks independence pred {pred8:.2f} (gap {gap:+.2f})"
    elif u8 >= 0.50:
        verdict = f"PARTIAL: numeric union@R{Rtop}={u8:.2f} (single p={p1:.2f}), below full independence (pred {pred8:.2f}, gap {gap:+.2f})"
    else:
        verdict = f"IMPOSSIBILITY: numeric union@R{Rtop}={u8:.2f} ~ single p={p1:.2f}, flat in R -> cross-chain erasures CORRELATED"
    res["verdict"] = verdict
    json.dump(res, open(f"out/pilot/e6_diversity_{tag}.json", "w"), indent=2)

    print(f"=== E6 DIVERSITY-CODED HANDOFFS ({tag}, temp={res['temp']}) ===")
    for t in TYPES:
        for k in PROBE_KS:
            row = res["by_type"][t][f"k{k}"]
            print(f"  [{t} @K={k}] single p={row['single_p']:.2f}")
            for R in R_VALUES:
                d = row["R"][R]
                print(f"      R={R}: single={d['single_mean']:.2f} union={d['union']:.2f} "
                      f"majority={d['majority']:.2f}  indep_pred={d['indep_pred_union']:.2f} "
                      f"(gap {d['union_minus_pred']:+.2f})")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
