# N2 -- natural-fact survival on a real document-summarization chain (round-6 (c)).
#
# External-validity gap: every headline number rests on TEMPLATED planted facts under
# an ENGINEERED multi-fact pressure regime. N2 tests whether tau_h and the moat exist
# for facts embedded mid-sentence in flowing prose, where the chain's task is the
# natural one (summarize the document), and information pressure arises naturally
# (a ~130-word doc under a 40-word budget).
#
# Corpus: 16 authored FICTIONAL documents x 4 embedded facts (n2corpus.py). Fictional
# specifics guarantee the model cannot answer from parametric memory -- verified
# empirically by the NOFACT arm (gate G3).
#
# Arms:
#   handoff   k lossy re-summarizations of the document (neutral head)
#   faithful  same chain, explicit preservation stance (mitigation transfer, N7-lite)
#   longctx   document + k x filler in ONE context (token-matched, no re-summarization)
#   nofact    query with no document (parametric-memory check / empirical chance)
#   (verbatim passthrough == querying the doc itself == the k=0 anchor, by construction)
#
# PRE-REGISTERED GATES: out/pilot/n2natural_predictions.json (committed before run).
import hashlib, json, os, sys
from statistics import mean
import numpy as np
from facts import Fact
from grade import grade
from n2corpus import DOCS
from run import FILLER, _build_backend
from analyze import fit_tau

KS = [0, 1, 2, 4, 8]
BUDGET = 40


def _prompt(carry, work, ask, budget=None, stance=None):
    parts = [f"CARRY: {carry}"]
    if work:
        parts.append(f"WORK: {work}")
    if ask:
        parts.append(f"QUESTION: {ask}\nAnswer concisely.")
    else:
        head = ""
        if stance == "faithful":
            head = ("Preserve the specific facts, names, and numbers below verbatim; "
                    "do not paraphrase them away. ")
        parts.append(f"{head}Summarize, in at most {budget} words, the essential "
                     f"information a downstream colleague must know. You cannot keep "
                     f"everything — keep only what matters most. Start your reply "
                     f"with 'CARRY:'.")
    return "\n".join(parts)


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_n2_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    work = " " + FILLER

    def gen(prompt, mt=140):
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        return backend.generate(prompt, item_id=f"n2:{h}", max_tokens=mt)

    def truncate(s, budget):
        words = s.split()
        return " ".join(words[: budget * 2])          # loose safety, mirrors relay

    def fobj(d, f):
        return Fact(f["fid"], f["ftype"], "", f["query"], f["answer"])

    rows = []
    for d in DOCS:
        # nofact + k=0 (query the raw doc = the verbatim anchor)
        for f in d["facts"]:
            fa = fobj(d, f)
            r0 = gen(_prompt(d["text"], "", f["query"]), 64)
            rows.append({"arm": "handoff", "k": 0, "fid": f["fid"], "ftype": f["ftype"],
                         "correct": grade(fa, r0, "actionable")})
            rn = gen(_prompt("(no prior information provided)", "", f["query"]), 64)
            rows.append({"arm": "nofact", "k": 0, "fid": f["fid"], "ftype": f["ftype"],
                         "correct": grade(fa, rn, "actionable")})
        # longctx: doc + k x filler in one context
        for k in KS[1:]:
            big = work * k * 3
            for f in d["facts"]:
                fa = fobj(d, f)
                r = gen(_prompt(d["text"], big, f["query"]), 64)
                rows.append({"arm": "longctx", "k": k, "fid": f["fid"],
                             "ftype": f["ftype"], "correct": grade(fa, r, "actionable")})
        # compression chains
        for arm in ("handoff", "faithful"):
            carry = d["text"]
            for h in range(1, max(KS) + 1):
                msg = gen(_prompt(carry, work, None, budget=BUDGET,
                                  stance=arm if arm != "handoff" else None))
                carry = truncate(msg, BUDGET)
                if h in KS:
                    for f in d["facts"]:
                        fa = fobj(d, f)
                        r = gen(_prompt(carry, "", f["query"]), 64)
                        rows.append({"arm": arm, "k": h, "fid": f["fid"],
                                     "ftype": f["ftype"],
                                     "correct": grade(fa, r, "actionable")})
    with open(f"out/pilot/n2_rows_{tag}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def S(arm, k, types=("numeric", "entity", "preference")):
        # gates use the checkable types only: negation answers are all "no", so a
        # yes/no guesser scores ~1.0 on them -- they cannot certify non-parametricity
        # (G3) and their 0.5 guess floor compresses drops (G1/G2). Negation curves
        # are reported separately as secondary.
        v = [r["correct"] for r in rows if r["arm"] == arm and r["k"] == k
             and r["ftype"] in types]
        return mean(v) if v else None

    chance = S("nofact", 0)
    chance_neg = S("nofact", 0, types=("negation",))
    res = {"model": model, "n_docs": len(DOCS), "n_facts": 4 * len(DOCS),
           "budget": BUDGET, "ks": KS, "chance_nofact": round(chance, 3),
           "chance_nofact_negation_secondary": round(chance_neg, 3),
           "gate_types": "numeric/entity/preference (48 probes); negation secondary"}
    hS = [S("handoff", k) for k in KS]
    fS = [S("handoff", 0)] + [S("faithful", k) for k in KS[1:]]
    lS = [S("handoff", 0)] + [S("longctx", k) for k in KS[1:]]
    fit_h = fit_tau(KS, hS, chance)
    res["handoff"] = {"S": [round(x, 3) for x in hS],
                      "tau": round(fit_h["tau"], 3), "r2": round(fit_h["r2"], 3),
                      "f": round(fit_h["f"], 3)}
    res["faithful"] = {"S": [round(x, 3) for x in fS]}
    negS = [S("handoff", k, types=("negation",)) for k in KS]
    res["handoff_negation_secondary"] = {"S": [round(x, 3) for x in negS]}
    res["longctx"] = {"S": [round(x, 3) for x in lS]}
    drop_h = hS[0] - hS[-1]
    drop_l = lS[0] - lS[-1]
    res["gates"] = {
        "G1_decay_exists": bool(drop_h >= 0.20),
        "G2_moat": bool(drop_h - drop_l >= 0.20),
        "G3_facts_not_parametric": bool(chance <= 0.15),
        "G4_faithful_lifts_deep_k": bool(fS[-1] - hS[-1] >= 0.10)}
    res["verdict"] = ("NATURAL PASS: tau_h and the moat exist for natural prose facts"
                      if all([res["gates"]["G1_decay_exists"], res["gates"]["G2_moat"],
                              res["gates"]["G3_facts_not_parametric"]])
                      else "check gates -- see JSON")
    json.dump(res, open(f"out/pilot/n2natural_{tag}.json", "w"), indent=2)
    print(f"=== N2 natural-fact (docs={len(DOCS)}, budget={BUDGET}) ===")
    print(f"  chance(nofact) = {res['chance_nofact']}")
    print(f"  handoff  S = {res['handoff']['S']}  tau={res['handoff']['tau']} "
          f"r2={res['handoff']['r2']} f={res['handoff']['f']}")
    print(f"  faithful S = {res['faithful']['S']}")
    print(f"  longctx  S = {res['longctx']['S']}")
    print("  gates:", res["gates"])
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
