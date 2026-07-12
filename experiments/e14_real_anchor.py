# REAL-TASK EXTERNAL-VALIDITY ANCHOR (prereg out/pilot/real_anchor_predictions.json).
# The reviewer's #1 objection: every law so far lives on synthetic planted facts +
# filler. This runs the same measurements on REAL-STYLE coherent workplace documents
# (realdocs.py: 6 docs x 8 naturally-embedded facts) under realistic summarize-and-
# hand-off relays. Three arms, k in {0,1,2,4,6}:
#   realistic = uncapped colleague-handoff instruction (what practitioners actually do;
#               tests whether VOLUNTARY compression alone erases facts)
#   budgeted  = 60-word cap (comparable pressure to the synthetic regime)
#   slots     = budgeted + [SLOTS] exemption block holding the 4 MEASURED-most-fragile
#               facts (Sonnet calibration: preference & numeric worst) copied verbatim
import hashlib, json, os, sys
from statistics import mean
import numpy as np
from grade import grade
from run import _build_backend
from realdocs import DOCS

KS = [int(x) for x in os.environ.get("REAL_KS", "0,1,2,4,6").split(",")]
NB = 60                      # budgeted-arm word cap
SLOT_TYPES = ("preference", "numeric")   # measured-most-fragile on Sonnet (e7-med)
RNG = np.random.default_rng(0)


def _q(backend, carry, fact, iid):
    p = f"CARRY: {carry}\nQUESTION: {fact.query}\nAnswer concisely."
    r = backend.generate(p, item_id=f"{iid}:q:{fact.fact_id}", max_tokens=64)
    return 1 if grade(fact, r, "actionable") else 0


def _hop_prompt(carry, arm):
    if arm == "realistic":
        return ("You are handing this work off to a colleague who is taking over from "
                "you. Below is the current context. Write the handoff note they will "
                "receive IN PLACE of this context.\n\n" + carry)
    if arm == "budgeted":
        return (f"CARRY: {carry}\nSummarize, in at most {NB} words, the essential "
                "information a downstream colleague must know. You cannot keep "
                "everything — keep only what matters most. Start your reply with 'CARRY:'.")
    # slots: carry already contains [SLOTS]/[NOTES]
    return ("Compress this running context for the next step. The [SLOTS] section is a "
            "protected record: reproduce EVERY slot line exactly as written — do not "
            "drop, merge, or rephrase any of them. Then compress the [NOTES] section to "
            f"at most {NB - 30} words. Output:\n[SLOTS]\n<all lines unchanged>\n[NOTES]\n"
            "<compressed>\n\n" + carry)


def run_arm(backend, doc, arm, k):
    iid = f"e14:{doc['doc_id']}:{arm}:k{k}"
    if arm == "slots":
        slotted = [f for f in doc["facts"] if f.ftype in SLOT_TYPES]
        rest_txt = doc["text"]
        carry = "[SLOTS]\n" + "\n".join(f.statement for f in slotted) + "\n[NOTES]\n" + rest_txt
    else:
        carry = doc["text"]
    for i in range(k):
        mt = 600 if arm == "realistic" else NB * 4 + 60
        msg = backend.generate(_hop_prompt(carry, arm), item_id=f"{iid}:h{i}", max_tokens=mt)
        carry = msg[: 6000]
    return {f.fact_id: (f.ftype, _q(backend, carry, f, iid)) for f in doc["facts"]}


def ci_mean(vals):
    v = np.asarray(vals, float)
    bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), \
        round(float(np.percentile(bs, 97.5)), 3)


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_real_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    arms = ["realistic", "budgeted", "slots"]
    rows = []
    for doc in DOCS:
        for k in KS:
            for arm in (arms if k > 0 else ["realistic"]):   # k=0 identical across arms
                surv = run_arm(backend, doc, arm, k)
                for fid, (t, s) in surv.items():
                    rows.append({"doc": doc["doc_id"], "arm": arm if k > 0 else "k0",
                                 "k": k, "fid": fid, "ftype": t, "correct": s})
        with open(f"out/pilot/real_progress_{tag}.txt", "w") as fh:
            fh.write(f"done doc {doc['doc_id']}\n")
    with open(f"out/pilot/e14_rows_{tag}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def sel(arm, k, ft=None):
        return [r["correct"] for r in rows
                if r["k"] == k and (r["arm"] == arm or (k == 0 and r["arm"] == "k0"))
                and (ft is None or r["ftype"] == ft)]

    print(f"=== REAL-TASK ANCHOR ({tag}, 6 docs x 8 embedded facts) ===")
    print(f"  {'k':>3} | realistic         budgeted          SLOTS")
    for k in KS:
        cells = []
        for arm in arms:
            m, lo, hi = ci_mean(sel(arm, k))
            cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
        print(f"  {k:>3} | " + "  ".join(cells))
    kmax = max(KS)
    print(f"\nper-type at k={kmax}:")
    for t in ("numeric", "entity", "negation", "preference"):
        line = "  ".join(f"{arm}={mean(sel(arm, kmax, t)):.2f}" for arm in arms)
        print(f"  {t:>11}: {line}")
    # paired slots - budgeted diff at kmax
    bmap = {r["fid"]: r["correct"] for r in rows if r["arm"] == "budgeted" and r["k"] == kmax}
    smap = {r["fid"]: r["correct"] for r in rows if r["arm"] == "slots" and r["k"] == kmax}
    fids = sorted(bmap)
    d = np.array([smap[f] - bmap[f] for f in fids], float)
    bs = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"\nslots - budgeted at k={kmax}: +{d.mean():.3f} "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
          f"{'  CI excl 0' if np.percentile(bs, 2.5) > 0 else ''}")
    j = {"model": model, "ks": KS, "n_facts": len(DOCS) * 8,
         "agg": {arm: {str(k): ci_mean(sel(arm, k)) for k in KS} for arm in arms},
         "slots_minus_budgeted_kmax": round(float(d.mean()), 3)}
    json.dump(j, open(f"out/pilot/e14_real_anchor_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
