# SOTA SHOWDOWN (prereg out/pilot/sota_showdown_predictions.json). Reveal-failure-and-fix:
# on the iterated-summarization axis (numeric fact through K handoffs), compare the
# DEPLOYED SOTA memory strategy (Mem0-style LLM extract+merge each hop -- REGENERATES the
# fact, per SOTA scan arXiv 2504.19413) against the naive free-text baseline and OURS
# (measurement-driven protected slot, budget-EXEMPT). Metric: numeric exact-match survival
# vs hops k -> fit a handoff half-life tau_h per method. Verbatim external store (Zep-
# upper-bound / 2601.00821) survives ~1.0 by construction (analytic reference line).
#
# Faithful Mem0 re-impl (scout, mem0/configs/prompts.py): per hop an LLM EXTRACTs salient
# facts as NL strings and MERGES (add/update/delete) into a bounded running memory -- the
# planted number is routed THROUGH the extractor (regeneration), never copied. That
# regeneration under selection pressure is exactly the drift we test.
import hashlib, json, os, sys
from statistics import mean
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import _planted, _agent_prompt, _truncate, run_chain, _LOAD_REPS
from run import FILLER, _build_backend
from e8_structured_slots import run_slots, NDIST

PROBES = int(os.environ.get("SHOW_PROBES", "12"))
KS = [int(x) for x in os.environ.get("SHOW_KS", "0,2,4,6,8").split(",")]
FTYPES = os.environ.get("SHOW_FTYPES", "numeric").split(",")
budget = int(os.environ.get("SHOW_BUDGET", "25"))
LOAD = "light"


def _mem0_extract_prompt(context, B):
    # paraphrase of Mem0 FACT_RETRIEVAL_PROMPT (Personal Information Organizer)
    return ("You are a Personal Information Organizer that accurately stores facts, memories, "
            "and preferences. Extract every relevant fact from the input below and return them "
            f"as a concise list, one fact per line, at most {B} words total. Keep specific "
            "values, names, and numbers.\nINPUT: " + context + "\nFACTS:")


def _mem0_merge_prompt(memory, new_ctx, B):
    # paraphrase of DEFAULT_UPDATE_MEMORY_PROMPT: ADD/UPDATE/DELETE against existing memory
    return ("You maintain a memory that is a list of facts. Given the current memory and new "
            "information, produce the updated memory: ADD new facts, UPDATE changed values, "
            "DELETE contradicted ones, keep the rest. Return the consolidated fact list, one "
            f"per line, at most {B} words total.\nCURRENT MEMORY:\n{memory}\nNEW INFORMATION: "
            f"{new_ctx}\nUPDATED MEMORY:")


def run_mem0(backend, probe, distractors, K, seed):
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    planted = _planted(probe, distractors, seed)
    iid = "mem0:" + hashlib.sha1(f"{probe.fact_id}|{K}|{budget}|{NDIST}|{seed}".encode()).hexdigest()[:14]
    if K == 0:
        resp = backend.generate(_agent_prompt(planted, "", probe.query), item_id=f"{iid}:q0", max_tokens=64)
        return 1 if grade(probe, resp, "actionable") else 0
    memory = backend.generate(_mem0_extract_prompt(planted, budget), item_id=f"{iid}:e", max_tokens=budget * 3 + 16)
    memory = _truncate(memory, budget * 2)
    for i in range(1, K):  # subsequent handoffs: consolidate the memory (regeneration under budget)
        memory = backend.generate(_mem0_merge_prompt(memory, work.strip(), budget),
                                  item_id=f"{iid}:m{i}", max_tokens=budget * 3 + 16)
        memory = _truncate(memory, budget * 2)
    resp = backend.generate(_agent_prompt(memory, "", probe.query), item_id=f"{iid}:q", max_tokens=64)
    return 1 if grade(probe, resp, "actionable") else 0


def fit_tau(ks, surv):
    # simple free-floor-less exponential fit via least squares on log; robust enough for a summary tau
    import math
    pts = [(k, s) for k, s in zip(ks, surv) if k > 0 and 0 < s < 1]
    if len(pts) < 2:
        return None
    # tau from average decay rate: s ~ exp(-k/tau) -> tau = -k/ln s, aggregate
    taus = [-k / math.log(s) for k, s in pts]
    return round(mean(taus), 2)


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_showdown_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)

    methods = ["free_text", "mem0", "ours_slot"]
    res = {"model": model, "budget": budget, "n_probes": PROBES, "ndist": NDIST,
           "ks": KS, "ftypes": FTYPES, "curves": {}, "tau": {}}
    for t in FTYPES:
        probes = bt[t][:PROBES]
        pool = [f for tt in FACT_TYPES for f in bt[tt][PROBES:PROBES + 3]]
        res["curves"][t] = {m: [] for m in methods}
        for K in KS:
            ft, m0, sl = [], [], []
            import e8_structured_slots as e8
            e8.K = K  # run_slots reads module-global K
            for probe in probes:
                dist = tuple(d for d in pool if d.fact_id != probe.fact_id)[:NDIST]
                if K == 0:
                    r = backend.generate(_agent_prompt(_planted(probe, dist, 0), "", probe.query),
                                         item_id="ft0:" + hashlib.sha1(f"{probe.fact_id}|{NDIST}".encode()).hexdigest()[:12],
                                         max_tokens=64)
                    ft.append(1 if grade(probe, r, "actionable") else 0)
                else:
                    ft.append(1 if grade(probe, run_chain(backend, probe, K, "handoff", budget=budget,
                              load=LOAD, filler=FILLER, seed=0, distractors=dist), "actionable") else 0)
                sc, _ = run_slots(backend, probe, dist, 0, "copy")
                m0.append(run_mem0(backend, probe, dist, K, 0))
                sl.append(sc)
            res["curves"][t]["free_text"].append(round(mean(ft), 3))
            res["curves"][t]["mem0"].append(round(mean(m0), 3))
            res["curves"][t]["ours_slot"].append(round(mean(sl), 3))
            with open(f"out/pilot/showdown_progress_{tag}.txt", "w") as fh:
                fh.write(f"type {t} K={K} done\n")
        res["tau"][t] = {m: fit_tau(KS, res["curves"][t][m]) for m in methods}
    json.dump(res, open(f"out/pilot/e9_sota_showdown_{tag}.json", "w"), indent=2)

    print(f"=== SOTA SHOWDOWN ({tag}, budget={budget}, ndist={NDIST}, probes/type={PROBES}) ===")
    print(f"verbatim external store (Zep-upper-bound / 2601.00821) = 1.00 flat by construction")
    for t in FTYPES:
        print(f"\n  [{t}] survival vs k={KS}")
        for m in methods:
            c = res["curves"][t][m]
            tau = res["tau"][t][m]
            print(f"    {m:>11}: {['%.2f' % x for x in c]}  tau_h={tau}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
