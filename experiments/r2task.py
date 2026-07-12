# R2.2 (round-5 CRITICAL C8): the facts are decorative planted strings; does a
# fact's DEATH cause a downstream TASK failure? We make the planted numeric value
# task-CRITICAL: same planted statement, but the final query requires ARITHMETIC on
# the value ("what is {subject} plus 1000?"), which is impossible without recalling
# the fact. We fit tau_recall (recall the raw value) and tau_task (compute value+1000)
# on the SAME chains and compare.
#   If tau_task ~ tau_recall -> recall is a valid proxy for task success (validates the
#     paper's metric; fact-death DOES cause task-failure). GOOD for the paper.
#   If tau_task << tau_recall -> recall over-states task retention; the paper must
#     report task-level tau. Either outcome is pre-registered and informative.
import copy, json, os, sys
from dataclasses import replace
from statistics import mean
from facts import make_facts
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend
from analyze import fit_tau

KS = [0, 1, 2, 4, 8]


THRESH = 5000


def task_fact(f):
    # Make the planted value TASK-CRITICAL via a THRESHOLD COMPARISON (not arithmetic:
    # the +1000 version coupled recall with an arithmetic sub-skill that Qwen-7B fails
    # even at k=0 in an 8-fact context, flooring S_task at 0 and making tau_task
    # uninterpretable). The comparison requires recalling the value; if the fact is
    # gone the model can only guess (chance 0.5). We grade with the yes/no logic
    # (ftype='negation' routes grade() to first-yes/no-token matching).
    subj = f.query.split("What is ", 1)[-1].split("?", 1)[0].strip()
    val = int(f.answer)
    ans = "yes" if val > THRESH else "no"
    return replace(f, ftype="negation",
                   query=f"Is {subj} greater than {THRESH}? Answer yes or no.", answer=ans)


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    tag = model.split("/")[-1]
    all_facts = make_facts(n, seed=0)
    numeric = [f for f in all_facts if f.ftype == "numeric"]
    by_type = {}
    for f in all_facts:
        by_type.setdefault(f.ftype, []).append(f)
    backend = _build_backend(provider, model, f"data/cache_r2task_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)

    rec = {k: [] for k in KS}
    tsk = {k: [] for k in KS}
    for f in numeric:
        pool = [g for g in by_type["numeric"] if g.fact_id != f.fact_id]
        distractors = tuple(pool[:m_facts - 1])
        tf = task_fact(f)
        for k in KS:
            # recall: original query/answer (query_tag="" reuses existing cache keys)
            r_resp = run_chain(backend, f, k, "handoff", budget=25, load="light",
                               filler=FILLER, seed=0, distractors=distractors)
            rec[k].append(grade(f, r_resp, "actionable"))
            # task: SAME planted content + distractors + compression hops, but a
            # DIFFERENT final query -> query_tag="task" keeps its answer off the
            # recall cache key (item_id_for ignores fact.query; without this the
            # task read back the recall answer and S_task was spuriously 0).
            t_resp = run_chain(backend, tf, k, "handoff", budget=25, load="light",
                               filler=FILLER, seed=0, distractors=distractors,
                               query_tag="task")
            tsk[k].append(grade(tf, t_resp, "actionable"))

    S_rec = [mean(rec[k]) for k in KS]
    S_tsk = [mean(tsk[k]) for k in KS]
    n_yes = sum(1 for f in numeric if int(f.answer) > 5000)
    fr = fit_tau(KS, S_rec, 0.001)
    ft = fit_tau(KS, S_tsk, 0.5)   # yes/no task -> chance floor 0.5
    ratio = ft["tau"] / fr["tau"] if fr["tau"] else float("nan")
    aligned = 0.6 <= ratio <= 1.67   # within ~1.7x either way
    out = {"model": model, "n_numeric": len(numeric), "ks": KS,
           "task": "threshold-comparison (>5000, yes/no)", "n_yes": n_yes,
           "S_recall": [round(x, 3) for x in S_rec],
           "S_task": [round(x, 3) for x in S_tsk],
           "tau_recall": round(fr["tau"], 3), "r2_recall": round(fr["r2"], 3),
           "tau_task": round(ft["tau"], 3), "r2_task": round(ft["r2"], 3),
           "tau_task_over_recall": round(ratio, 3), "aligned": bool(aligned),
           "verdict": ("ALIGNED: recall is a valid proxy for task success -- fact-death "
                       "causes task-failure (validates the paper's metric)" if aligned else
                       "MISALIGNED: task tau differs from recall tau; report task-level tau")}
    json.dump(out, open(f"out/pilot/r2task_{tag}.json", "w"), indent=2)
    print(f"=== R2.2 recall-vs-task (numeric, n={len(numeric)}) ===")
    print(f"  S_recall={out['S_recall']}  tau_recall={out['tau_recall']} r2={out['r2_recall']}")
    print(f"  S_task  ={out['S_task']}  tau_task  ={out['tau_task']} r2={out['r2_task']}")
    print(f"  tau_task/tau_recall = {out['tau_task_over_recall']}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
