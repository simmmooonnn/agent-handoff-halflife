# R2.1 (round-5 CRITICAL, deepest external-validity threat): does tau_h exist
# WITHOUT a forced tight per-hop word budget? The whole positive story runs on
# 8-50-word forced compression; real frameworks pass state via free-length
# summaries, structured JSON, or append-only scratchpads. If facts survive those
# unforced handoffs, the half-life is SPECIFIC to forced budgeted summarization
# and the title must scope; if they still decay, the effect generalizes.
#
# Arms at n=100, M=8, ks=0,1,2,4,8:
#   handoff@25   forced tight budget (the paper's baseline; tau~1.5, S8~0.23)
#   freesum      "summarize, write as much as you judge" -- NO word cap
#   jsonstate    "emit JSON with a facts[] array" -- structured, NO cap
#   scratch      "append to the running scratchpad, keep every entry" -- no compression
#   verbatim     non-model passthrough (ceiling, ~no loss)
# Unforced arms run at budget=200 so no length limit binds.
#
# PRE-REGISTERED (predictions committed before run, out/pilot/r2unforced_predictions.json):
#   Primary interpretation gate: if jsonstate AND scratch S(8) > 0.5 (>> handoff 0.23)
#   -> the half-life is forced-budget-specific; scope the title. If unforced arms
#   ALSO collapse (S(8) < 0.35) -> effect generalizes beyond forced budgets (stronger).
#   Report tau per arm either way.
import json, os, sys
from statistics import mean
from facts import make_facts
from analyze import fit_tau
from run import run_sweep, FILLER, _build_backend

KS = [0, 1, 2, 4, 8]
FORCED = [("handoff", 25), ("verbatim", 25)]
UNFORCED = [("freesum", 200), ("jsonstate", 200), ("scratch", 200)]


def main(provider="hf", model="Qwen/Qwen2.5-7B-Instruct", n="100", m_facts="8"):
    n = int(n); m_facts = int(m_facts)
    tag = model.split("/")[-1]
    facts = make_facts(n, seed=0)
    backend = _build_backend(provider, model, f"data/cache_r2unf_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    all_rows, results = [], {}
    for cond, budget in FORCED + UNFORCED:
        rows = run_sweep(backend, facts, ks=KS, conditions=[cond], budget=budget,
                         load="light", filler=FILLER, seeds=[0], level="actionable",
                         m_facts=m_facts)
        for r in rows:
            r["label"] = f"{cond}@{budget}"
        all_rows.extend(rows)
        with open(f"out/pilot/r2unforced_rows_{tag}.jsonl", "w") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
        S = [mean([r["correct"] for r in rows if r["k"] == k]) for k in KS]
        chance = 0.13  # global guess-rate floor (no nofact arm in this run)
        try:
            fit = fit_tau(KS, S, chance)
            tau = fit["tau"]; r2 = fit["r2"]
        except Exception:
            tau = r2 = float("nan")
        results[f"{cond}@{budget}"] = {"S": [round(x, 3) for x in S],
                                       "S8": round(S[-1], 3), "tau": tau, "r2": r2}
        print(f"  {cond:>10}@{budget:<3} S={[round(x,2) for x in S]}  tau={tau:.2f}  S8={S[-1]:.2f}",
              flush=True)

    s8 = {k: v["S8"] for k, v in results.items()}
    forced_specific = (s8.get("jsonstate@200", 0) > 0.5 and s8.get("scratch@200", 0) > 0.5)
    generalizes = (s8.get("freesum@200", 1) < 0.35 and s8.get("jsonstate@200", 1) < 0.35)
    verdict = ("FORCED-BUDGET-SPECIFIC: unforced handoffs retain (jsonstate/scratch S8>0.5) "
               "-> scope the title to forced budgeted summarization"
               if forced_specific else
               ("GENERALIZES: unforced handoffs also collapse (S8<0.35) -> effect is not "
                "budget-specific, keep general title" if generalizes else
                "MIXED: report per-arm tau; neither clean interpretation triggered"))
    out = {"model": model, "arms": results, "handoff_baseline_S8": s8.get("handoff@25"),
           "forced_specific": forced_specific, "generalizes": generalizes, "verdict": verdict}
    json.dump(out, open(f"out/pilot/r2unforced_{tag}.json", "w"), indent=2)
    print(f"\nhandoff@25 S8={s8.get('handoff@25')} | freesum={s8.get('freesum@200')} "
          f"jsonstate={s8.get('jsonstate@200')} scratch={s8.get('scratch@200')} "
          f"verbatim={s8.get('verbatim@25')}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
