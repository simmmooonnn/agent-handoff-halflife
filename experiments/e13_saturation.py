# FRONTIER BUDGET-SATURATION GENERALITY (prereg out/pilot/saturation_predictions.json).
# e12 found on Sonnet that budget cannot buy retention: B 15->100 (6.7x) moved k>=1
# survival only .12->.28 -- the model voluntarily discards below its word cap, so the
# channel stops binding (the Qwen budget law tau~B^1.0 does NOT transfer). But that is
# ONE model. This runs the same sweep on additional frontier models to decide whether
# "budget cannot buy retention" is a frontier LAW, a capability-graded transition, or a
# Sonnet quirk. Grid B in {15,40,100} x k in {1,2,4} (+ k=0 anchor at B=25), n=32
# probes/cell (8/type, ndist=7 mixed) -- identical regime to e12 for direct comparison.
import json, os, sys
from statistics import mean
from run import _build_backend
from e12_provisioning import probe_set, run_cell

BUDGETS = [int(x) for x in os.environ.get("SAT_B", "15,40,100").split(",")]
KS = [int(x) for x in os.environ.get("SAT_K", "1,2,4").split(",")]


def main(provider="anthropic", model="claude-haiku-4-5"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_sat_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    pset = probe_set()

    def prog(m):
        with open(f"out/pilot/sat_progress_{tag}.txt", "w") as fh:
            fh.write(m + "\n")

    grid = {}
    grid["25:0"] = run_cell(backend, pset, 25, 0)   # S0 anchor (budget-free at k=0)
    for B in BUDGETS:
        for k in KS:
            grid[f"{B}:{k}"] = run_cell(backend, pset, B, k)
            prog(f"B={B} k={k} done")

    kmax = max(KS)
    gain = round(grid[f"{BUDGETS[-1]}:{kmax}"] - grid[f"{BUDGETS[0]}:{kmax}"], 4)
    res = {"model": model, "n_probes": len(pset), "budgets": BUDGETS, "ks": KS,
           "grid": grid,
           "S_kmax_at_Bmax": grid[f"{BUDGETS[-1]}:{kmax}"],
           "gain_kmax_Bmin_to_Bmax": gain,
           "budget_ratio": round(BUDGETS[-1] / BUDGETS[0], 2)}
    json.dump(res, open(f"out/pilot/e13_saturation_{tag}.json", "w"), indent=2)

    print(f"=== BUDGET SATURATION ({tag}, n={len(pset)}/cell) ===")
    print(f"  S0 anchor (k=0): {grid['25:0']:.2f}")
    print(f"  {'B':>5} | " + "  ".join(f"k={k}" for k in KS))
    for B in BUDGETS:
        print(f"  {B:>5} | " + "  ".join(f"{grid[f'{B}:{k}']:.2f}" for k in KS))
    print(f"  k={kmax}: S({BUDGETS[0]})={grid[f'{BUDGETS[0]}:{kmax}']:.2f} -> "
          f"S({BUDGETS[-1]})={grid[f'{BUDGETS[-1]}:{kmax}']:.2f}  "
          f"gain {gain:+.2f} over {res['budget_ratio']}x budget")
    print("  SATURATES if S(kmax,Bmax) < 0.5 and gain < 0.2 (Sonnet ref: .12->.28, gain +.16)")
    print("  BINDS if survival rises Qwen-like (Qwen ref k=4: .19@B8 -> .44@B40, still rising)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
