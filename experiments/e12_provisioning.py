# PREDICT-THEN-VERIFY PROVISIONING (prereg out/pilot/provisioning_predictions.json).
# Method B: the half-life law as a PREDICTIVE design rule, not a post-hoc fit. Fit the
# survival surface S(k;B) = f + (S0-f)*exp(-k/tau(B)), tau(B) = a*B^beta, on a small
# CALIBRATION grid; then predict, with ZERO free parameters, held-out cells never run
# (budget interpolation B=20, budget extrapolation B=60, depth extrapolation k=8) and
# verify live. Also INVERT the law: solve the budget B* that hits target survival T at
# depth k_t, run B*, check it lands. Memory papers report post-hoc accuracy; a validated
# a-priori provisioning rule is the capability nobody has. Doubles as the first frontier-
# model (Sonnet) measurement of the budget power law (beta was Qwen-only).
import json, os, sys
from statistics import mean
import numpy as np
from scipy.optimize import least_squares
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import run_chain
from run import FILLER, _build_backend

PROBES = int(os.environ.get("PROV_PROBES", "8"))       # per type
NDIST = 7
CAL_BUDGETS = [int(x) for x in os.environ.get("PROV_CAL_B", "15,25,40").split(",")]
CAL_KS = [int(x) for x in os.environ.get("PROV_CAL_K", "0,1,2,4").split(",")]
HELD = [tuple(map(int, c.split(":")))
        for c in os.environ.get("PROV_HELD", "20:2,20:4,60:1,60:2,60:8,25:8").split(",")]
T_TARGET = float(os.environ.get("PROV_T", "0.5"))
K_TARGET = int(os.environ.get("PROV_KT", "4"))


def probe_set():
    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    probes = [f for t in FACT_TYPES for f in bt[t][:PROBES]]
    pool = [f for t in FACT_TYPES for f in bt[t][PROBES:PROBES + 3]]
    return [(p, tuple(d for d in pool if d.fact_id != p.fact_id)[:NDIST]) for p in probes]


def run_cell(backend, pset, B, k):
    vals = []
    for probe, dist in pset:
        r = run_chain(backend, probe, k, "handoff", budget=B, load="light",
                      filler=FILLER, seed=0, distractors=dist)
        vals.append(1 if grade(probe, r, "actionable") else 0)
    return round(mean(vals), 4)


def surface(params, B, k):
    S0, f, a, beta = params
    tau = a * (B ** beta)
    return f + (S0 - f) * np.exp(-np.asarray(k, float) / np.maximum(tau, 1e-6))


def fit_surface(cells):  # cells: {(B,k): s}
    ys = np.array([cells[c] for c in cells], float)
    Bs = np.array([c[0] for c in cells], float); ks = np.array([c[1] for c in cells], float)
    def resid(p):
        return surface(p, Bs, ks) - ys
    sol = least_squares(resid, x0=[0.9, 0.15, 0.05, 1.0],
                        bounds=([0.5, 0.0, 1e-4, 0.05], [1.0, 0.6, 10.0, 3.0]))
    return sol.x


def naive_pred(cal, B, k):
    # nearest calibration cell (nearest B, tie -> larger; then nearest k, tie -> larger)
    bc = min(CAL_BUDGETS, key=lambda b: (abs(b - B), -b))
    kc = min(CAL_KS, key=lambda x: (abs(x - k), -x))
    return cal[(bc, kc)]


def invert_budget(params, T, k):
    S0, f, a, beta = params
    if not (f < T < S0):
        return None
    tau_needed = -k / np.log((T - f) / (S0 - f))
    Bstar = (tau_needed / a) ** (1.0 / beta)
    return int(round(min(max(Bstar, 8), 100)))


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_prov_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)
    pset = probe_set()

    def prog(m):
        with open(f"out/pilot/prov_progress_{tag}.txt", "w") as fh:
            fh.write(m + "\n")

    # 1) calibration grid
    cal = {}
    for B in CAL_BUDGETS:
        for k in CAL_KS:
            cal[(B, k)] = run_cell(backend, pset, B, k)
            prog(f"cal B={B} k={k} done ({len(cal)}/{len(CAL_BUDGETS) * len(CAL_KS)})")
    params = fit_surface(cal)
    S0, f, a, beta = [round(float(x), 4) for x in params]

    # 2) held-out predictions (zero free parameters), then verify live
    rows = []
    for B, k in HELD:
        pred = float(surface(params, B, k))
        nv = naive_pred(cal, B, k)
        obs = run_cell(backend, pset, B, k)
        rows.append({"B": B, "k": k, "pred_law": round(pred, 3), "pred_naive": round(nv, 3),
                     "obs": obs, "err_law": round(abs(pred - obs), 3),
                     "err_naive": round(abs(nv - obs), 3)})
        prog(f"held-out B={B} k={k} done")
    mae_law = round(mean(r["err_law"] for r in rows), 4)
    mae_naive = round(mean(r["err_naive"] for r in rows), 4)

    # 3) provisioning inversion: budget for target T at depth K_TARGET
    Bstar = invert_budget(params, T_TARGET, K_TARGET)
    prov = None
    if Bstar is not None:
        obs = run_cell(backend, pset, Bstar, K_TARGET)
        prov = {"target": T_TARGET, "k": K_TARGET, "B_star": Bstar, "obs": obs,
                "abs_err": round(abs(obs - T_TARGET), 3)}

    res = {"model": model, "n_probes": len(pset), "cal_budgets": CAL_BUDGETS, "cal_ks": CAL_KS,
           "calibration": {f"{b}:{k}": v for (b, k), v in cal.items()},
           "fit": {"S0": S0, "f": f, "a": a, "beta": beta,
                   "tau_at": {str(B): round(a * B ** beta, 3) for B in CAL_BUDGETS + [20, 60]}},
           "held_out": rows, "mae_law": mae_law, "mae_naive": mae_naive,
           "provisioning": prov}
    json.dump(res, open(f"out/pilot/e12_provisioning_{tag}.json", "w"), indent=2)

    print(f"=== PREDICT-THEN-VERIFY PROVISIONING ({tag}, n={len(pset)}/cell) ===")
    print("calibration grid (obs):")
    for B in CAL_BUDGETS:
        print(f"  B={B:>3}: " + "  ".join(f"k{k}={cal[(B, k)]:.2f}" for k in CAL_KS))
    print(f"fit: S0={S0} f={f} a={a} beta={beta}  "
          f"tau: " + ", ".join(f"B{B}->{a * B ** beta:.2f}" for B in CAL_BUDGETS + [20, 60]))
    print("held-out cells (predicted BEFORE running):")
    for r in rows:
        print(f"  B={r['B']:>3} k={r['k']}: law={r['pred_law']:.2f} naive={r['pred_naive']:.2f} "
              f"obs={r['obs']:.2f}  err_law={r['err_law']:.2f} err_naive={r['err_naive']:.2f}")
    print(f"MAE: law={mae_law:.3f}  naive={mae_naive:.3f}  (law must beat naive)")
    if prov:
        print(f"provisioning: law says B*={prov['B_star']} hits S={T_TARGET} at k={K_TARGET}; "
              f"observed {prov['obs']:.2f} (|err|={prov['abs_err']:.2f})")
    else:
        print(f"provisioning: target {T_TARGET} unreachable (floor {f} / S0 {S0})")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
