# Round-6 (f): the pending paired-over-facts CIs, from per-fact rows (no new calls).
#  1. FLOOR-GAP CIs (Moat-2 error bars): bootstrap facts -> refit each arm's floor f
#     -> preserve-min minus collapse-max gap, per family. The spine's +0.21/+0.32/
#     +0.24 gaps get CIs.
#  2. Budget beta paired CI (C8): resample the SAME fact indices across all budget
#     cells (paired -- preserves cross-cell correlation), refit tau per budget, log-log
#     slope. Uses the Anvil-reconstructed rows (budget_rows_reconstructed_*).
#  3. Grid beta,mu paired CIs (P6): joint fit log tau = c + beta logB + mu logM on
#     M={4,8,16} x budgets={8,16,25,50}; paired resample within each M-panel.
#     Arithmetic null requires beta = -mu -> report CI on beta+mu.
import json
import numpy as np
from analyze import fit_tau

RNG = np.random.default_rng(0)
B = 2000
PRESERVE = {"faithful", "distrust", "manifest", "faithman", "itemize"}
COLLAPSE = {"node", "rolecont", "persona"}
FAMS = {"Qwen2.5-7B-Instruct": 0.13, "Mistral-7B-Instruct-v0.3": None,
        "Phi-3-medium-4k-instruct": None}


def rows_of(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build(rows, key_cond=True):
    # d[cond][fact_id][k] = 0/1
    d = {}
    for r in rows:
        c = r["condition"] if key_cond else r["budget"]
        d.setdefault(c, {}).setdefault(r["fact_id"], {})[r["k"]] = int(r["correct"])
    return d


def S_of(dc, fids, ks):
    return [np.mean([dc[f][k] for f in fids if k in dc[f]]) for k in ks]


def floor_gap_ci(fam):
    # TWO estimands, reported side by side (round-6 honesty):
    #  fitted-floor gap -- the spine's quantity, but the free-floor f is bootstrap-
    #    UNSTABLE on a 4-point k-grid (CIs include 0 in all 3 families) -> reported
    #    as a finding, NOT used as the load-bearing contrast;
    #  S(kmax) gap -- min preserve S(4) minus max collapse S(4), directly observed,
    #    no fit (the fixed-depth move, mirroring C11). This is the robust contrast.
    rows = rows_of(f"out/pilot/framing_rows_{fam}.jsonl")
    chance = FAMS[fam] or np.mean([r["correct"] for r in rows if r["condition"] == "nofact"])
    rows = [r for r in rows if r["condition"] != "nofact"]
    d = build(rows)
    ks = sorted({r["k"] for r in rows})
    kmax = max(ks)
    pres = [c for c in d if c in PRESERVE]
    coll = [c for c in d if c in COLLAPSE]
    gaps_f, gaps_s = [], []
    for _ in range(B):
        floors, s4 = {}, {}
        for c in list(pres) + list(coll):
            fl = list(d[c])
            fids = [fl[j] for j in RNG.integers(0, len(fl), len(fl))]
            S = S_of(d[c], fids, ks)
            s4[c] = S[ks.index(kmax)]
            try:
                floors[c] = fit_tau(ks, S, chance)["f"]
            except Exception:
                floors[c] = np.nan
        pv = [floors[c] for c in pres if not np.isnan(floors[c])]
        cv = [floors[c] for c in coll if not np.isnan(floors[c])]
        if pv and cv:
            gaps_f.append(min(pv) - max(cv))
        gaps_s.append(min(s4[c] for c in pres) - max(s4[c] for c in coll))
    return {"arms_preserve": pres, "arms_collapse": coll, "kmax": kmax,
            "fitted_floor_gap": {
                "point": round(float(np.median(gaps_f)), 3),
                "ci": [round(float(np.percentile(gaps_f, 2.5)), 3),
                       round(float(np.percentile(gaps_f, 97.5)), 3)],
                "excludes_0": bool(np.percentile(gaps_f, 2.5) > 0),
                "note": "free-floor param is bootstrap-unstable on a 4-pt grid"},
            "S4_gap": {
                "point": round(float(np.median(gaps_s)), 3),
                "ci": [round(float(np.percentile(gaps_s, 2.5)), 3),
                       round(float(np.percentile(gaps_s, 97.5)), 3)],
                "excludes_0": bool(np.percentile(gaps_s, 2.5) > 0)}}


def beta_paired_ci():
    try:
        rows = rows_of("out/pilot/budget_rows_reconstructed_Qwen2.5-7B-Instruct.jsonl")
    except FileNotFoundError:
        return {"skipped": "reconstructed rows not fetched yet"}
    rows = [r for r in rows if r["budget"] <= 50]
    d = build(rows, key_cond=False)          # d[budget][fid][k]
    ks = sorted({r["k"] for r in rows})
    budgets = sorted(d)
    fids0 = sorted(d[budgets[0]])
    betas = []
    for _ in range(B):
        idx = RNG.integers(0, len(fids0), len(fids0))     # SAME facts across cells
        taus = []
        for b in budgets:
            fids = [fids0[j] for j in idx]
            S = S_of(d[b], fids, ks)
            try:
                taus.append(fit_tau(ks, S, 0.13)["tau"])
            except Exception:
                taus.append(np.nan)
        if not any(np.isnan(taus)):
            betas.append(np.polyfit(np.log(budgets), np.log(taus), 1)[0])
    return {"budgets": budgets, "beta_point": round(float(np.median(betas)), 3),
            "beta_ci_paired": [round(float(np.percentile(betas, 2.5)), 3),
                               round(float(np.percentile(betas, 97.5)), 3)],
            "n_boot_ok": len(betas)}


def grid_ci():
    srcs = {}
    try:
        srcs[8] = [r for r in rows_of(
            "out/pilot/budget_rows_reconstructed_Qwen2.5-7B-Instruct.jsonl")
            if r["budget"] <= 50]
    except FileNotFoundError:
        pass
    for m, fn in ((4, "mxbudget_M4_rows_Qwen2.5-7B-Instruct"),
                  (16, "mxbudget_M16_rows_Qwen2.5-7B-Instruct")):
        srcs[m] = [r for r in rows_of(f"out/pilot/{fn}.jsonl") if r["budget"] <= 50]
    panels = {m: build(rs, key_cond=False) for m, rs in srcs.items()}
    kss = {m: sorted({r["k"] for r in rs}) for m, rs in srcs.items()}
    coefs = []
    for _ in range(B):
        Bv, Mv, Tv = [], [], []
        ok = True
        for m, d in panels.items():
            budgets = sorted(d)
            fids0 = sorted(d[budgets[0]])
            idx = RNG.integers(0, len(fids0), len(fids0))   # paired within panel
            for b in budgets:
                fids = [fids0[j] for j in idx]
                S = S_of(d[b], fids, kss[m])
                try:
                    t = fit_tau(kss[m], S, 0.13)["tau"]
                except Exception:
                    ok = False
                    break
                Bv.append(b); Mv.append(m); Tv.append(t)
            if not ok:
                break
        if not ok:
            continue
        X = np.vstack([np.log(Bv), np.log(Mv), np.ones(len(Bv))]).T
        c, *_ = np.linalg.lstsq(X, np.log(Tv), rcond=None)
        coefs.append((c[0], c[1]))
    be = np.array([c[0] for c in coefs]); mu = np.array([c[1] for c in coefs])
    s = be + mu
    return {"panels_M": sorted(srcs), "n_boot_ok": len(coefs),
            "beta_ci": [round(float(np.percentile(be, 2.5)), 3),
                        round(float(np.percentile(be, 97.5)), 3)],
            "mu_ci": [round(float(np.percentile(mu, 2.5)), 3),
                      round(float(np.percentile(mu, 97.5)), 3)],
            "beta_plus_mu_ci": [round(float(np.percentile(s, 2.5)), 3),
                                round(float(np.percentile(s, 97.5)), 3)],
            "arithmetic_null_rejected": bool(np.percentile(s, 2.5) > 0
                                             or np.percentile(s, 97.5) < 0)}


if __name__ == "__main__":
    out = {"floor_gap": {fam: floor_gap_ci(fam) for fam in FAMS},
           "budget_beta_paired": beta_paired_ci(),
           "grid_beta_mu": grid_ci()}
    json.dump(out, open("out/pilot/f_paired_ci.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
