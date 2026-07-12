# Round-6 (e): a GENUINE per-instance predictor, replacing the mislabeled aggregate
# Brier (R1-M4: brier.py's only covariate was hop k -- curve reliability, not
# PredictaBoard-style per-instance predictability).
# Predictor: P(correct | arm, ftype, k) estimated on a TRAIN split of facts (Laplace
# smoothing); evaluated on HELD-OUT facts. Baselines: (a) global mean, (b) the per-k
# marginal (= the old aggregate curve). Metrics: Brier, skill vs per-k marginal,
# AUROC; bootstrap CI over held-out facts. All from cached framing_rows (no calls).
import json
import numpy as np

TAG = "Qwen2.5-7B-Instruct"
RNG = np.random.default_rng(0)


def load_rows():
    rows = [json.loads(l) for l in open(f"out/pilot/framing_rows_{TAG}.jsonl",
                                        encoding="utf-8")]
    return [r for r in rows if r["condition"] != "nofact" and r["k"] > 0]


def main():
    rows = load_rows()
    fids = sorted({r["fact_id"] for r in rows}, key=lambda s: int(s[1:]))
    train_f = set(fids[::2])            # deterministic 50/50 split by fact index
    test_f = set(fids) - train_f
    tr = [r for r in rows if r["fact_id"] in train_f]
    te = [r for r in rows if r["fact_id"] in test_f]

    def cellkey(r):
        return (r["condition"], r["ftype"], r["k"])

    # train: per-(arm,ftype,k) rate with Laplace(1,2); back off to (arm,k) then k
    cell, armk, kk = {}, {}, {}
    for r in tr:
        cell.setdefault(cellkey(r), []).append(r["correct"])
        armk.setdefault((r["condition"], r["k"]), []).append(r["correct"])
        kk.setdefault(r["k"], []).append(r["correct"])

    def rate(xs):
        return (sum(xs) + 1) / (len(xs) + 2)

    P_cell = {c: rate(v) for c, v in cell.items()}
    P_armk = {c: rate(v) for c, v in armk.items()}
    P_k = {c: rate(v) for c, v in kk.items()}
    gmean = rate([r["correct"] for r in tr])

    def predict(r):
        return P_cell.get(cellkey(r),
                          P_armk.get((r["condition"], r["k"]), P_k.get(r["k"], gmean)))

    y = np.array([1.0 if r["correct"] else 0.0 for r in te])
    p = np.array([predict(r) for r in te])
    pk = np.array([P_k.get(r["k"], gmean) for r in te])       # aggregate-curve baseline
    pg = np.full_like(y, gmean)

    def brier(pp):
        return float(np.mean((pp - y) ** 2))

    def auroc(pp):
        pos, neg = pp[y == 1], pp[y == 0]
        if not len(pos) or not len(neg):
            return None
        gt = (pos[:, None] > neg[None, :]).mean()
        eq = (pos[:, None] == neg[None, :]).mean()
        return float(gt + 0.5 * eq)

    b, bk, bg = brier(p), brier(pk), brier(pg)
    skill = 1 - b / bk
    # bootstrap over held-out FACTS (cluster bootstrap: all rows of a fact together)
    te_by_f = {}
    for i, r in enumerate(te):
        te_by_f.setdefault(r["fact_id"], []).append(i)
    fkeys = list(te_by_f)
    sk_b, au_b = [], []
    for _ in range(2000):
        idx = np.concatenate([te_by_f[fkeys[j]]
                              for j in RNG.integers(0, len(fkeys), len(fkeys))])
        yy, pp, ppk = y[idx], p[idx], pk[idx]
        bb, bbk = np.mean((pp - yy) ** 2), np.mean((ppk - yy) ** 2)
        sk_b.append(1 - bb / bbk)
        pos, neg = pp[yy == 1], pp[yy == 0]
        if len(pos) and len(neg):
            au_b.append(float((pos[:, None] > neg[None, :]).mean()
                              + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    out = {"tag": TAG, "n_train_facts": len(train_f), "n_test_facts": len(test_f),
           "n_test_rows": len(te),
           "covariates": "arm x ftype x k (Laplace, backoff arm-k -> k -> global)",
           "brier_perinstance": round(b, 4),
           "brier_aggregate_curve_baseline": round(bk, 4),
           "brier_global_baseline": round(bg, 4),
           "skill_vs_aggregate_curve": round(skill, 4),
           "skill_ci": [round(float(np.percentile(sk_b, 2.5)), 4),
                        round(float(np.percentile(sk_b, 97.5)), 4)],
           "auroc": round(auroc(p), 4),
           "auroc_ci": [round(float(np.percentile(au_b, 2.5)), 4),
                        round(float(np.percentile(au_b, 97.5)), 4)],
           "auroc_aggregate_curve": round(auroc(pk), 4),
           "note": ("held-out-fact evaluation; skill>0 with CI>0 means instance covariates "
                    "(arm, ftype) add real predictability beyond the aggregate curve -- the "
                    "honest PredictaBoard-style claim. If skill~0, keep only the curve-"
                    "calibration wording (round-6 C1 fallback).")}
    json.dump(out, open(f"out/pilot/e_perinstance_{TAG}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
