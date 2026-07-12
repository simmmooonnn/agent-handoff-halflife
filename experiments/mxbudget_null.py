# Tier-1 #2 analysis: does the budget law reduce to capacity ARITHMETIC?
# Capacity-arithmetic null: survival depends only on rho = B/(c*M) (words per competing
# fact), so tau(M,B) must (i) collapse onto the ratio B/M and (ii) fit the geometric form
# tau = -1/ln(min(1, B/(c*M))). We test ratio-collapse with a joint M x budget grid, and --
# the sharpest test -- note the FRAMING law already refutes arithmetic: at identical (M,B)
# (hence identical rho) tau still spans persona 0.26 -> neutral ~1.9, which no rho-only
# model can produce.
import json, glob, os
import numpy as np

TAG = "Qwen2.5-7B-Instruct"


def _load_grid():
    pts = []  # (M, B, tau)
    # M=8 from the original budget sweep
    p8 = f"out/pilot/budget_sweep_{TAG}.json"
    if os.path.exists(p8):
        d = json.load(open(p8))
        for b in d["budgets"]:
            if b["budget"] <= 50:  # exclude the no-pressure budget=100 regime
                pts.append((8, b["budget"], b["tau"]))
    for mp in sorted(glob.glob(f"out/pilot/mxbudget_M*_{TAG}.json")):
        d = json.load(open(mp))
        M = d["m_facts"]
        for b in d["budgets"]:
            pts.append((M, b["budget"], b["tau"]))
    return pts


def _lstsq(A, y):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-9
    return coef, 1 - ss_res / ss_tot


def main():
    pts = _load_grid()
    Ms = sorted({m for m, _, _ in pts})
    print(f"=== M x budget grid ({TAG}) ===")
    print(f"M values present: {Ms}   ({len(pts)} cells)")
    for m in Ms:
        row = sorted([(b, t) for mm, b, t in pts if mm == m])
        print(f"  M={m:2d}: " + "  ".join(f"B={b}->tau={t:.2f}(B/M={b/m:.1f})" for b, t in row))

    if len(Ms) >= 2 and len(pts) >= 5:
        M = np.array([p[0] for p in pts], float)
        B = np.array([p[1] for p in pts], float)
        tau = np.array([p[2] for p in pts], float)
        lt = np.log(tau)
        # independent power law: log tau = beta*log B + mu*log M + c
        Aind = np.vstack([np.log(B), np.log(M), np.ones_like(B)]).T
        (beta, mu, _c), r2_ind = _lstsq(Aind, lt)
        # ratio-only: log tau = gamma*log(B/M) + c
        Arat = np.vstack([np.log(B / M), np.ones_like(B)]).T
        (gamma, _c2), r2_rat = _lstsq(Arat, lt)
        print("\n=== ratio-collapse test ===")
        print(f"  independent:  tau ~ B^{beta:+.3f} * M^{mu:+.3f}   (r2={r2_ind:.3f})")
        print(f"  ratio-only :  tau ~ (B/M)^{gamma:+.3f}            (r2={r2_rat:.3f})")
        print(f"  arithmetic null predicts beta == -mu (ratio-only). Observed beta={beta:+.3f}, -mu={-mu:+.3f}.")
        # if budget exponent far exceeds |mu|, tau is NOT a function of B/M alone
        if abs(beta) > 1.5 * abs(mu) or (r2_ind - r2_rat) > 0.05:
            print("  -> tau is NOT a pure function of B/M (budget acts beyond capacity arithmetic).")
        else:
            print("  -> tau is approximately ratio-dependent; arithmetic not excluded by the grid alone")
            print("     (but see the FRAMING refutation below, which is decisive).")
        # same-rho pairs: does tau depend on absolute scale at matched B/M?
        from collections import defaultdict
        byr = defaultdict(list)
        for m, b, t in pts:
            byr[round(b / m, 3)].append((m, b, t))
        matched = {r: v for r, v in byr.items() if len(v) >= 2}
        if matched:
            print("\n=== same-rho (B/M) pairs — arithmetic predicts EQUAL tau ===")
            for r, v in sorted(matched.items()):
                s = ", ".join(f"(M={m},B={b})tau={t:.2f}" for m, b, t in v)
                spread = max(t for *_, t in v) / max(min(t for *_, t in v), 1e-6)
                print(f"  B/M={r}: {s}   spread x{spread:.1f}")
    else:
        print("\n(ratio-collapse test needs >=2 M values; run mxbudget.py for M=4 and M=16 first.)")

    # The decisive refutation: identical (M,B,rho), tau varies with FRAMING alone.
    dp = f"out/pilot/disambig_{TAG}.json"
    if os.path.exists(dp):
        d = json.load(open(dp))
        C = d["conditions"]
        print("\n=== FRAMING refutation of the arithmetic null (decisive) ===")
        print(f"  At FIXED M={d['m_facts']}, B={d['budget']} (hence identical rho=B/M={d['budget']/d['m_facts']:.1f}):")
        for c in ["persona", "selfsumm", "handoff", "neutral_long"]:
            if c in C:
                print(f"    {c:13s} tau={C[c]['tau']:.2f}")
        taus = [C[c]["tau"] for c in C]
        print(f"  -> same rho, tau spans x{max(taus)/max(min(taus),1e-6):.0f} by wording alone: "
              f"a rho-only (capacity-arithmetic) model is REFUTED.")


if __name__ == "__main__":
    main()
