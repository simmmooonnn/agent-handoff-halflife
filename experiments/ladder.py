# Aggregate per-model pilot verdicts into a tau ~ N^alpha ladder report.
import json, glob, sys
from analyze import fit_power_law

SIZES = {
    "Qwen2.5-0.5B-Instruct": 0.5e9, "Qwen2.5-1.5B-Instruct": 1.5e9,
    "Qwen2.5-3B-Instruct": 3e9, "Qwen2.5-7B-Instruct": 7e9,
    "Qwen2.5-14B-Instruct": 14e9, "Qwen2.5-32B-Instruct": 32e9,
    "Qwen2.5-72B-Instruct": 72e9,
}

def main(pattern="out/pilot/verdict_*.json"):
    rows = []
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        name = d["model"].split("/")[-1]
        if name not in SIZES:
            continue
        h = d["handoff"]
        emp = (d.get("empirical_chance") or {}).get("overall")
        floor = emp if emp is not None else d.get("chance", 0.0)
        below = (h["S0"] - floor) < 0.15           # probe not clearly above guess rate
        rows.append({"model": name, "N": SIZES[name], "tau": h["tau"], "S0": h["S0"],
                     "r2": h["r2"], "below_floor": below,
                     "shape": (d.get("decay_shape") or {}).get("best"), "GO": d.get("GO")})
    rows.sort(key=lambda r: r["N"])
    print(f"{'model':26}{'N':>10}{'tau':>8}{'S0':>7}{'r2':>7}{'shape':>12}  below_floor")
    for r in rows:
        print(f"{r['model']:26}{r['N']:>10.2e}{r['tau']:>8.2f}{r['S0']:>7.2f}{r['r2']:>7.2f}"
              f"{str(r['shape']):>12}  {r['below_floor']}")
    good = [r for r in rows if not r["below_floor"] and r["r2"] > 0.8]
    if len(good) >= 2:
        pl = fit_power_law([r["N"] for r in good], [r["tau"] for r in good])
        print(f"\ntau ~ N^alpha:  alpha={pl['alpha']:.3f}  r2={pl['r2']:.3f}  "
              f"(over {len(good)} above-floor models: {[r['model'] for r in good]})")
    else:
        print(f"\nnot enough above-floor models for a power-law fit yet ({len(good)})")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
