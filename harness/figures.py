# Generate the paper figures from the pilot result JSONs.
#   F1  the moat: HANDOFF decays while token-matched LONGCTX / VERBATIM stay flat
#   F_framing  headline #2 (§8a): same compression op, only the per-hop HEAD differs
#              -> tau spans ~80x; role/produce framing collapses, verbatim-fidelity
#              (faithful) preserves; the harmful axis is role/rewrite, not agent identity
#   F2  two panels: NO universal size law across 3 families (Qwen +0.06 n.s. vs Phi -0.36)
#              but tau RISES strongly with budget -> the channel governs, capability doesn't
#   F3  decay shape by AICc vs competitors (exp best parsimonious; power-law rejected)
# Reads out/pilot/verdict_*.json (ladder + moat), disambig_*.json (framing),
# tier1_stats.json (slope CIs + AICc), budget_sweep_*.json (budget panel).
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out/figs"
os.makedirs(OUT, exist_ok=True)

# Param counts for the Qwen2.5 ladder (billions).
SIZES = {"0.5B": 0.5e9, "1.5B": 1.5e9, "3B": 3e9, "7B": 7e9, "14B": 14e9, "32B": 32e9}

# Budget sweep on Qwen2.5-7B (n=40, M=8). JSON did not persist (job timeout at
# budget=100) so the four completed points are recorded here; overridden if the
# JSON later exists.
BUDGET_FALLBACK = {"model": "Qwen2.5-7B-Instruct",
                   "budgets": [8, 16, 25, 50],
                   "tau": [0.47, 0.82, 1.39, 1.83]}


def _exp(k, f, S0, tau):
    return f + (S0 - f) * np.exp(-np.asarray(k) / tau)


def _yerr(point, cis):
    # Build a 2xN asymmetric yerr array from [lo,hi] CIs; None CIs -> 0 error.
    lo, hi = [], []
    for p, ci in zip(point, cis):
        if ci and None not in ci:
            lo.append(max(0.0, p - ci[0]))
            hi.append(max(0.0, ci[1] - p))
        else:
            lo.append(0.0); hi.append(0.0)
    if not any(lo) and not any(hi):
        return None
    return np.array([lo, hi])


def load_ladder():
    rows = []
    for fp in sorted(glob.glob("out/pilot/verdict_Qwen*.json")):
        d = json.load(open(fp))
        tag = os.path.basename(fp).split("Qwen2.5-")[1].split("-Instruct")[0]
        if tag not in SIZES:
            continue
        rows.append({"tag": tag, "N": SIZES[tag], "v": d})
    rows.sort(key=lambda r: r["N"])
    return rows


def fig1_moat(rows):
    # Use the 7B verdict as the representative moat panel.
    d = next(r["v"] for r in rows if r["tag"] == "7B")
    h, lc, vb = d["handoff"], d["longctx"], d["verbatim"]
    ks = h["ks"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    kf = np.linspace(0, max(ks), 100)
    ax.plot(kf, _exp(kf, h["f"], h["S0"], h["tau"]), "-", color="#c0392b", lw=1.6,
            zorder=1, label=f"HANDOFF fit  (τ={h['tau']:.2f}, r²={h['r2']:.2f})")
    ax.plot(ks, h["S"], "o", color="#c0392b", ms=7, zorder=3)
    ax.plot(ks, lc["S"], "s--", color="#2471a3", ms=6, lw=1.4, label="LONG-CONTEXT (same tokens)")
    ax.plot(ks, vb["S"], "^:", color="#7d3c98", ms=6, lw=1.4, label="VERBATIM (k hops, no rewrite)")
    ch = d.get("chance", 0.125)
    ax.axhline(ch, color="gray", ls=(0, (1, 1)), lw=1, label=f"empirical chance ({ch:.2f})")
    ax.set_xlabel("agent handoffs  k")
    ax.set_ylabel("fact survival  S(k)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Compression at the boundary, not context length,\ndestroys the fact (Qwen2.5-7B, M=8 facts)", fontsize=10)
    ax.legend(fontsize=8, loc="center right", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F1_moat.png", dpi=160)
    plt.close(fig)
    print("wrote F1_moat.png")


def fig_framing():
    # Headline #2 (full map, §8a): identical compression op, only the per-hop
    # instruction HEAD differs -> tau spans ~80x. The harmful ingredient is
    # role/produce framing ("you are X; produce Y"), NOT agent identity per se:
    # rolecont (one continuing role, no boundary) and node (functional, no persona)
    # collapse hardest. faithful (verbatim-fidelity instruction) is the actionable
    # mitigation. Reads framing_sweep_*.json (falls back to disambig if absent).
    fp = "out/pilot/framing_sweep_Qwen2.5-7B-Instruct.json"
    if not os.path.exists(fp):
        fp = "out/pilot/disambig_Qwen2.5-7B-Instruct.json"
    d = json.load(open(fp))
    C = d["conditions"]
    # worst -> best along the "role/rewrite -> verbatim-preservation" axis
    order = ["node", "rolecont", "persona", "distrust", "selfsumm",
             "handoff", "faithful", "neutral_long"]
    labels = {"node": "node\n(produce\nstate)",
              "rolecont": "role-cont\n(one role,\ncontinue)",
              "persona": "persona\n(multi-agent\nhandoff)",
              "distrust": "distrust\n(prior agent\nunreliable)",
              "selfsumm": "self-notes\n(one agent)",
              "handoff": "handoff\n(neutral,\nbaseline)",
              "faithful": "faithful\n(preserve\nverbatim)",
              "neutral_long": "neutral_long\n(length ctrl,\nno identity)"}
    colors = {"node": "#7b241c", "rolecont": "#a93226", "persona": "#c0392b",
              "distrust": "#cd6155", "selfsumm": "#e67e22", "handoff": "#7f8c8d",
              "faithful": "#27ae60", "neutral_long": "#2471a3"}
    order = [c for c in order if c in C]
    tau = [C[c]["tau"] for c in order]
    err = _yerr(tau, [C[c].get("tau_ci") for c in order])
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(len(order))
    ax.bar(x, tau, yerr=err, capsize=4, color=[colors[c] for c in order],
           ecolor="#2c3e50", width=0.66)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in order], fontsize=8)
    ax.set_ylabel("handoff half-life  τ_h")
    for xi, t in zip(x, tau):
        ax.text(xi, t + 0.06, f"{t:.2f}", ha="center", va="bottom", fontsize=8.5)
    # neutral baseline reference line
    if "handoff" in C:
        hb = C["handoff"]["tau"]
        ax.axhline(hb, color="#7f8c8d", ls=(0, (4, 3)), lw=1)
        ax.text(len(order) - 0.5, hb + 0.05, "neutral baseline", ha="right",
                va="bottom", fontsize=7.5, color="#566573")
    ax.set_title("Same compression op — only the per-hop instruction HEAD differs → τ_h spans ~80×\n"
                 "role/produce framing collapses to ~chance in one hop; verbatim-fidelity preserves "
                 "(Qwen2.5-7B, n=100, budget=25, M=8, ks=0,1,2,4)", fontsize=9)
    # highlight the two mechanistic points: (a) collapse is not "agent identity"
    if "rolecont" in C and "node" in C:
        ax.annotate("no multi-agent boundary,\nyet collapses hardest\n→ it's role/produce framing,\nnot agent identity",
                    xy=(0.5, max(C["node"]["tau"], C["rolecont"]["tau"])),
                    xytext=(1.4, 3.4), fontsize=7.6, color="#7b241c",
                    arrowprops=dict(arrowstyle="->", color="#7b241c", lw=1))
    # (b) faithful = actionable mitigation
    if "faithful" in C and "handoff" in C:
        r = C["faithful"]["tau"] / max(C["handoff"]["tau"], 1e-6)
        fi = order.index("faithful")
        ax.annotate(f"one-line fidelity\ninstruction: ×{r:.1f}\nvs baseline (mitigation)",
                    xy=(fi, C["faithful"]["tau"]), xytext=(fi - 1.7, 4.6),
                    fontsize=7.6, color="#1e8449",
                    arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1))
    ax.set_ylim(0, 5.6)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/F_framing.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote F_framing.png  ({len(order)} variants; "
          f"node {C.get('node',{}).get('tau',float('nan')):.2f} .. "
          f"neutral_long {C.get('neutral_long',{}).get('tau',float('nan')):.2f})")


def fig_method():
    # METHOD Pareto: tau-aware re-injection dominates uniform fixed-period.
    d = json.load(open("out/pilot/scheduler_sim.json"))
    K, theta = d["K"], d["theta"]
    tau = sorted(d["tau_curve"])           # [cost, meanS, minS]
    fix = sorted(d["fixed_curve"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot([c for c, _, m in tau], [m for _, _, m in tau], "-o", color="#27ae60",
            ms=4, lw=1.8, label="τ-aware (ours)")
    ax.plot([c for c, _, m in fix], [m for _, _, m in fix], "s--", color="#c0392b",
            ms=6, lw=1.3, label="fixed-period (uniform)")
    ax.scatter([d["never"]["cost"]], [d["never"]["minS"]], color="#7f8c8d", s=60,
               zorder=5, label="never / always")
    ax.scatter([d["always"]["cost"]], [d["always"]["minS"]], color="#7f8c8d", s=60, zorder=5)
    ax.axhline(theta, color="gray", ls=(0, (1, 1)), lw=1, label=f"target θ={theta}")
    ax.set_xlabel("re-injection cost  (total restatements over pipeline)")
    ax.set_ylabel("worst-case fact survival  min_i S_i(K)")
    ax.set_title(f"τ-aware re-injection Pareto-dominates uniform\n"
                 f"(K={K} hops; fact τ spans ×19; worst-case retention per cost)", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F_method.png", dpi=160)
    plt.close(fig)
    print("wrote F_method.png")


def fig_method_live():
    # LIVE head-to-head (§8b iii): the actual a_star schedule vs matched-cost uniform,
    # executed on real Qwen2.5-7B. LEFT: cost vs mean worst-case retention -> tau-aware
    # sits up-and-left (more retention for less/equal cost). RIGHT: the matched-cost
    # point (theta=0.3, both cost 4) per fact type -> the win is reallocation to the
    # fastest-decaying type (numeric).
    fp = "out/pilot/sched_live_Qwen2.5-7B-Instruct.json"
    if not os.path.exists(fp):
        print("skip F_method_live (no sched_live json)")
        return
    d = json.load(open(fp))
    runs = d["runs"]
    ta = sorted((r["tau_aware"]["cost_per_fact"], r["tau_aware"]["mean_clean"], r["theta"]) for r in runs)
    un = sorted((r["uniform"]["cost_per_fact"], r["uniform"]["mean_clean"], r["theta"]) for r in runs)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    axL.plot([c for c, _, _ in ta], [m for _, m, _ in ta], "-o", color="#27ae60", ms=8,
             lw=2.0, label="τ-aware (ours)", zorder=3)
    axL.plot([c for c, _, _ in un], [m for _, m, _ in un], "s--", color="#c0392b", ms=7,
             lw=1.5, label="uniform (τ-agnostic)", zorder=2)
    for c, m, th in ta:
        axL.annotate(f"θ={th}", (c, m), textcoords="offset points", xytext=(4, 6),
                     fontsize=7.5, color="#1e8449")
    # equal-quality-cheaper: the two 0.70 points (tau-aware cost 6, uniform cost 8)
    axL.annotate("same worst-case,\n−25% cost", xy=(6, 0.70), xytext=(3.1, 0.80),
                 fontsize=8, color="#1e8449",
                 arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1))
    axL.annotate("", xy=(8, 0.70), xytext=(6, 0.70),
                 arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1, ls=":"))
    axL.set_xlabel("re-injection cost  (restatements / fact)")
    axL.set_ylabel("mean worst-case retention  (clean types)")
    axL.set_ylim(0, 0.9)
    axL.set_title("τ-aware retains more per unit cost\n(live, Qwen2.5-7B, K=8; worst-case = min over hops)", fontsize=9)
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(alpha=0.25)

    # RIGHT: matched-cost point (theta=0.3), per clean fact type
    r3 = min(runs, key=lambda r: abs(r["theta"] - 0.3))
    types = d["clean"]
    taw = [r3["tau_aware"]["by_type_worst"][t] for t in types]
    unw = [r3["uniform"]["by_type_worst"][t] for t in types]
    x = np.arange(len(types)); w = 0.38
    b1 = axR.bar(x - w / 2, taw, w, color="#27ae60", label="τ-aware")
    b2 = axR.bar(x + w / 2, unw, w, color="#c0392b", label="uniform")
    for bars in (b1, b2):
        for b in bars:
            axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    # annotate the per-type periods tau-aware chose
    per = r3["tau_aware"]["period"]
    axR.set_xticks(x)
    axR.set_xticklabels([f"{t}\n(τ-aware every-{per[t]})" for t in types], fontsize=8)
    axR.set_ylabel("worst-case retention")
    axR.set_ylim(0, 0.85)
    tac = r3["tau_aware"]["cost_per_fact"]; unc = r3["uniform"]["cost_per_fact"]
    axR.set_title(f"Matched cost (θ=0.3, both {tac:.0f} restatements/fact):\n"
                  f"budget saved on chance-bound negation → the fastest type", fontsize=9)
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(alpha=0.25, axis="y")

    fig.suptitle("The τ-aware schedule's advantage reproduces on the real model (not just fitted curves)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F_method_live.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote F_method_live.png")


def fig_surface():
    # §8d: the two-dial instruction-surface model, validated out-of-sample.
    # LEFT: decompose all 12 training arms vs the neutral length curve L(w). Collapse/
    #   rewrite arms sit ON the curve (residual ~0 = fully length-mediated); preservation
    #   arms sit ABOVE (a real stance bonus). RIGHT: the 6 PRE-REGISTERED held-out cells,
    #   predicted-vs-observed -> two-dial hugs y=x; length-only collapses to a flat band.
    d = json.load(open(f"{OUT}/../pilot/surface_Qwen2.5-7B-Instruct.json"))
    c, slope, floor = d["curve"]["c"], d["curve"]["slope"], d["curve"]["floor"]
    S0 = 0.91

    def L(w):
        return floor + (S0 - floor) * np.exp(-4.0 / (c * np.asarray(w, float) ** slope))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- LEFT: length curve + per-arm stance residual ---
    wf = np.linspace(7, 45, 200)
    axL.plot(wf, L(wf), "-", color="#34495e", lw=1.8, zorder=2,
             label="neutral length curve  L(w)")
    anc = d["anchors"]
    axL.plot([a["w"] for a in anc], [a["S4"] for a in anc], "D", color="#34495e",
             ms=7, zorder=4, label="neutral anchors (curve fit)")
    disp = {"node@25": "node", "persona@25": "persona", "rolecont@25": "role-cont",
            "selfsumm@25": "self-notes", "neutral_long@25": "neutral_long",
            "distrust@25": "distrust", "faithful@25": "faithful", "itemize@25": "itemize",
            "manifest@25": "manifest", "faithman@25": "faithman",
            "itemize@12": "itemize@12", "faithful@50": "faithful@50"}
    first_col = first_pre = True
    for a in d["arms"]:
        w, s4, resid = a["realized_w"], a["S4"], a["stance_residual"]
        preserve = resid >= 0.10
        col = "#27ae60" if preserve else "#c0392b"
        if preserve:  # draw the bonus as a vertical offset from the curve up to the point
            axL.plot([w, w], [float(L(w)), s4], "-", color="#27ae60", lw=1.0, alpha=0.6, zorder=3)
        lab = None
        if preserve and first_pre:
            lab = "preservation arms (bonus > 0)"; first_pre = False
        elif not preserve and first_col:
            lab = "on the length curve (bonus ≈ 0)"; first_col = False
        axL.plot(w, s4, "o", color=col, ms=8, zorder=5, label=lab,
                 markeredgecolor="white", markeredgewidth=0.6)
    for a in d["arms"]:  # label the four mechanistically important arms
        nm = disp.get(a["arm"], a["arm"])
        if nm in ("manifest", "faithful", "node", "persona"):
            dy = 0.045 if a["stance_residual"] >= 0.10 else -0.075
            axL.annotate(f"{nm}\n(+{a['stance_residual']:.2f})" if a["stance_residual"] >= 0.10
                         else f"{nm}\n({a['stance_residual']:+.2f})",
                         (a["realized_w"], a["S4"]), textcoords="offset points",
                         xytext=(6, 10 if dy > 0 else -22), fontsize=7.4,
                         color="#1e8449" if a["stance_residual"] >= 0.10 else "#922b21")
    axL.set_xlabel("realized carry length  (words the model actually emits)")
    axL.set_ylabel("fact survival  S(4)")
    axL.set_ylim(0, 1.0)
    axL.set_title("Two dials: realized length (curve) + preservation bonus (offset)\n"
                  "collapse arms sit ON the curve (length-mediated), not below "
                  "(Qwen2.5-7B, 12 training arms)", fontsize=9)
    axL.legend(fontsize=7.6, loc="upper left", framealpha=0.92)
    axL.grid(alpha=0.25)

    # --- RIGHT: pre-registered held-out predicted-vs-observed ---
    # Observed values are from job 18836612 (heldout_Qwen2.5-7B-Instruct.json lives on
    # Anvil); predictions were git-committed (d5cf4e1) BEFORE the run.
    HELD = [  # (cell, S4_obs, S4_pred_twodial, S4_pred_lenonly)
        ("manifest@12", 0.61, 0.60, 0.23), ("manifest@50", 0.80, 0.73, 0.39),
        ("node@50", 0.16, 0.15, 0.23),     ("persona@50", 0.21, 0.23, 0.26),
        ("faithful@8", 0.34, 0.46, 0.22),  ("faithman@8", 0.45, 0.59, 0.22)]
    obs = [h[1] for h in HELD]
    axR.plot([0, 0.9], [0, 0.9], "-", color="#95a5a6", lw=1.2, zorder=1, label="perfect (y = x)")
    axR.fill_between([0, 0.9], [-0.1, 0.8], [0.1, 1.0], color="#95a5a6", alpha=0.12,
                     zorder=0, label="±0.10 band")
    axR.scatter(obs, [h[3] for h in HELD], marker="s", s=80, facecolors="none",
                edgecolors="#c0392b", linewidths=1.6, zorder=3, label="length-only null")
    axR.scatter(obs, [h[2] for h in HELD], marker="o", s=90, color="#27ae60",
                edgecolors="white", linewidths=0.7, zorder=4, label="two-dial (ours)")
    for cell, o, p2, pl in HELD:
        axR.annotate(cell, (o, p2), textcoords="offset points", xytext=(7, -3),
                     fontsize=7.2, color="#1e8449")
    axR.set_xlabel("observed  S(4)   (held-out, real 7B)")
    axR.set_ylabel("predicted  S(4)")
    axR.set_xlim(0, 0.9); axR.set_ylim(0, 0.9)
    axR.set_aspect("equal", adjustable="box")
    axR.set_title("Pre-registered held-out test: two-dial hugs y=x,\n"
                  "length-only is a flat band  (median |err|=0.043; MSE 0.0065 vs 0.0638)",
                  fontsize=9)
    axR.legend(fontsize=7.6, loc="upper left", framealpha=0.92)
    axR.grid(alpha=0.25)

    fig.suptitle("The instruction is a two-dial control surface — and it predicts out-of-sample",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F_surface.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote F_surface.png")


def fig_probe():
    # §8d probe-and-forecast: LEFT depth trajectories -- 2 measured probe hops
    #   (k=1,2) + the geometric continuation forecast the whole curve; observed
    #   k=4/8/16 land on the flat forecast (loss is front-loaded). RIGHT the
    #   pre-registered predicted-vs-observed at every forecast depth.
    FLOOR, PILOT = 0.22158, "out/pilot"

    def srv(rows, lab, k):
        v = [r["correct"] for r in rows if r["_lab"] == lab and r["k"] == k]
        return float(np.mean(v)) if v else None

    ns = [json.loads(l) for l in open(f"{PILOT}/newstance_rows_Qwen2.5-7B-Instruct.jsonl")]
    for r in ns:
        r["_lab"] = r.get("label") or f"{r['condition']}@25"
    pr = [json.loads(l) for l in open(f"{PILOT}/protocol_rows_Qwen2.5-7B-Instruct.jsonl")]
    for r in pr:
        r["_lab"] = f"{r['condition']}@25"
    dp = [json.loads(l) for l in open(f"{PILOT}/deepprobe_rows_Qwen2.5-7B-Instruct.jsonl")]
    for r in dp:
        r["_lab"] = r["label"]
    shallow = ns + pr

    ARMS = [  # (label, color, display)
        ("manifest@25", "#27ae60", "manifest (preservation)"),
        ("faithman@25", "#82c785", "faithman (preservation)"),
        ("neutral_long@25", "#7f8c8d", "neutral_long"),
        ("ledger@25", "#c0392b", "ledger (NEW, role-poisoned)"),
        ("editor@25", "#e67e22", "editor (NEW, rewrite)"),
        ("link@25", "#8e44ad", "link (NEW, neutral)")]

    def probe_forecast(s1, s2):
        # 2-hop floor-anchored geometric continuation; zero free parameters
        if s2 <= FLOOR or s1 <= FLOOR:
            return max(s2, 0.0)
        return FLOOR + (s2 - FLOOR) * ((s2 - FLOOR) / (s1 - FLOOR)) ** 2

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    axL.axvspan(0, 4, color="#3498db", alpha=0.08, zorder=0)
    axL.text(2.0, 0.985, "PROBE (k ≤ 4 measured)", ha="center", va="top", fontsize=7.6,
             color="#21618c", fontweight="bold")
    axL.text(10.5, 0.985, "FORECAST  S(k≥8) = S(4)\n(persistence law, zero free params)",
             ha="center", va="top", fontsize=7.6, color="#616a6b", fontweight="bold")
    scat = {"s4": [[], []], "s8": [[], []], "s16": [[], []]}
    for lab, col, disp in ARMS:
        s0, s1, s2 = (srv(shallow, lab, k) for k in (0, 1, 2))
        s4 = srv(shallow, lab, 4)
        s8, s16 = srv(dp, lab, 8), srv(dp, lab, 16)
        p4 = probe_forecast(s1, s2)
        axL.plot([0, 1, 2, 4], [s0, s1, s2, s4], "-o", color=col, ms=6, lw=1.4, zorder=4,
                 markeredgecolor="white", markeredgewidth=0.5, label=disp)
        # cheaper sub-claim: S(4) forecast from the 2-hop prefix alone (x marker)
        axL.plot(4, p4, "x", color=col, ms=8, markeredgewidth=1.8, zorder=5)
        # persistence forecast: flat at S(4) measured, from k=4 out to 16
        axL.plot([4, 16], [s4, s4], "--", color=col, lw=1.2, alpha=0.75, zorder=2)
        obs_pts = [(8, s8)] + ([(16, s16)] if s16 is not None else [])
        axL.plot([k for k, s in obs_pts], [s for k, s in obs_pts], "o", color=col, ms=7,
                 markerfacecolor="none", markeredgewidth=1.6, zorder=5)
        scat["s4"][0].append(p4); scat["s4"][1].append(s4)
        scat["s8"][0].append(s4); scat["s8"][1].append(s8)  # persistence pred = S4 measured
        if s16 is not None:
            scat["s16"][0].append(s4); scat["s16"][1].append(s16)
    axL.plot([], [], "o", color="#2c3e50", markerfacecolor="none", markeredgewidth=1.6,
             label="observed at forecast depths")
    axL.plot([], [], "x", color="#2c3e50", ms=8, markeredgewidth=1.8,
             label="S(4) from 2-hop prefix only")
    axL.axhline(FLOOR, color="#95a5a6", lw=0.9, ls=":", zorder=1)
    axL.text(15.8, FLOOR + 0.012, "global floor", ha="right", fontsize=7, color="#7f8c8d")
    axL.set_xticks([0, 1, 2, 4, 8, 16])
    axL.set_xlabel("handoff depth  k")
    axL.set_ylabel("fact survival  S(k)")
    axL.set_xlim(-0.4, 16.4); axL.set_ylim(0, 1.0)
    axL.set_title("Probe ≤4 hops → forecast the whole depth curve\n"
                  "loss is front-loaded: whatever survives k≈4 survives k=16 (Qwen2.5-7B)",
                  fontsize=9)
    axL.legend(fontsize=7.0, loc="center right", framealpha=0.92)
    axL.grid(alpha=0.25)

    # RIGHT: pre-registered predicted-vs-observed at every forecast depth
    axR.plot([0, 0.85], [0, 0.85], "-", color="#95a5a6", lw=1.2, zorder=1, label="perfect (y = x)")
    axR.fill_between([0, 0.85], [-0.05, 0.80], [0.05, 0.90], color="#95a5a6", alpha=0.15,
                     zorder=0, label="±0.05 band (gate)")
    axR.scatter(scat["s4"][1], scat["s4"][0], marker="o", s=85, color="#2980b9",
                edgecolors="white", linewidths=0.7, zorder=4,
                label="2-hop probe → S(4)   (new-stance med. |err| 0.005)")
    mi = int(np.argmax(np.abs(np.array(scat["s4"][1]) - np.array(scat["s4"][0]))))
    axR.annotate("manifest: private floor 0.66 ≫ global 0.22\n(known limit — probe to k=4 instead)",
                 (scat["s4"][1][mi], scat["s4"][0][mi]), textcoords="offset points",
                 xytext=(-8, -26), fontsize=6.8, color="#21618c", ha="right")
    axR.scatter(scat["s8"][1], scat["s8"][0], marker="^", s=85, color="#27ae60",
                edgecolors="white", linewidths=0.7, zorder=4,
                label="persistence → S(8)   (med. |err| 0.000)")
    axR.scatter(scat["s16"][1], scat["s16"][0], marker="s", s=85, color="#8e44ad",
                edgecolors="white", linewidths=0.7, zorder=4,
                label="persistence → S(16)   (|err| 0.00, 0.00)")
    axR.set_xlabel("observed survival   (live chains, pre-registered)")
    axR.set_ylabel("forecast survival")
    axR.set_xlim(0, 0.85); axR.set_ylim(0, 0.85)
    axR.set_aspect("equal", adjustable="box")
    axR.set_title("Pre-registered forecasts vs observation (job 18847747)\n"
                  "text-level prediction FAILED here (median err 0.29); probes do not",
                  fontsize=9)
    axR.legend(fontsize=7.2, loc="upper left", framealpha=0.92)
    axR.grid(alpha=0.25)

    fig.suptitle("Probe-and-forecast: survival is unreadable from instruction text but "
                 "forecastable from a shallow (≤4-hop) probe", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F_probe.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote F_probe.png")


def _powerlaw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(np.log(x), np.log(y), 1)  # slope, intercept in log-log
    yhat = a + b * np.log(x)
    ss = 1 - np.sum((np.log(y) - yhat) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)
    return b, np.exp(a), ss


def fig2_two_panel(rows):
    st = json.load(open("out/pilot/tier1_stats.json"))
    aRaw, ciRaw = st["alpha_raw"]["alpha_point"], st["alpha_raw"]["ci"]
    aClean, ciClean = st["alpha_clean"]["alpha_point"], st["alpha_clean"]["ci"]
    beta, ciB = st["beta"]["beta_point"], st["beta"]["ci"]

    N = [r["N"] for r in rows]
    tau = [r["v"]["handoff"]["tau"] for r in rows]
    tauN_err = _yerr(tau, [r["v"]["handoff"].get("tau_ci") for r in rows])

    bs = BUDGET_FALLBACK
    j = glob.glob("out/pilot/budget_sweep_*.json")
    if j:
        d = json.load(open(j[0]))
        if len(d["budgets"]) >= 3:
            bs = {"model": d["model"], "budgets": [b["budget"] for b in d["budgets"]],
                  "tau": [b["tau"] for b in d["budgets"]],
                  "tau_ci": [b.get("tau_ci") for b in d["budgets"]]}
    _, cB, r2B = _powerlaw(bs["budgets"], bs["tau"])
    tauB_err = _yerr(bs["tau"], bs.get("tau_ci", [None] * len(bs["tau"])))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # LEFT: size across THREE families (cross_family.json) using clean-type tau.
    # No universal tau~N^alpha: Qwen alpha_clean=+0.06 (n.s.), Phi alpha_clean=-0.36
    # (OPPOSITE sign) -> a sign reversal rules out any universal size law.
    cf = json.load(open("out/pilot/cross_family.json"))
    fam_style = {"Qwen2.5": ("#2c3e50", "o"), "Phi-3": ("#e67e22", "s"), "Mistral": ("#2471a3", "^")}
    # shade the ~2x band spanned by all clean tau
    all_clean = [m["clean"]["tau"] for m in cf["models"].values()]
    axL.axhspan(min(all_clean), max(all_clean), color="#bdc3c7", alpha=0.18, zorder=0)
    for fam, (col, mk) in fam_style.items():
        ms = sorted([m for m in cf["models"].values() if m["family"] == fam], key=lambda m: m["N"])
        if not ms:
            continue
        Nf = [m["N"] for m in ms]
        tf = [m["clean"]["tau"] for m in ms]
        ef = _yerr(tf, [m["clean"]["tau_ci"] for m in ms])
        axL.errorbar(Nf, tf, yerr=ef, fmt=mk, color=col, ms=8, lw=0, capsize=3,
                     ecolor=col, label=fam, zorder=3)
        if len(ms) >= 2:  # trend line + slope label
            a = cf["families"][fam]["alpha_clean"]
            axL.plot(Nf, tf, "-", color=col, lw=1.2, alpha=0.7, zorder=2)
            axL.annotate(f"{fam}: α_clean={a:+.2f}", (Nf[-1], tf[-1]),
                         textcoords="offset points", xytext=(-4, -14), fontsize=7.5,
                         color=col, ha="right", fontweight="bold")
    axL.set_xscale("log")
    axL.set_xlabel("model size  N (parameters)")
    axL.set_ylabel("clean-type handoff half-life  τ_h")
    axL.set_ylim(0, 3.2)
    axL.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axL.set_title("No universal size law across families\n"
                  f"Qwen α={aClean:+.2f} [{ciClean[0]:.2f},{ciClean[1]:.2f}] (n.s.)  vs  "
                  f"Phi-3 α={cf['families']['Phi-3']['alpha_clean']:+.2f} (point est., n.s.)", fontsize=9)
    axL.grid(alpha=0.25, which="both")

    # RIGHT: budget -- strong, significant power law.
    axR.errorbar(bs["budgets"], bs["tau"], yerr=tauB_err, fmt="o", color="#c0392b", ms=8,
                 capsize=3, ecolor="#e59866")
    xf = np.linspace(min(bs["budgets"]), max(bs["budgets"]), 100)
    axR.plot(xf, cB * xf ** beta, "-", color="#c0392b", lw=1.4)
    axR.set_xlabel("per-hop compression budget  (words)")
    axR.set_ylabel("handoff half-life  τ_h")
    axR.set_ylim(0, 3.2)
    axR.set_title(f"The compression channel governs\nτ ∝ budget^{beta:.2f} "
                  f"[{ciB[0]:.2f},{ciB[1]:.2f}], r²={r2B:.2f}", fontsize=9)
    axR.grid(alpha=0.25)

    fig.suptitle("The channel (budget), not model capability, sets the handoff half-life",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F2_budget_not_size.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote F2_budget_not_size.png  (raw α={aRaw:.3f}{ciRaw} | clean α={aClean:.3f}{ciClean} | β={beta:.3f}{ciB})")


def fig3_decay_shape(rows):
    # AICc vs competitors (from tier1_stats). Plot ΔAICc (relative to best).
    st = json.load(open("out/pilot/tier1_stats.json"))["decay_shape_7B"]["models"]
    disp = {"exponential(=geometric)": "exponential\n(=geometric)", "stretched_exp": "stretched\nexp",
            "power_law": "power-law", "linear": "linear", "constant": "constant"}
    items = [(disp.get(m, m), v["delta_aicc"], m) for m, v in st.items() if "aicc" in v]
    items.sort(key=lambda t: t[1])
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    keys = [i[2] for i in items]
    # Highlight the exponential as the parsimonious headline model (green); the
    # stretched-exp is numerically lowest but near-saturated (n=6, 4 params) -> hatched.
    colors, hatch = [], []
    for k in keys:
        if k == "exponential(=geometric)":
            colors.append("#27ae60"); hatch.append("")
        elif k == "stretched_exp":
            colors.append("#7fbf7f"); hatch.append("//")
        else:
            colors.append("#95a5a6"); hatch.append("")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    bars = ax.bar(names, vals, color=colors)
    for b, h in zip(bars, hatch):
        if h:
            b.set_hatch(h)
    ax.set_ylabel("ΔAICc  (0 = best; lower better)")
    ax.set_title("Decay is exponential; power-law & linear rejected\n"
                 "(Qwen2.5-7B handoff; stretched-exp lowest but near-saturated at n=6)",
                 fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", va="bottom", ha="center", fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/F3_decay_shape.png", dpi=160)
    plt.close(fig)
    print("wrote F3_decay_shape.png")


if __name__ == "__main__":
    rows = load_ladder()
    print(f"loaded {len(rows)} ladder models:", [r['tag'] for r in rows])
    fig1_moat(rows)
    fig_framing()
    fig2_two_panel(rows)
    fig3_decay_shape(rows)
    fig_method()
    fig_method_live()
    fig_surface()
    fig_probe()
    print("done ->", OUT)
