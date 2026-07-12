# Generates assets/overview.png (the README Figure-1) from the result JSONs in
# out/pilot/. Run from the repository root:  python scripts/make_overview_figure.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACCENT = "#D55E00"      # the tracked fact / protected slots
BLUE, GREEN, GRAY = "#0173B2", "#029E73", "#7f7f7f"
PURPLE = "#CC78BC"

fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22,
                      left=0.06, right=0.98, top=0.93, bottom=0.07)


def _load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ---------- (A) schematic: a fact decaying across budgeted handoffs ----------
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.0, 9.6, "A", fontsize=15, fontweight="bold")
boxes = [(0.2, "context\n8 facts", 1.0), (2.35, "agent 1", 0.85),
         (4.5, "agent 2", 0.45), (6.65, "agent k", 0.12)]
for x, label, alpha in boxes:
    ax.add_patch(FancyBboxPatch((x, 4.6), 1.7, 2.6, boxstyle="round,pad=0.08",
                                fc="white", ec="0.35", lw=1.2))
    ax.text(x + 0.85, 6.75, label, ha="center", fontsize=9.5, color="0.25")
    fact = "budget:\n48500" if alpha > 0.2 else "budget:\n????"
    ax.text(x + 0.85, 5.35, fact, ha="center", fontsize=9.5,
            color=ACCENT, alpha=max(alpha, 0.35), fontweight="bold")
for x in (1.95, 4.1, 6.25):
    ax.annotate("", xy=(x + 0.38, 5.9), xytext=(x, 5.9),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4))
ax.text(4.6, 7.9, "each hand-off re-summarizes the running context\n"
                  "into at most $B$ words", ha="center", fontsize=9.5, color="0.35")
ax.add_patch(FancyBboxPatch((8.55, 4.6), 1.35, 2.6, boxstyle="round,pad=0.08",
                            fc="#f5f5f5", ec="0.35", lw=1.2))
ax.text(9.22, 5.85, "query:\nwhat is\nthe budget?", ha="center", fontsize=8.6, color="0.25")
ax.annotate("", xy=(8.5, 5.9), xytext=(8.35 - 0.2, 5.9),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4))
ax.text(5.0, 2.9, r"fact survival  $S(k) = f + (S_0 - f)\,e^{-k/\tau_h}$",
        ha="center", fontsize=13)
ax.text(5.0, 1.7, r"$\tau_h$ = hand-off half-life: how many hops until a planted fact"
                  "\nis no longer usable", ha="center", fontsize=9.5, color="0.35")

# ---------- (B) real-document relay: decay + exemption rescue (e14) ----------
ax = fig.add_subplot(gs[0, 1])
e14 = _load("out/pilot/e14_real_anchor_claude-sonnet-4-6.json")
ks = e14["ks"]
for arm, color, label in (("budgeted", GRAY, "free-text hand-off (60 words)"),
                          ("realistic", BLUE, "uncapped hand-off"),
                          ("slots", ACCENT, "budget-exempt [SLOTS] (ours)")):
    m = [e14["agg"][arm][str(k)][0] for k in ks]
    lo = [e14["agg"][arm][str(k)][1] for k in ks]
    hi = [e14["agg"][arm][str(k)][2] for k in ks]
    ax.plot(ks, m, "-o", color=color, lw=2, ms=5, label=label)
    ax.fill_between(ks, lo, hi, color=color, alpha=0.13)
ax.set_xlabel("hand-offs $k$"); ax.set_ylabel("fact survival")
ax.set_ylim(0, 1.05); ax.set_xticks(ks)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("real documents: compression erases facts;\nexemption preserves them", fontsize=10.5)
ax.text(0.02, 1.06, "B", transform=ax.transAxes, fontsize=15, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# ---------- (C) budget response is model-dependent (e12 / e13) ----------
ax = fig.add_subplot(gs[1, 0])
e13h = _load("out/pilot/e13_saturation_claude-haiku-4-5.json")["grid"]
e13o = _load("out/pilot/e13_saturation_claude-opus-4-8.json")["grid"]
e12s = _load("out/pilot/e12_provisioning_claude-sonnet-4-6.json")
e12q = _load("out/pilot/e12_provisioning_Qwen2.5-7B-Instruct.json")
sonnet = [(15, e12s["calibration"]["15:4"]), (25, e12s["calibration"]["25:4"]),
          (40, e12s["calibration"]["40:4"]), (100, e12s["provisioning"]["obs"])]
qwen = [(8, e12q["calibration"]["8:4"]), (16, e12q["calibration"]["16:4"]),
        (40, e12q["calibration"]["40:4"])] + \
       [(r["B"], r["obs"]) for r in e12q["held_out"] if r["B"] == 64 and r["k"] == 4]
haiku = [(b, e13h[f"{b}:4"]) for b in (15, 40, 100)]
opus = [(b, e13o[f"{b}:4"]) for b in (15, 40, 100)]
for pts, color, name in ((opus, GREEN, "Opus 4.8"), (haiku, BLUE, "Haiku 4.5"),
                         (qwen, PURPLE, "Qwen2.5-7B"), (sonnet, GRAY, "Sonnet 4.6")):
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, "-o", color=color, lw=2, ms=5)
    ax.annotate(name, xy=(xs[-1], ys[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=9.5, color=color, va="center")
ax.set_xscale("log"); ax.set_xticks([8, 15, 25, 40, 64, 100])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(7, 210)
ax.set_xlabel("per-hop word budget $B$ (log scale)")
ax.set_ylabel("survival at $k=4$")
ax.set_ylim(0, 1.0)
ax.set_title("the budget lever is model-dependent:\nbudget buys retention on most models — far less on Sonnet",
             fontsize=10.5)
ax.text(0.02, 1.06, "C", transform=ax.transAxes, fontsize=15, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# ---------- (D) mechanism: exemption, not verbatim copy, not restating (e8) ----------
ax = fig.add_subplot(gs[1, 1])
e8 = _load("out/pilot/e8_structured_slots_claude-sonnet-4-6.json")
arms = ["neutral", "reinject", "slots_regen", "slots_copy"]
labels = ["free-text", "restate\nevery hop", "protected slot,\nrewritten", "protected slot,\ncopied"]
colors = [GRAY, GRAY, ACCENT, ACCENT]
vals = [e8["avg"][a] for a in arms]
num = [e8["per_type"]["numeric"][a] for a in arms]
xpos = range(len(arms))
ax.bar(xpos, vals, 0.56, color=colors, alpha=0.85)
ax.plot([x + 0.19 for x in xpos], num, "D", color="0.15", ms=6, label="numeric facts only")
for x, v in zip(xpos, vals):
    ax.text(x - 0.13, v + 0.025, f"{v:.2f}", ha="center", fontsize=9.5)
ax.set_xticks(list(xpos)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("survival at $k=6$"); ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("mechanism: exemption from the compression budget is the lever\n"
             "(rewriting inside a protected slot is fine; restating in prose is not)",
             fontsize=10.5)
ax.text(0.02, 1.06, "D", transform=ax.transAxes, fontsize=15, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

fig.savefig("assets/overview.png", dpi=180)
print("wrote assets/overview.png")
