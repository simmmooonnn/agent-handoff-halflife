# R0.1a — Leave-one-STANCE-out CV for the two-dial surface (round-4 CRITICAL #1).
#
# The round-4 panel showed heldout.py only held out (arm,budget) cells of stances
# whose bonus was already fit -> it validated budget-transfer, not stance
# generalization. This script asks the harder question on cached data alone:
# can an UNSEEN stance's S4 be predicted from (a) its realized length via the
# neutral curve L(w) plus (b) a bonus inferred WITHOUT ever fitting that stance,
# via a class rule stated from the instruction TEXT?
#
# Class rule (from the per-hop head text, not from outcomes):
#   preservation - head explicitly instructs verbatim carry / itemized list /
#                  keys checklist          -> faithful, faithman, itemize, manifest
#   rewrite      - head assigns a role or asks to produce a new artifact
#                                          -> node, persona, rolecont
#   neutral      - neither                 -> selfsumm, distrust, neutral_long
#
# LOSO: hold out ALL cells of one stance; bonus_pred = mean residual of the OTHER
# stances in its class; S4_pred = clip(L(w_obs) + bonus_pred). Baselines:
#   length-only  S4 = L(w_obs)                  (no bonus dial)
#   global-mean  S4 = L(w_obs) + mean(all other stances' residuals)
#                                               (a bonus dial with no class structure)
# The class model must beat BOTH to show the class-structured bonus generalizes.
#
# HONEST CAVEAT (report verbatim in the paper): the class rule was written after
# the training results were seen, so this CV bounds but does not prove
# generalization; the pre-registered NEW-stance live test (newstance.py) is the
# load-bearing evidence.
import json
import math
from statistics import mean, median

CLASSES = {
    "faithful": "preservation", "faithman": "preservation",
    "itemize": "preservation", "manifest": "preservation",
    "node": "rewrite", "persona": "rewrite", "rolecont": "rewrite",
    "selfsumm": "neutral", "distrust": "neutral", "neutral_long": "neutral",
}


def main():
    d = json.load(open("out/pilot/surface_Qwen2.5-7B-Instruct.json"))
    c, slope, floor = d["curve"]["c"], d["curve"]["slope"], d["curve"]["floor"]
    S0 = 0.91

    def L(w):
        return floor + (S0 - floor) * math.exp(-4.0 / (c * w ** slope))

    cells = []
    for a in d["arms"]:
        stance = a["arm"].split("@")[0]
        cells.append({"cell": a["arm"], "stance": stance, "cls": CLASSES[stance],
                      "w": a["realized_w"], "S4": a["S4"],
                      "resid": a["stance_residual"]})

    # per-stance mean residual (a stance with 2 cells contributes its mean once,
    # so multi-cell stances don't dominate the class means)
    stances = sorted({x["stance"] for x in cells})
    stance_resid = {s: mean([x["resid"] for x in cells if x["stance"] == s])
                    for s in stances}

    rows, e_cls, e_len, e_glob = [], [], [], []
    for x in cells:
        held = x["stance"]
        others = [s for s in stances if s != held]
        cls_others = [s for s in others if CLASSES[s] == x["cls"]]
        b_cls = mean([stance_resid[s] for s in cls_others])
        b_glob = mean([stance_resid[s] for s in others])
        base = L(x["w"])
        p_cls = min(1.0, max(0.0, base + b_cls))
        p_glob = min(1.0, max(0.0, base + b_glob))
        p_len = min(1.0, max(0.0, base))
        e_cls.append(abs(p_cls - x["S4"]))
        e_glob.append(abs(p_glob - x["S4"]))
        e_len.append(abs(p_len - x["S4"]))
        rows.append((x["cell"], x["cls"], x["w"], x["S4"], p_cls, e_cls[-1],
                     p_len, e_len[-1], p_glob, e_glob[-1]))

    def mse(e):
        return sum(v * v for v in e) / len(e)

    print("=== R0.1a leave-one-STANCE-out CV (12 training cells, cached only) ===")
    print(f"{'cell':>15} {'class':>12} {'w':>5} {'S4':>5} {'p_cls':>6} {'e':>5} "
          f"{'p_len':>6} {'e':>5} {'p_glb':>6} {'e':>5}")
    for r in rows:
        print(f"{r[0]:>15} {r[1]:>12} {r[2]:>5.1f} {r[3]:>5.2f} {r[4]:>6.2f} "
              f"{r[5]:>5.2f} {r[6]:>6.2f} {r[7]:>5.2f} {r[8]:>6.2f} {r[9]:>5.2f}")
    out = {"cells": [{"cell": r[0], "cls": r[1], "w": r[2], "S4": r[3],
                      "pred_class": round(r[4], 3), "err_class": round(r[5], 3),
                      "pred_lenonly": round(r[6], 3), "err_lenonly": round(r[7], 3),
                      "pred_globalmean": round(r[8], 3), "err_globalmean": round(r[9], 3)}
                     for r in rows],
           "median_abs_err": {"class": round(median(e_cls), 3),
                              "lenonly": round(median(e_len), 3),
                              "globalmean": round(median(e_glob), 3)},
           "mse": {"class": round(mse(e_cls), 4), "lenonly": round(mse(e_len), 4),
                   "globalmean": round(mse(e_glob), 4)},
           "class_rule": CLASSES}
    print(f"\nmedian|err|: class={out['median_abs_err']['class']}  "
          f"length-only={out['median_abs_err']['lenonly']}  "
          f"global-mean={out['median_abs_err']['globalmean']}")
    print(f"MSE:         class={out['mse']['class']}  "
          f"length-only={out['mse']['lenonly']}  "
          f"global-mean={out['mse']['globalmean']}")
    win = out["mse"]["class"] < out["mse"]["lenonly"] and \
        out["mse"]["class"] < out["mse"]["globalmean"]
    out["class_beats_both"] = bool(win)
    print(f"class-structured bonus beats BOTH baselines: {win}")
    json.dump(out, open("out/pilot/loso_Qwen2.5-7B-Instruct.json", "w"), indent=2)
    print("wrote out/pilot/loso_Qwen2.5-7B-Instruct.json")


if __name__ == "__main__":
    main()
