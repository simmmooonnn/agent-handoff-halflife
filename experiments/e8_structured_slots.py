# STRUCTURED-SLOT CARRY (prereg out/pilot/structured_slots_predictions.json). The
# erasure law says free-text compression kills a fact because every hop REGENERATES it
# (a lossy channel -> drift/erasure). Hypothesis: change the operation from REGENERATION
# to TRANSCRIPTION. Put the fact in a [SLOTS] block the compressor is told to reproduce
# VERBATIM; only the [NOTES] prose is compressed. If numbers survive in-band this way,
# an in-band STRUCTURED carry beats free-text handoff (SOTA practice) and matches the
# out-of-band store -- without external infrastructure. Three carry representations of
# the SAME probe fact under the SAME distractor pressure and K hops:
#   neutral  = probe mixed into free-text summary, recompressed each hop  (baseline/SOTA)
#   reinject = probe restated verbatim each hop, then whole carry recompressed (DROPWATCH;
#              still REGENERATION) -- our known in-band failure
#   slots    = probe in a copy-verbatim [SLOTS] block; [NOTES] compressed  (OURS)
import hashlib, json, os, random, sys
from statistics import mean
from facts import make_facts, FACT_TYPES
from grade import grade
from relay import _agent_prompt, _truncate, run_chain, _LOAD_REPS
from run import FILLER, _build_backend

PROBES = int(os.environ.get("SLOT_PROBES", "12"))   # probes per type
NDIST = int(os.environ.get("SLOT_NDIST", "7"))      # distractors per probe (contention)
K = int(os.environ.get("SLOT_K", "6"))
budget = int(os.environ.get("SLOT_BUDGET", "25"))
LOAD = "light"


def _slot_carry(probe, distractors, seed):
    ds = [d.statement for d in distractors]
    random.Random(f"{probe.fact_id}:slot:{seed}").shuffle(ds)
    return f"[SLOTS]\n{probe.statement}\n[NOTES]\n" + " ".join(ds)


def _slot_prompt(carry, work, budget, mode):
    if mode == "copy":  # TRANSCRIPTION: slots copied verbatim (OURS)
        head = ("You are compressing a running context for the next step in a pipeline. The "
                "[SLOTS] section holds critical values that MUST survive exactly. Reproduce the "
                "[SLOTS] section VERBATIM -- character for character, unchanged, do not "
                "paraphrase, round, or drop it. Then compress the [NOTES] section to at most "
                f"{budget} words. Output exactly this shape:\n[SLOTS]\n<the slots, copied "
                "verbatim>\n[NOTES]\n<your compressed notes>")
    else:  # "regen": SAME structure, but slots are REWRITTEN (isolates structure vs copy)
        head = ("You are compressing a running context for the next step in a pipeline. Keep "
                "the [SLOTS] section for the important values and a [NOTES] section for the "
                "rest. Rewrite the [SLOTS] section concisely in your own words and compress "
                f"the [NOTES] section to at most {budget} words. Output exactly this shape:\n"
                "[SLOTS]\n<the important values, rewritten>\n[NOTES]\n<your compressed notes>")
    parts = [head, carry]
    if work:
        parts.append(f"WORK: {work}")
    return "\n".join(parts)


def run_slots(backend, probe, distractors, seed, mode):
    work = (" " + FILLER) * _LOAD_REPS[LOAD]
    carry = _slot_carry(probe, distractors, seed)
    iid = f"slots{mode}:" + hashlib.sha1(f"{probe.fact_id}|{K}|{budget}|{NDIST}|{seed}".encode()).hexdigest()[:14]
    for i in range(K):
        p = _slot_prompt(carry, work, budget, mode)
        msg = backend.generate(p, item_id=f"{iid}:h{i}", max_tokens=budget * 4 + 40)
        carry = _truncate(msg, budget * 2)  # slots block needs headroom beyond notes budget
    resp = backend.generate(_agent_prompt(carry, "", probe.query), item_id=f"{iid}:q", max_tokens=64)
    verbatim = 1 if probe.statement.strip().rstrip(".") in carry else 0  # slot transcription fidelity
    return (1 if grade(probe, resp, "actionable") else 0), verbatim


def main(provider="anthropic", model="claude-sonnet-4-6"):
    tag = model.split("/")[-1]
    backend = _build_backend(provider, model, f"data/cache_slots_{tag}.json")
    os.makedirs("out/pilot", exist_ok=True)

    facts = make_facts(400, seed=1)
    bt = {}
    for f in facts:
        bt.setdefault(f.ftype, []).append(f)
    # distractor pool per type comes from the tail; probes from the head (disjoint)
    res = {"model": model, "K": K, "budget": budget, "n_probes": PROBES, "ndist": NDIST,
           "per_type": {}}
    for t in FACT_TYPES:
        probes = bt[t][:PROBES]
        # distractors: a fixed mixed-type set (exclude the probe) -> realistic contention
        pool = [f for tt in FACT_TYPES for f in bt[tt][PROBES:PROBES + 3]]
        neu, rei, sco, srg, vrb = [], [], [], [], []
        for pi, probe in enumerate(probes):
            dist = tuple(d for d in pool if d.fact_id != probe.fact_id)[:NDIST]
            neu.append(1 if grade(probe, run_chain(backend, probe, K, "handoff", budget=budget,
                       load=LOAD, filler=FILLER, seed=0, distractors=dist), "actionable") else 0)
            rei.append(1 if grade(probe, run_chain(backend, probe, K, "handoff", budget=budget,
                       load=LOAD, filler=FILLER, seed=0, distractors=dist,
                       reinject_hops=tuple(range(1, K + 1))), "actionable") else 0)
            sc, v = run_slots(backend, probe, dist, 0, "copy")
            sr, _ = run_slots(backend, probe, dist, 0, "regen")
            sco.append(sc); vrb.append(v); srg.append(sr)
        res["per_type"][t] = {"neutral": round(mean(neu), 3), "reinject": round(mean(rei), 3),
                              "slots_regen": round(mean(srg), 3), "slots_copy": round(mean(sco), 3),
                              "slot_verbatim": round(mean(vrb), 3)}
        with open(f"out/pilot/slots_progress_{tag}.txt", "w") as fh:
            fh.write(f"done type {t}\n")

    # aggregates
    def avg(mode):
        return round(mean(res["per_type"][t][mode] for t in FACT_TYPES), 3)
    res["avg"] = {m: avg(m) for m in ("neutral", "reinject", "slots_regen", "slots_copy", "slot_verbatim")}
    res["copy_minus_neutral"] = {t: round(res["per_type"][t]["slots_copy"] - res["per_type"][t]["neutral"], 3)
                                 for t in FACT_TYPES}
    res["copy_minus_regen"] = {t: round(res["per_type"][t]["slots_copy"] - res["per_type"][t]["slots_regen"], 3)
                               for t in FACT_TYPES}
    json.dump(res, open(f"out/pilot/e8_structured_slots_{tag}.json", "w"), indent=2)

    print(f"=== STRUCTURED SLOTS ({tag}, K={K}, ndist={NDIST}, probes/type={PROBES}) ===")
    print(f"  {'type':>11} | neutral reinject slot_regen SLOT_COPY | verbatim")
    for t in FACT_TYPES:
        c = res["per_type"][t]
        print(f"  {t:>11} |  {c['neutral']:.2f}    {c['reinject']:.2f}     {c['slots_regen']:.2f}     "
              f"{c['slots_copy']:.2f}   |  {c['slot_verbatim']:.2f}")
    a = res["avg"]
    print(f"  {'AVG':>11} |  {a['neutral']:.2f}    {a['reinject']:.2f}     {a['slots_regen']:.2f}     "
          f"{a['slots_copy']:.2f}   |  {a['slot_verbatim']:.2f}")
    print(f"copy-neutral by type: {res['copy_minus_neutral']}")
    print(f"copy-regen   by type: {res['copy_minus_regen']}  (isolates transcription vs structure)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
