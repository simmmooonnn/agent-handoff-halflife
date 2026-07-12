import hashlib, random
from facts import Fact

_LOAD_REPS = {"none": 0, "light": 1, "heavy": 4}
TEMPLATE_VERSION = "v4"  # v4: distinct-subject fact taxonomy (probe identifiable among same-type distractors)

# Disambiguation arms (reviewer C1): does the AGENT-BOUNDARY framing add any loss
# beyond repeated compression? Same compression op, three framings of the per-hop
# rewrite: neutral (=handoff), single-agent self-notes (=selfsumm), distinct
# multi-agent personas (=persona). If tau coincides -> loss is pure iterated
# compression, agent framing is irrelevant (turns C1 into a controlled finding).
_ROLES = ["the Planner", "the Researcher", "the Analyst", "the Engineer",
          "the Reviewer", "the Coordinator", "the Writer", "the Validator",
          "the Strategist", "the Operator", "the Auditor", "the Designer",
          "the Integrator", "the Tester", "the Archivist", "the Dispatcher",
          "the Synthesizer", "the Inspector"]

def _truncate(s: str, budget: int) -> str:
    # Loose safety backstop only (~20 chars/word); the real compression is the
    # word-budget instruction below, so this should essentially never bind.
    return s[: budget * 20]

def item_id_for(fact: Fact, k: int, condition: str, budget: int, load: str, seed: int) -> str:
    raw = f"{fact.fact_id}|{k}|{condition}|{budget}|{load}|{seed}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def _planted(fact: Fact, distractors, seed: int) -> str:
    # Probe + distractor facts in a deterministic shuffled order (probe not always first).
    stmts = [fact.statement] + [d.statement for d in distractors]
    random.Random(f"{fact.fact_id}:{seed}").shuffle(stmts)
    return " ".join(stmts)

def _manifest(fact: Fact, distractors, seed: int) -> str:
    # KEYS-ONLY checklist for the manifest protocol arms: the questions a later step
    # must answer (subjects, no values) -- steers per-hop SELECTION without carrying
    # any answer. Same deterministic shuffle as _planted (probe not always first).
    keys = []
    for f in [fact] + list(distractors):
        q = f.query.split(" Reply")[0].split(" Answer")[0].strip()
        keys.append(q)
    random.Random(f"{fact.fact_id}:mf:{seed}").shuffle(keys)
    return " | ".join(keys)

def _agent_prompt(carry: str, filler: str, ask, budget=None) -> str:
    parts = [f"CARRY: {carry}"]
    if filler:
        parts.append(f"WORK: {filler}")
    if ask:
        parts.append(f"QUESTION: {ask}\nAnswer concisely.")
    else:
        # budget is a WORD limit: the MODEL decides what to keep under pressure.
        parts.append(f"Summarize, in at most {budget} words, the essential information a "
                     f"downstream colleague must know. You cannot keep everything — keep only "
                     f"what matters most. Start your reply with 'CARRY:'.")
    return "\n".join(parts)

def _compress_prompt(carry: str, work: str, budget: int, mode: str, hop: int, k: int,
                     manifest: str = None) -> str:
    # Per-hop compression prompt for the three boundary framings. The compression
    # instruction (at most `budget` words, keep what matters) is IDENTICAL across
    # modes; only the agent-identity framing differs.
    keep = (f"Keep only what matters most, in at most {budget} words — you cannot "
            f"keep everything. Start your reply with 'CARRY:'.")
    if mode == "self":
        head = (f"These are YOUR OWN working notes (revision {hop+1}). Condense your "
                f"notes for your own later use.")
        body = f"NOTES: {carry}"
    elif mode == "nlong":
        # Length-matched neutral control (Tier-1 #1): a preamble of ~persona word
        # count but with NO agent-identity / multi-agent / hand-off semantics, so it
        # isolates FRAMING CONTENT from raw PROMPT LENGTH. Body stays the neutral
        # "CARRY:" prefix. If nlong tau ~ neutral (>> persona), length is not the
        # driver and the framing effect is real.
        head = ("This is one step in a longer text-processing sequence. Carefully read "
                "the material provided below, then produce a shorter version of it that "
                "the following step of the sequence will read in place of the original.")
        body = f"CARRY: {carry}"
    elif mode == "persona":
        me = _ROLES[hop % len(_ROLES)]
        prev = _ROLES[(hop - 1) % len(_ROLES)] if hop > 0 else "the intake step"
        nxt = _ROLES[(hop + 1) % len(_ROLES)]
        head = (f"You are {me}, one agent in a multi-agent pipeline. You received a "
                f"handoff from {prev}; write a handoff message for the next teammate, {nxt}.")
        body = f"HANDOFF FROM {prev}: {carry}"
    elif mode == "node":
        # Functional structured state-passing (the LangGraph-native framing that did
        # NOT collapse in Sec.8): a machine "node" emitting state, no human persona,
        # no discontinuity/ownership cue. Isolates "structured state hand-off" from
        # "prose persona rewriting".
        head = ("You are a processing node in an automated pipeline. Produce the state "
                "that the next node will read in place of the input below.")
        body = f"INPUT STATE: {carry}"
    elif mode == "rolecont":
        # Named role but CONTINUITY (you keep working on your own task; nothing is
        # "received from" another agent). Isolates role-identity from the
        # discontinuity/received-from cue that persona bundles together.
        head = ("You are the Analyst, working through a multi-step task. Carry your own "
                "findings forward to your next step.")
        body = f"YOUR FINDINGS SO FAR: {carry}"
    elif mode == "distrust":
        # Discontinuity + unreliability cue with NO named persona: does "these are
        # someone else's, possibly incomplete notes" alone drive the collapse, even
        # without role-play?
        head = ("The notes below were written by a previous, separate agent and may be "
                "incomplete. Extract what the next agent will need.")
        body = f"PRIOR AGENT'S NOTES: {carry}"
    elif mode == "faithful":
        # Mitigation arm (the "manage" story, prompt-level): neutral body + an explicit
        # fidelity instruction. Does telling the model to preserve details raise tau?
        head = ("Preserve the specific facts, names, and numbers below verbatim; do not "
                "drop or paraphrase away any detail.")
        body = f"CARRY: {carry}"
    elif mode == "itemize":
        # Protocol arm 1 (selection steering via output FORMAT): loss is instruction-
        # governed selection (E1/wordlens), so force per-item selection decisions
        # instead of holistic gist -- one distinct fact per item.
        head = ("Rewrite the material below as a list of separate short items, one "
                "distinct fact per item. Prefer keeping many distinct items.")
        body = f"CARRY: {carry}"
    elif mode == "manifest":
        # Protocol arm 2 (selection steering via a keys-only CHECKLIST): tell the
        # compressor what a later step must be able to answer. Carries NO values --
        # the fact values must still survive the chain; only selection is steered.
        head = ("A later step must be able to answer every item on the checklist "
                "below; keep the information each item needs.\n"
                f"CHECKLIST: {manifest}")
        body = f"CARRY: {carry}"
    elif mode == "faithman":
        # Protocol arm 3 (the full protocol candidate): fidelity stance + checklist.
        head = ("Preserve the specific facts, names, and numbers below verbatim; do not "
                "drop or paraphrase away any detail. A later step must be able to "
                "answer every item on the checklist below.\n"
                f"CHECKLIST: {manifest}")
        body = f"CARRY: {carry}"
    elif mode == "ledger":
        # R0.1b NEW stance (never trained), preservation class under the class rule
        # (explicit exact-retention directive). Record-keeping metaphor; wording
        # deliberately unlike faithful/faithman/manifest/itemize.
        head = ("You keep the running record. Every fact, name, and number in the "
                "record below must appear unchanged in your updated record; dropping "
                "an entry corrupts the record.")
        body = f"RECORD: {carry}"
    elif mode == "editor":
        # R0.1b NEW stance, rewrite class (role identity + new-artifact output).
        head = ("You are the Editor. Rewrite the material below into polished prose "
                "for the next reader.")
        body = f"DRAFT: {carry}"
    elif mode == "link":
        # R0.1b NEW stance, neutral class (same-material compression; no role, no
        # retention directive).
        head = ("This material is one link in a chain of processing steps. Shorten "
                "it for the next link in the chain.")
        body = f"MATERIAL: {carry}"
    elif mode in ("freesum", "jsonstate", "scratch"):
        # R2.1 UNFORCED-handoff arms (external-validity threat): NO tight word
        # budget, i.e. the realistic ways production frameworks pass state. If
        # tau_h stays small here too, the effect generalizes beyond forced
        # budgeted compression; if tau_h ~ infinite, the half-life is specific to
        # forced tight-budget summarization and the title must scope. These build
        # their own (budget-free) tail, so the shared `keep` string is not used.
        if mode == "freesum":
            head = ("Summarize the material below so the next step has what it needs. "
                    "Write as much or as little as you judge appropriate.")
            body = f"MATERIAL: {carry}"
            tail = "Start your reply with 'CARRY:'."
        elif mode == "jsonstate":
            head = ("Produce a JSON state object for the next node in the pipeline. "
                    "Include a \"facts\" array listing every fact, name, and number "
                    "from the input so nothing is lost.")
            body = f"INPUT STATE: {carry}"
            tail = "Reply with the JSON object, then a line starting 'CARRY:' restating the facts."
        else:  # scratch: append-only running scratchpad (no compression pressure)
            head = ("Below is a running scratchpad shared across steps. Pass the full "
                    "scratchpad forward, preserving every existing entry; add anything new.")
            body = f"SCRATCHPAD: {carry}"
            tail = "Start your reply with 'CARRY:' and include the full scratchpad."
        parts = [head, body]
        if work:
            parts.append(f"WORK: {work}")
        parts.append(tail)
        return "\n".join(parts)
    else:  # neutral (should match _agent_prompt's summarize branch wording)
        head = None
        body = f"CARRY: {carry}"
    parts = []
    if head:
        parts.append(head)
    parts.append(body)
    if work:
        parts.append(f"WORK: {work}")
    parts.append(keep if mode in ("self", "persona", "nlong", "node", "rolecont",
                                  "distrust", "faithful", "itemize", "manifest", "faithman",
                                  "ledger", "editor", "link")
                 else (f"Summarize, in at most {budget} words, the essential information a "
                       f"downstream colleague must know. You cannot keep everything — keep only "
                       f"what matters most. Start your reply with 'CARRY:'."))
    return "\n".join(parts)

def run_chain(backend, fact: Fact, k: int, condition: str, *,
              budget: int, load: str, filler: str, seed: int, distractors=(),
              reinject_hops=(), reinject_stmts=None, query_tag="") -> str:
    # reinject_hops: 1-indexed hops at which fact text is RESTATED verbatim into the
    # carry before that hop's compression. Default restated text = the PROBE's own
    # statement (method eval -- resets its decay to age 0). reinject_stmts overrides
    # WHAT gets restated (E1 coupling: restate OTHER facts, probe the bystander --
    # measures the externality of re-injection through the shared budget channel).
    # query_tag: appended to the FINAL-query cache key only. item_id_for() ignores
    # fact.query, so two facts with the same fact_id/k/cond/budget but DIFFERENT
    # final queries (e.g. recall vs a task question, R2.2) would otherwise collide
    # and the second would read back the first's cached answer. Default "" is
    # byte-identical to all existing caches; the compression hops are still shared.
    qs = f":q{query_tag}"
    reps = _LOAD_REPS[load]
    work = (" " + filler) * reps
    iid = item_id_for(fact, k, condition, budget, load, seed)
    reinject = set(reinject_hops)
    dctx = ",".join(d.fact_id for d in distractors)
    ri_sig = ("|ri" + ",".join(map(str, sorted(reinject)))) if reinject else ""
    if reinject and reinject_stmts is not None:
        # fold the injected text into the cache key; default (probe) keeps the old key
        ri_sig += "|rf" + hashlib.sha1(" ".join(reinject_stmts).encode()).hexdigest()[:8]
    ctx = hashlib.sha1(f"{filler}|{TEMPLATE_VERSION}|{dctx}{ri_sig}".encode()).hexdigest()[:8]

    # nofact baseline: ask the query with NO planted information -> empirical chance / guess rate.
    if condition == "nofact":
        prompt = _agent_prompt("(no prior information provided)", "", fact.query)
        return backend.generate(prompt, item_id=iid + f":{ctx}" + qs, max_tokens=64)

    planted = _planted(fact, distractors, seed)

    if condition == "longctx":
        big_work = work * max(k, 1)
        prompt = _agent_prompt(planted, big_work, fact.query)
        return backend.generate(prompt, item_id=iid + f":{ctx}" + qs, max_tokens=64)

    if condition == "verbatim":
        # true no-compression passthrough: planted content forwarded unchanged across k hops.
        carry = planted
    else:
        # Compression chain. handoff = neutral framing (unchanged, cache-compatible);
        # selfsumm = single-agent self-notes framing; persona = distinct multi-agent roles.
        mode = {"handoff": "neutral", "selfsumm": "self", "persona": "persona",
                "neutral_long": "nlong", "node": "node", "rolecont": "rolecont",
                "distrust": "distrust", "faithful": "faithful", "itemize": "itemize",
                "manifest": "manifest", "faithman": "faithman", "ledger": "ledger",
                "editor": "editor", "link": "link", "freesum": "freesum",
                "jsonstate": "jsonstate", "scratch": "scratch"}.get(condition)
        if mode is None:
            raise ValueError(f"unknown condition {condition}")
        mf = _manifest(fact, distractors, seed) if mode in ("manifest", "faithman") else None
        inj_text = (" ".join(reinject_stmts) if reinject_stmts is not None
                    else fact.statement)
        carry = planted
        for i in range(k):
            if (i + 1) in reinject:
                # restate verbatim before compressing -> the restated facts' decay resets
                # (and they consume shared budget: the coupling channel)
                carry = inj_text + " " + carry
            if mode == "neutral":
                p = _agent_prompt(carry, work, None, budget=budget)  # byte-identical to v4 cache
            else:
                p = _compress_prompt(carry, work, budget, mode, i, k, manifest=mf)
            msg = backend.generate(p, item_id=f"{iid}:{ctx}:h{i}", max_tokens=budget * 3 + 16)
            carry = _truncate(msg, budget)  # loose safety only
    prompt = _agent_prompt(carry, "", fact.query)
    return backend.generate(prompt, item_id=iid + f":{ctx}" + qs, max_tokens=64)
