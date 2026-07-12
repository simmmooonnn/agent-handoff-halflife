# N2 corpus: 16 authored prose documents with embedded FICTIONAL verifiable facts.
# Natural = facts live mid-sentence in flowing discourse (not templated statements),
# and the chain's task is summarizing the DOCUMENT. All specifics are invented
# (names, 4-digit numbers, fictional products) so parametric memory cannot answer;
# the nofact arm verifies this empirically (gate G3 <= 0.15).
# Each fact: fid, ftype (numeric/entity/negation/preference), query, answer.
# negation queries are yes/no (grade() routes by first yes/no token).

DOCS = [
 {"doc_id": "d01", "text": (
   "Postmortem for the Talvex outage of March 12. The incident began when the batch "
   "reconciler stalled after processing 4271 invoices, leaving the remainder queued "
   "overnight. On-call was rotated mid-incident, and it was Ferrand who ultimately "
   "traced the stall to a deadlock in the ledger writer. We confirmed the archived "
   "replica was not affected at any point. Going forward the team agreed that retries "
   "should use exponential backoff rather than fixed intervals, since fixed retries "
   "amplified the queue pressure during the stall window."),
  "facts": [
   {"fid": "d01n", "ftype": "numeric", "query": "How many invoices had the batch reconciler processed when it stalled? Reply with the number only.", "answer": "4271"},
   {"fid": "d01e", "ftype": "entity", "query": "Who traced the stall to the ledger writer deadlock? Reply with the name only.", "answer": "Ferrand"},
   {"fid": "d01g", "ftype": "negation", "query": "Was the archived replica affected? Answer yes or no.", "answer": "no"},
   {"fid": "d01p", "ftype": "preference", "query": "Which retry strategy did the team agree to use going forward? Reply with the strategy only.", "answer": "exponential backoff"}]},
 {"doc_id": "d02", "text": (
   "Minutes, Harkway migration sync. The cutover rehearsal completed in 96 minutes, "
   "which is inside the maintenance window but leaves little slack. Oyelaran will own "
   "the rollback runbook and present it Thursday. The legacy exporter is not being "
   "carried into the new stack; its consumers must move to the streaming feed. For "
   "schema versioning the group settled on date-stamped tags instead of semantic "
   "versions, mainly to simplify the audit trail."),
  "facts": [
   {"fid": "d02n", "ftype": "numeric", "query": "How many minutes did the cutover rehearsal take? Reply with the number only.", "answer": "96"},
   {"fid": "d02e", "ftype": "entity", "query": "Who owns the rollback runbook? Reply with the name only.", "answer": "Oyelaran"},
   {"fid": "d02g", "ftype": "negation", "query": "Is the legacy exporter being carried into the new stack? Answer yes or no.", "answer": "no"},
   {"fid": "d02p", "ftype": "preference", "query": "Which schema versioning scheme did the group settle on? Reply with the scheme only.", "answer": "date-stamped tags"}]},
 {"doc_id": "d03", "text": (
   "Field notes from the Brenmoor site survey. Soil probes went down to 3840 "
   "millimetres before hitting the clay shelf, deeper than either previous survey. "
   "Access permissions for the north gate are held by Castellane, who should be "
   "contacted a week before any equipment move. The eastern access road is not "
   "usable for heavy vehicles after rain. If floodlights are needed the crew prefers "
   "tripod rigs over pole-mounted units because the ground is too soft for anchoring."),
  "facts": [
   {"fid": "d03n", "ftype": "numeric", "query": "How many millimetres deep did the soil probes go before hitting the clay shelf? Reply with the number only.", "answer": "3840"},
   {"fid": "d03e", "ftype": "entity", "query": "Who holds the access permissions for the north gate? Reply with the name only.", "answer": "Castellane"},
   {"fid": "d03g", "ftype": "negation", "query": "Is the eastern access road usable for heavy vehicles after rain? Answer yes or no.", "answer": "no"},
   {"fid": "d03p", "ftype": "preference", "query": "Which floodlight rig does the crew prefer? Reply with the rig type only.", "answer": "tripod"}]},
 {"doc_id": "d04", "text": (
   "Vendor review, Q2. Quilvane Systems quoted 7150 per seat for the analytics tier, "
   "which undercuts the incumbent by roughly a fifth. Their references were checked "
   "by Abernathy, who spoke to three current customers. Note that the quote does not "
   "include onboarding support; that is billed separately. Between the two contract "
   "shapes on the table, finance prefers annual prepay over monthly billing because "
   "it locks the discount for the full term."),
  "facts": [
   {"fid": "d04n", "ftype": "numeric", "query": "What per-seat price did Quilvane Systems quote for the analytics tier? Reply with the number only.", "answer": "7150"},
   {"fid": "d04e", "ftype": "entity", "query": "Who checked Quilvane's references? Reply with the name only.", "answer": "Abernathy"},
   {"fid": "d04g", "ftype": "negation", "query": "Does the Quilvane quote include onboarding support? Answer yes or no.", "answer": "no"},
   {"fid": "d04p", "ftype": "preference", "query": "Which contract shape does finance prefer? Reply with the shape only.", "answer": "annual prepay"}]},
 {"doc_id": "d05", "text": (
   "Lab notebook, enzyme stability run. The Kervalin batch retained activity for "
   "5230 minutes at room temperature before dropping below threshold, a record for "
   "this series. Sample prep was handled by Winterbourne under the new sterile "
   "protocol. The control batch was not exposed to the stabilizer at any stage. For "
   "future runs the group prefers glass vials to polymer ones, as the polymer walls "
   "appear to adsorb trace stabilizer."),
  "facts": [
   {"fid": "d05n", "ftype": "numeric", "query": "For how many minutes did the Kervalin batch retain activity at room temperature? Reply with the number only.", "answer": "5230"},
   {"fid": "d05e", "ftype": "entity", "query": "Who handled the sample prep? Reply with the name only.", "answer": "Winterbourne"},
   {"fid": "d05g", "ftype": "negation", "query": "Was the control batch exposed to the stabilizer? Answer yes or no.", "answer": "no"},
   {"fid": "d05p", "ftype": "preference", "query": "Which vial material does the group prefer for future runs? Reply with the material only.", "answer": "glass"}]},
 {"doc_id": "d06", "text": (
   "Travel logistics for the Ostrellin conference. The block booking covers 1480 "
   "room-nights across the two hotels, negotiated down from the rack rate by "
   "Marchetti. The venue shuttle does not run after midnight, so late arrivals need "
   "taxis. Between the two badge pickup options, the committee prefers hall pickup "
   "over hotel delivery because last year a third of the couriered badges arrived "
   "after the opening session."),
  "facts": [
   {"fid": "d06n", "ftype": "numeric", "query": "How many room-nights does the block booking cover? Reply with the number only.", "answer": "1480"},
   {"fid": "d06e", "ftype": "entity", "query": "Who negotiated the room rate down? Reply with the name only.", "answer": "Marchetti"},
   {"fid": "d06g", "ftype": "negation", "query": "Does the venue shuttle run after midnight? Answer yes or no.", "answer": "no"},
   {"fid": "d06p", "ftype": "preference", "query": "Which badge pickup option does the committee prefer? Reply with the option only.", "answer": "hall pickup"}]},
 {"doc_id": "d07", "text": (
   "Support escalation summary. Ticket volume for the Pellswick connector spiked to "
   "2610 open cases after the certificate rotation, most tagged as login loops. "
   "Root cause analysis is with Nakagome, who suspects a stale intermediate cert in "
   "the mobile build. The desktop client was not impacted. When the fix ships, "
   "support prefers a staged rollout to a global push, so the queue can drain "
   "between waves."),
  "facts": [
   {"fid": "d07n", "ftype": "numeric", "query": "How many open cases did the Pellswick connector spike to? Reply with the number only.", "answer": "2610"},
   {"fid": "d07e", "ftype": "entity", "query": "Who has the root cause analysis? Reply with the name only.", "answer": "Nakagome"},
   {"fid": "d07g", "ftype": "negation", "query": "Was the desktop client impacted? Answer yes or no.", "answer": "no"},
   {"fid": "d07p", "ftype": "preference", "query": "Which rollout style does support prefer for the fix? Reply with the style only.", "answer": "staged"}]},
 {"doc_id": "d08", "text": (
   "Grant progress note. The imaging pipeline has processed 6890 scans since the "
   "January refresh, roughly double the original projection. The ethics amendment "
   "was drafted by Villanoro and is with the board. Raw scans are not leaving the "
   "secure enclave under the amended protocol; only derived features are exported. "
   "For the follow-up cohort the team prefers rolling enrollment over a fixed intake "
   "window, to smooth scanner utilisation."),
  "facts": [
   {"fid": "d08n", "ftype": "numeric", "query": "How many scans has the imaging pipeline processed since the January refresh? Reply with the number only.", "answer": "6890"},
   {"fid": "d08e", "ftype": "entity", "query": "Who drafted the ethics amendment? Reply with the name only.", "answer": "Villanoro"},
   {"fid": "d08g", "ftype": "negation", "query": "Are raw scans leaving the secure enclave under the amended protocol? Answer yes or no.", "answer": "no"},
   {"fid": "d08p", "ftype": "preference", "query": "Which enrollment style does the team prefer for the follow-up cohort? Reply with the style only.", "answer": "rolling"}]},
 {"doc_id": "d09", "text": (
   "Kitchen renovation walkthrough. The contractor's revised bid came in at 8460 "
   "for the cabinetry alone, absorbing the plywood surcharge. Permits are being "
   "expedited by Solheim at the county office. The load-bearing wall between kitchen "
   "and pantry is not being moved in this plan. On finishes, the owners prefer "
   "matte laminate to gloss, mostly because the showroom gloss samples showed every "
   "fingerprint."),
  "facts": [
   {"fid": "d09n", "ftype": "numeric", "query": "What was the revised bid for the cabinetry alone? Reply with the number only.", "answer": "8460"},
   {"fid": "d09e", "ftype": "entity", "query": "Who is expediting the permits at the county office? Reply with the name only.", "answer": "Solheim"},
   {"fid": "d09g", "ftype": "negation", "query": "Is the load-bearing wall between kitchen and pantry being moved in this plan? Answer yes or no.", "answer": "no"},
   {"fid": "d09p", "ftype": "preference", "query": "Which finish do the owners prefer? Reply with the finish only.", "answer": "matte laminate"}]},
 {"doc_id": "d10", "text": (
   "Fleet maintenance digest. Odometer audits flagged unit 47 at 9130 kilometres "
   "past its service interval, the worst in the fleet. Scheduling the catch-up "
   "services falls to Britvang this cycle. The refrigerated trailers are not "
   "included in this audit round; they run on a separate calendar. For the overdue "
   "units the depot prefers weekend servicing to weekday slots, to keep route "
   "coverage intact."),
  "facts": [
   {"fid": "d10n", "ftype": "numeric", "query": "How many kilometres past its service interval was unit 47? Reply with the number only.", "answer": "9130"},
   {"fid": "d10e", "ftype": "entity", "query": "Who is scheduling the catch-up services this cycle? Reply with the name only.", "answer": "Britvang"},
   {"fid": "d10g", "ftype": "negation", "query": "Are the refrigerated trailers included in this audit round? Answer yes or no.", "answer": "no"},
   {"fid": "d10p", "ftype": "preference", "query": "Which servicing slot does the depot prefer for the overdue units? Reply with the slot only.", "answer": "weekend"}]},
 {"doc_id": "d11", "text": (
   "Editorial planning memo. The longform piece on the Averlane inquiry is budgeted "
   "at 5470 words, our longest this quarter. Fact-checking is assigned to Okonjima, "
   "who has the source archive. The leaked committee transcript is not being quoted "
   "directly on legal advice; we paraphrase and cite the hearing record. For "
   "publication timing the desk prefers a Tuesday morning slot over the weekend "
   "edition, where longform tends to get buried."),
  "facts": [
   {"fid": "d11n", "ftype": "numeric", "query": "How many words is the Averlane longform piece budgeted at? Reply with the number only.", "answer": "5470"},
   {"fid": "d11e", "ftype": "entity", "query": "Who is assigned to fact-checking? Reply with the name only.", "answer": "Okonjima"},
   {"fid": "d11g", "ftype": "negation", "query": "Is the leaked committee transcript being quoted directly? Answer yes or no.", "answer": "no"},
   {"fid": "d11p", "ftype": "preference", "query": "Which publication slot does the desk prefer? Reply with the slot only.", "answer": "Tuesday morning"}]},
 {"doc_id": "d12", "text": (
   "Observatory scheduling note. The spectrograph queue stands at 3220 exposure "
   "requests after the call closed, half again over allocation. Time-assignment "
   "arbitration goes to Ferrantelli for this semester. The damaged guide camera is "
   "not back in service; targets needing fine guiding should defer. Among the "
   "backlog strategies discussed, the panel prefers shortening default exposures "
   "over cutting whole programs."),
  "facts": [
   {"fid": "d12n", "ftype": "numeric", "query": "How many exposure requests are in the spectrograph queue? Reply with the number only.", "answer": "3220"},
   {"fid": "d12e", "ftype": "entity", "query": "Who handles time-assignment arbitration this semester? Reply with the name only.", "answer": "Ferrantelli"},
   {"fid": "d12g", "ftype": "negation", "query": "Is the damaged guide camera back in service? Answer yes or no.", "answer": "no"},
   {"fid": "d12p", "ftype": "preference", "query": "Which backlog strategy does the panel prefer? Reply with the strategy only.", "answer": "shortening default exposures"}]},
 {"doc_id": "d13", "text": (
   "Warehouse relocation update. The racking teardown is quoted at 1840 labour "
   "hours, spread over three weekends. Forklift certification for the new site is "
   "coordinated by Aldenhoven. The mezzanine stock is not moving in phase one; it "
   "ships with the office fit-out in phase two. For pallet labelling the ops team "
   "prefers preprinted rolls to on-demand printing, after the printhead failures "
   "during the last move."),
  "facts": [
   {"fid": "d13n", "ftype": "numeric", "query": "How many labour hours is the racking teardown quoted at? Reply with the number only.", "answer": "1840"},
   {"fid": "d13e", "ftype": "entity", "query": "Who coordinates forklift certification for the new site? Reply with the name only.", "answer": "Aldenhoven"},
   {"fid": "d13g", "ftype": "negation", "query": "Is the mezzanine stock moving in phase one? Answer yes or no.", "answer": "no"},
   {"fid": "d13p", "ftype": "preference", "query": "Which pallet labelling approach does the ops team prefer? Reply with the approach only.", "answer": "preprinted rolls"}]},
 {"doc_id": "d14", "text": (
   "Localization sprint recap. The Veytari release shipped with 2790 strings "
   "translated, clearing the launch bar with margin. Glossary disputes get settled "
   "by Szablewski as term owner. The marketing site is not covered by this sprint; "
   "it runs on the agency contract. Between the two QA passes available, the team "
   "prefers in-context review to spreadsheet review, since layout truncation only "
   "shows up in context."),
  "facts": [
   {"fid": "d14n", "ftype": "numeric", "query": "How many strings shipped translated in the Veytari release? Reply with the number only.", "answer": "2790"},
   {"fid": "d14e", "ftype": "entity", "query": "Who settles glossary disputes as term owner? Reply with the name only.", "answer": "Szablewski"},
   {"fid": "d14g", "ftype": "negation", "query": "Is the marketing site covered by this sprint? Answer yes or no.", "answer": "no"},
   {"fid": "d14p", "ftype": "preference", "query": "Which QA pass does the team prefer? Reply with the pass only.", "answer": "in-context review"}]},
 {"doc_id": "d15", "text": (
   "Community garden AGM notes. The seed exchange raised 1360 in donations, the "
   "best year since the shed rebuild. Plot reassignments are administered by "
   "Quennehen under the waiting-list rules. The rainwater tanks are not connected "
   "to the drip lines yet; hand watering continues until the pump is fitted. For "
   "the shared beds the members prefer raised planters over ground beds, given the "
   "drainage problems on the low side."),
  "facts": [
   {"fid": "d15n", "ftype": "numeric", "query": "How much did the seed exchange raise in donations? Reply with the number only.", "answer": "1360"},
   {"fid": "d15e", "ftype": "entity", "query": "Who administers plot reassignments? Reply with the name only.", "answer": "Quennehen"},
   {"fid": "d15g", "ftype": "negation", "query": "Are the rainwater tanks connected to the drip lines yet? Answer yes or no.", "answer": "no"},
   {"fid": "d15p", "ftype": "preference", "query": "Which bed style do the members prefer for the shared beds? Reply with the style only.", "answer": "raised planters"}]},
 {"doc_id": "d16", "text": (
   "Compliance audit readout. Sampling covered 4520 transactions from the payments "
   "corridor, with two soft findings and no criticals. Remediation tracking sits "
   "with Iverstadt until closure. The archival cold store was not in scope this "
   "cycle. On evidence collection, the auditors prefer system-generated exports to "
   "screenshots, which they flagged as tamper-prone."),
  "facts": [
   {"fid": "d16n", "ftype": "numeric", "query": "How many transactions did the sampling cover? Reply with the number only.", "answer": "4520"},
   {"fid": "d16e", "ftype": "entity", "query": "Who holds remediation tracking until closure? Reply with the name only.", "answer": "Iverstadt"},
   {"fid": "d16g", "ftype": "negation", "query": "Was the archival cold store in scope this cycle? Answer yes or no.", "answer": "no"},
   {"fid": "d16p", "ftype": "preference", "query": "Which evidence collection method do the auditors prefer? Reply with the method only.", "answer": "system-generated exports"}]},
]
