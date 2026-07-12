# Real-content corpus for the external-validity anchor (e14). Six realistic workplace
# documents; each embeds 8 verifiable facts (2 numeric, 2 entity, 2 negation incl. one
# permitted, 2 preference) WOVEN INTO coherent narrative prose -- no templated fact
# lists, no synthetic filler. Fact sentences appear verbatim in the text (checked by
# test below); numbers, names, and preference tokens are unique within each document.
from facts import Fact

DOCS = []


def _doc(doc_id, paragraphs, facts):
    text = "\n\n".join(paragraphs)
    fs = []
    for i, (ftype, stmt, query, answer) in enumerate(facts):
        assert stmt in text, f"{doc_id}: fact sentence not verbatim in doc: {stmt[:50]}"
        fs.append(Fact(f"{doc_id}_f{i}", ftype, stmt, query, answer))
    DOCS.append({"doc_id": doc_id, "text": text, "facts": fs})


_doc("incident", [
    "POSTMORTEM DRAFT — checkout outage, Tuesday. The payments API began rejecting "
    "card authorizations around 09:20 after the schema migration rolled out. Okafor "
    "was incident commander for the duration of the event. Peak latency hit 4620 "
    "milliseconds before the rollback completed. The refund backlog reached 2875 "
    "transactions by the time the queue drained, and support is still working through "
    "escalations from the morning.",
    "Brennan handled external communications and will own the customer-facing summary. "
    "Two constraints for the follow-up work: the change freeze is still active, so it "
    "is not permitted to modify the reconciliation cron until finance signs off. The "
    "retry dashboard, however, may be modified freely while we tune the alert "
    "thresholds.",
    "For the postmortem format, the VP of engineering prefers timeline over "
    "narrative, so structure the doc around the incident clock. For paging, the "
    "on-call rotation prefers pager alerts rather than email, which buried two early "
    "warnings on Tuesday."],
    [("numeric", "Peak latency hit 4620 milliseconds before the rollback completed.",
      "What was the peak latency in milliseconds? Reply with the number only.", "4620"),
     ("numeric", "The refund backlog reached 2875 transactions",
      "How many transactions were in the refund backlog? Reply with the number only.", "2875"),
     ("entity", "Okafor was incident commander for the duration of the event.",
      "Who was the incident commander? Reply with the name only.", "Okafor"),
     ("entity", "Brennan handled external communications and will own the customer-facing summary.",
      "Who handled external communications? Reply with the name only.", "Brennan"),
     ("negation", "it is not permitted to modify the reconciliation cron until finance signs off",
      "Is it permitted to modify the reconciliation cron right now? Answer yes or no.", "no"),
     ("negation", "The retry dashboard, however, may be modified freely",
      "Is it permitted to modify the retry dashboard? Answer yes or no.", "yes"),
     ("preference", "For the postmortem format, the VP of engineering prefers timeline over narrative",
      "For the postmortem format, what does the VP prefer? Reply with the preference only.", "timeline"),
     ("preference", "the on-call rotation prefers pager alerts rather than email",
      "For alerts, what does the on-call rotation prefer? Reply with the preference only.", "pager")])

_doc("sprint", [
    "SPRINT 41 KICKOFF NOTES — mobile checkout squad. Vasquez is sprint lead this "
    "cycle and will run planning and the mid-sprint review. The main goal is the "
    "one-tap payment flow behind a flag. The beta cohort is 1780 users, drawn from "
    "the loyalty program, and we will not expand it until crash telemetry is clean. "
    "The release train is capped at 6240 build minutes this cycle, so batch your CI "
    "runs and avoid speculative rebuilds.",
    "Odum owns QA sign-off for the payment flow and needs test devices reserved by "
    "Wednesday. Compliance reminder: it is not permitted to modify the signing "
    "keychain under any circumstances while the store review is pending. The "
    "feature-flag config may be modified freely by anyone on the squad as flags are "
    "server-side.",
    "Process notes: for standup timing, the squad prefers mornings over afternoons, "
    "before the platform sync. For tracking, Vasquez prefers Jira to Linear, so file "
    "everything against the sprint board."],
    [("numeric", "The beta cohort is 1780 users",
      "How many users are in the beta cohort? Reply with the number only.", "1780"),
     ("numeric", "The release train is capped at 6240 build minutes this cycle",
      "What is the build-minute cap for the release train? Reply with the number only.", "6240"),
     ("entity", "Vasquez is sprint lead this cycle",
      "Who is the sprint lead? Reply with the name only.", "Vasquez"),
     ("entity", "Odum owns QA sign-off for the payment flow",
      "Who owns QA sign-off? Reply with the name only.", "Odum"),
     ("negation", "it is not permitted to modify the signing keychain under any circumstances",
      "Is it permitted to modify the signing keychain? Answer yes or no.", "no"),
     ("negation", "The feature-flag config may be modified freely by anyone on the squad",
      "Is it permitted to modify the feature-flag config? Answer yes or no.", "yes"),
     ("preference", "for standup timing, the squad prefers mornings over afternoons",
      "For standup timing, what does the squad prefer? Reply with the preference only.", "mornings"),
     ("preference", "Vasquez prefers Jira to Linear",
      "For tracking, what does Vasquez prefer? Reply with the preference only.", "Jira")])

_doc("migration", [
    "MEETING MINUTES — data warehouse migration, vendor sync. Liang is the vendor-side "
    "contact and will send the revised statement of work by Friday. The migration "
    "budget is 48500 dollars all-in, including the parallel-run period. The agreed "
    "cutover window is 340 minutes on the last Saturday of the month, coordinated "
    "with the BI freeze.",
    "Petrov owns the DBA workstream and will validate row counts after each batch. "
    "Hard rule from finance: it is not permitted to modify the legacy billing schema "
    "at any point during the migration, since auditors are sampling from it. The "
    "sandbox replicas may be modified freely for testing and load rehearsal.",
    "Decisions: for sync mode, Liang prefers incremental over bulk loads, to keep "
    "the parallel-run lag small. For the rollback plan, the CTO prefers snapshot "
    "restore rather than replaying the change log."],
    [("numeric", "The migration budget is 48500 dollars all-in",
      "What is the migration budget in dollars? Reply with the number only.", "48500"),
     ("numeric", "The agreed cutover window is 340 minutes",
      "How long is the cutover window in minutes? Reply with the number only.", "340"),
     ("entity", "Liang is the vendor-side contact",
      "Who is the vendor-side contact? Reply with the name only.", "Liang"),
     ("entity", "Petrov owns the DBA workstream",
      "Who owns the DBA workstream? Reply with the name only.", "Petrov"),
     ("negation", "it is not permitted to modify the legacy billing schema at any point during the migration",
      "Is it permitted to modify the legacy billing schema? Answer yes or no.", "no"),
     ("negation", "The sandbox replicas may be modified freely for testing",
      "Is it permitted to modify the sandbox replicas? Answer yes or no.", "yes"),
     ("preference", "for sync mode, Liang prefers incremental over bulk loads",
      "For sync mode, what does Liang prefer? Reply with the preference only.", "incremental"),
     ("preference", "the CTO prefers snapshot restore rather than replaying the change log",
      "For the rollback plan, what does the CTO prefer? Reply with the preference only.", "snapshot")])

_doc("onboarding", [
    "ONBOARDING BRIEF — platform team, week one. Castillo is your onboarding mentor; "
    "book the architecture walkthrough with them first. Nwosu approves all access "
    "requests, so route IAM tickets through them rather than the helpdesk. Your "
    "namespace deploy quota is 2260 pods across the dev clusters. Note that the VPN "
    "token rotates every 7180 seconds, so long-running local sessions will need "
    "re-auth mid-afternoon.",
    "Guardrails: it is not permitted to modify the terraform state bucket by hand, "
    "ever — state surgery goes through the platform rotation. Your personal dev "
    "namespaces may be modified freely, including deleting and recreating them.",
    "Culture notes: for code review, the team prefers small pull requests over "
    "long-lived branches, ideally under two hundred lines. For documentation, "
    "Castillo prefers Notion to the wiki, and the onboarding checklist lives there."],
    [("numeric", "Your namespace deploy quota is 2260 pods across the dev clusters.",
      "What is the deploy quota in pods? Reply with the number only.", "2260"),
     ("numeric", "the VPN token rotates every 7180 seconds",
      "How often does the VPN token rotate, in seconds? Reply with the number only.", "7180"),
     ("entity", "Castillo is your onboarding mentor",
      "Who is the onboarding mentor? Reply with the name only.", "Castillo"),
     ("entity", "Nwosu approves all access requests",
      "Who approves access requests? Reply with the name only.", "Nwosu"),
     ("negation", "it is not permitted to modify the terraform state bucket by hand",
      "Is it permitted to modify the terraform state bucket by hand? Answer yes or no.", "no"),
     ("negation", "Your personal dev namespaces may be modified freely",
      "Is it permitted to modify your personal dev namespaces? Answer yes or no.", "yes"),
     ("preference", "for code review, the team prefers small pull requests over long-lived branches",
      "For code review, what does the team prefer? Reply with the preference only.", "small"),
     ("preference", "Castillo prefers Notion to the wiki",
      "For documentation, what does Castillo prefer? Reply with the preference only.", "Notion")])

_doc("designreview", [
    "DESIGN REVIEW SUMMARY — search relevance revamp. Ilves chaired the review and "
    "approved the phased plan with two amendments. The end-to-end latency budget is "
    "190 milliseconds at p95, which the reranker must fit inside. The index will be "
    "split into 1450 shards to keep per-shard fanout balanced during peak traffic.",
    "Duarte owns relevance quality and will curate the golden query set before phase "
    "one. Amendments: it is not permitted to modify the ranking weights file outside "
    "the experiment framework — weight changes must ship as experiments. The "
    "query-suggestion service may be modified freely since it is fully decoupled "
    "from ranking.",
    "Rollout: for the A/B design, Ilves prefers geo cohorts over random assignment "
    "because of cache locality. For monitoring, the panel prefers Grafana to the "
    "in-house dashboards, which are being retired."],
    [("numeric", "The end-to-end latency budget is 190 milliseconds at p95",
      "What is the end-to-end latency budget in milliseconds? Reply with the number only.", "190"),
     ("numeric", "The index will be split into 1450 shards",
      "How many shards will the index be split into? Reply with the number only.", "1450"),
     ("entity", "Ilves chaired the review",
      "Who chaired the design review? Reply with the name only.", "Ilves"),
     ("entity", "Duarte owns relevance quality",
      "Who owns relevance quality? Reply with the name only.", "Duarte"),
     ("negation", "it is not permitted to modify the ranking weights file outside the experiment framework",
      "Is it permitted to modify the ranking weights file outside the experiment framework? Answer yes or no.", "no"),
     ("negation", "The query-suggestion service may be modified freely",
      "Is it permitted to modify the query-suggestion service? Answer yes or no.", "yes"),
     ("preference", "for the A/B design, Ilves prefers geo cohorts over random assignment",
      "For the A/B design, what does Ilves prefer? Reply with the preference only.", "geo"),
     ("preference", "the panel prefers Grafana to the in-house dashboards",
      "For monitoring, what does the panel prefer? Reply with the preference only.", "Grafana")])

_doc("escalation", [
    "ESCALATION HANDOFF — Meridian Health account. Farah is the account executive "
    "and owns the renewal conversation, which is now entangled with this escalation. "
    "The contract value is 96400 dollars annually, so treat this as a retention "
    "risk. Under the support agreement, each missed-SLA incident accrues a credit of "
    "1120 dollars, and we have already accrued two this quarter.",
    "Kowalski is the support engineer on point and has the full incident timeline. "
    "Boundaries: it is not permitted to modify the customer's SSO configuration from "
    "our side — their IT team applies all identity changes. The internal ticket tags "
    "may be modified freely to keep the escalation queue readable.",
    "Communication: for status updates, the customer prefers weekly calls over "
    "email threads, Thursdays at ten. For urgent items, their platform lead prefers "
    "Slack via the shared channel rather than the ticket portal."],
    [("numeric", "The contract value is 96400 dollars annually",
      "What is the annual contract value in dollars? Reply with the number only.", "96400"),
     ("numeric", "each missed-SLA incident accrues a credit of 1120 dollars",
      "What is the credit per missed-SLA incident, in dollars? Reply with the number only.", "1120"),
     ("entity", "Farah is the account executive",
      "Who is the account executive? Reply with the name only.", "Farah"),
     ("entity", "Kowalski is the support engineer on point",
      "Who is the support engineer on point? Reply with the name only.", "Kowalski"),
     ("negation", "it is not permitted to modify the customer's SSO configuration from our side",
      "Is it permitted to modify the customer's SSO configuration? Answer yes or no.", "no"),
     ("negation", "The internal ticket tags may be modified freely",
      "Is it permitted to modify the internal ticket tags? Answer yes or no.", "yes"),
     ("preference", "for status updates, the customer prefers weekly calls over email threads",
      "For status updates, what does the customer prefer? Reply with the preference only.", "weekly"),
     ("preference", "their platform lead prefers Slack via the shared channel",
      "For urgent items, what does the platform lead prefer? Reply with the preference only.", "Slack")])
