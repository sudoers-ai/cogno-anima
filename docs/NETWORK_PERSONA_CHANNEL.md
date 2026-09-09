# Contract — network, persona, channel: what this library owns, and what it only believes

**Status: CONTRACT.** It describes what the code in this repo GUARANTEES, verified against the
code at `4dd7cf7`. Where the guarantee is weaker than the prose around it, this document says
so — a contract that describes an intention is worth nothing.

Three things have grown across the Cogno system and were written down only in passing:

* a **persona** — who answers, with what voice, tools and limits;
* a **channel** — how a contact reaches it (WhatsApp, Telegram, webchat, the CLI);
* the **network** between them — a hub that consults a specialist mid-turn, a transfer that
  moves the conversation.

**All three are decided by the host.** `cogno-anima` owns a carrier for one of them, a metakey
for another, and *nothing at all* for the third — and the third being empty is the contract, not
a gap. The host's half is `cogno-host/docs/NETWORK_PERSONA_CHANNEL.md`; the two are meant to be
read together and neither repeats the other.

The rule this file exists to keep is the repo's own: **the core signals, the host decides.** The
recurring defect is its shadow — a layer re-deriving a decision another layer already made. Each
section below therefore ends with *who decides*, and that line is the load-bearing one.

---

## 1. The NETWORK — one carrier, one name set, no graph

### 1.1 The intra-turn CONSULT

`PipelineContext.consult_result: Optional[EgoResult]` (`cogno_anima/types.py`) is the whole of
this library's model of a second executor: a hub persona asked a specialist MID-TURN and
consolidated her answer into its own reply.

**What it IS.** Her `EgoStage` trace, unmerged. The tools she ran are THIS turn's tools, so the
carrier enters the source walk every write predicate shares (`types._any_execution`) and
`committed_this_turn`, `wrote_for_the_contact` and `write_attempted_this_turn` all see it — in
one edit, because a source only one of them reads is the defect that walk exists to prevent.
`tests/unit/test_consult_is_the_second_source.py` exercises all three and, since this change,
DERIVES the list from the module so a fourth predicate cannot arrive unpinned.

**What the core may assume.** That the host ran the specialist and handed back her result. That
is all. It never runs one, never chooses one, never knows one exists until the field is set.

**What the core may NOT assume, stated because each has a reader that would otherwise be wrong:**

* **That the field is a trace at all.** Every read goes through a sentinel inside a `try`
  (`_consult_source`, `SuperegoStage._consulted_calls`). An unreadable record degrades to the
  STRICT direction and asserts nothing — "she executed nothing" is a claim about the world, and
  a guard must not make one it failed to read.
* **That `EgoResult.persona` is set.** It is `Optional[str]`, described at its declaration as an
  *opaque host label (trace/billing)*. There is no closed vocabulary, no validation, and no
  requirement that it name a persona anybody has heard of. The judge prompt renders
  `persona not named` when it is absent, and flattens/caps it when present precisely because it
  is untrusted configuration landing next to real headings.
* **That the host also appended her calls to `turn_executions`.** Harmless if it does (the walk
  is a union), not required, and keeping them apart is what lets a reader tell her calls from a
  discarded attempt's.
* **That `None` and an empty trace mean the same thing.** They are opposite facts: `None` says
  NOBODY WAS CONSULTED, an empty `EgoResult` says a specialist was consulted and came back with
  nothing. Only the second grounds a draft that says so. This is why the carrier is an
  `Optional[EgoResult]` and not a `list[ToolExecution]`.

**Cost** rides on `retry_metrics`, not on a sixth stage slot: a consult is a second EXECUTOR,
not a stage, and `stage_metrics` is an enumeration every reader of `seq` depends on.

### 1.2 The TRANSFER

**The core does not model a transfer at all.** It has no concept of moving a conversation, no
lock, no target, no give-back. It knows exactly one thing about the class of tools that do it:
`metakeys.ROUTING_ONLY_TOOLS`, a set of tool NAMES the host declares, whose successful execution
changes nothing the contact can see.

It is read by `wrote_for_the_contact` and DELIBERATELY ignored by `committed_this_turn`, because
they are two questions:

| question | predicate | a transfer counts? |
|---|---|---|
| did the contact's world change? | `wrote_for_the_contact` | **no** — only who, on our side, answers next |
| did anything happen that makes REPEATING unsafe? | `committed_this_turn` | **yes** — a cached reply replayed without it promises a handoff that never happens |

**Absent means "declare nothing"**, and `wrote_for_the_contact` then answers exactly like
`committed_this_turn`. That default is not caution — it is the only one available: the same
value is right for one consumer and wrong for another, so no filter can be applied by default to
all of them.

### 1.3 Who decides

The host, on every axis, and the core holds no opinion it could contradict:

| decision | where it is made |
|---|---|
| whether a consult runs at all | host (it is behind a deploy flag there) |
| which persona is consulted, and whether one may consult another | host |
| what the specialist is allowed to do while consulted | host (it stamps the read-only mask and the step budget) |
| which tools are "routing only" | host, through `mk.ROUTING_ONLY_TOOLS` |
| what a transfer does to the conversation afterwards | host — the core never sees it |

### 1.4 The gap, named rather than implied

The comment on `consult_result` says *provenance travels with the data*, and that sentence is
TRUE OF THIS CARRIER and of the judge prompt built from it: her calls are rendered in a section
of their own, under her name, never folded into `# What the EGO executed`.

**It is not true of everything downstream, and a reader of this repo alone would conclude that
it is.** In `cogno-host`, `company_focus.executions_of` merges the consult's calls, the turn
accumulator and the surviving attempt into ONE flat list, and `grounding._tool_calls` builds its
`_Call` records from it with no persona field at all — so every anti-fabrication net grants the
hub its exemptions off calls the specialist made. That is a host-side fact and a host-side fix;
what belongs here is that this library's carrier keeps a distinction its most important consumer
does not, and that nobody should read the comment as a system-wide guarantee. The declaration
now says so at both sites.

---

## 2. The PERSONA — four strings and a label

**What a persona IS to this library:** the arguments of a call. Nothing more.

| what arrives | how | who owns it |
|---|---|---|
| the execution prompt | `EgoStage.process(..., system_prompt=)` | host |
| the scope / limits / voice prompts | `SuperegoStage.check_input_scope/evaluate/voice(..., *_prompt=)` | host |
| an opaque label | `mk.EGO_PERSONA` → `EgoResult.persona` | host |
| declared voice traits | `mk.VOICE_TRAITS` | host |

**What the core may assume.** That the strings are the text to run. It does not load them, store
them, cache them, version them or know where they came from — `cogno_anima.prompts.load_prompt`
exists for the library's OWN stage prompts (NOUMENO, NER), never for a persona's.

**What the core may NOT assume:**

* **That the label names a real persona.** See §1.1 — it is opaque, optional and untrusted.
* **That the traits are valid.** `vocab.sanitize_voice_traits` is applied at the door on every
  turn: unknown values, both sides of a contradicting axis, a JSON-array string, garbage — all
  handled, never raising. That the host's admin API refuses at save time exactly what this drops
  is not a second copy of the rule: **both ends call this same function**, so they cannot
  disagree about what is valid; they differ only in the reaction (422 there, a logged drop here).
* **That the prompt it received is the persona's configured text.** The host assembles it from a
  shipped file plus tenant appendices. The core cannot tell them apart and does not try;
  `SuperegoResult.prompt_blocks` and `prompt_text` exist so a host can answer that question
  after the fact about the VOICE prompt, and they are an inventory of what was sent, never a
  claim about where it came from.

**Who decides:** the host, entirely. There is no persona registry, no default persona, no
notion of a persona being enabled, and no place where this library could disagree with the host
about who is answering.

---

## 3. The CHANNEL — the core is blind, on purpose

**`cogno-anima` has no concept of a channel.** No field on any type, no metakey, no vocabulary,
no branch. `grep -ri "channel"` over `cogno_anima/` returns prose only: illustrative examples in
comments ("measured on a real WhatsApp conversation"), and the word used in its other sense (the
confirmation *channel* a skill's own context provides). That is grep-provable and it is the
contract: **the library cannot behave differently depending on how a contact reached it, because
it is never told.**

Two consequences worth stating, because both look like oversights and are decisions:

* **Mobile brevity, markup and delivery shaping are the host's.** The SUPEREGO writes a reply in
  the persona's voice and limits; how it is chunked, marked up or narrowed for a small screen is
  applied after this library has finished.
* **The outgoing-PII rule decides on PROVENANCE and on the READER, never on the DESTINATION.**
  `security/redaction.py` asks where a value came from (this turn's input, the session allowlist)
  and who is reading (`mk.PII_READER_ROLE`). It does not ask what transport the reply is about to
  cross. The metakey's own comment already records the dissent this creates — a value re-emitted
  onto a chat transport ends up in the provider's database and the tenant's history — and widens
  it deliberately, because the alternative measured worse.

**What the core may NOT assume about the reader role.** That it was verified. It arrives as a
string the host stamps; `sanitize_reader_role` coerces a non-`str` to `""`, and absent or blank
reads as GUEST — the stricter side, so a host that forgets the key gets more masking and never a
silent widening. **In `cogno-host` that role is, on every contact-facing channel but one, an
attribute of a row looked up by an identifier the sender asserted and nothing verified.** The
host contract enumerates which channel is which. Nothing here should be read as implying the
core received an authenticated principal: it received a string.

**Who decides:** the host. Which channels exist, what each one verifies, and what the resulting
role is worth are all questions this library is structurally unable to ask.

---

## 4. What this contract does NOT settle

Written down rather than invented, because the honest answer is that the code does not decide
these yet:

* **Whether a consulted turn should be judged under the hub's limits or the specialist's.** The
  judge is handed one `limits_prompt`. On a consulted turn half the execution belonged to
  somebody else, and the core has no way to tell which text it was given. `cogno-host` fixed the
  equivalent mismatch for its delegation lock; the consult reintroduces the same shape and
  neither repo resolves it.
* **Whether a specialist's read may ground the hub's claim.** §1.4. The carrier keeps the
  distinction and the most important consumer discards it. Naming the reader that must change is
  as far as this document goes.
