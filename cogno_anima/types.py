from __future__ import annotations

from typing import Callable, Optional, Any
from pydantic import BaseModel, Field


class StageMetrics(BaseModel):
    """Telemetry captured during execution of one LLM call or stage.

    ``seq``/``attempt``/``prompt_sha`` make ONE CALL the addressable unit. They are stamped by
    the orchestrator (the only layer that sequences the stages) and default to inert, so a
    stage that constructs its own metrics — every stage in this package does — needs no change
    and an orchestrator that stamps nothing behaves exactly as before.

    Why they belong here: ``PipelineContext.stage_metrics`` is composed as "the populated
    canonical slots, then ``retry_metrics``", and each canonical slot holds the value that
    SURVIVED. So the list is NOT call order, and reading it as one attributes a retried turn
    backwards — on a turn the judge rejected once, the ``ego`` among the canonical slots is
    attempt 2 and the ``ego`` further down is attempt 1. Measured on a real run: two ``ego``
    entries at 3874 and 3747 prompt tokens, and reading top-to-bottom blames the wrong call
    for the write. ``seq`` is the fix at the root — sort by it and the sequence is true.
    """
    stage: str
    elapsed_ms: float
    tokens_in: int             # prompt tokens consumed by the LLM generate call
    tokens_out: int            # completion tokens produced by the LLM generate call
    # ── per-call identity (orchestrator-stamped; 0/"" = not stamped) ──────────────────
    seq: int = 0               # 1-based call order within the turn; 0 = unstamped
    attempt: int = 0           # correction-loop attempt this call belongs to; 0 = n/a
    prompt_sha: str = ""       # digest of the HOST-PARAMETRISED text handed to this call —
    #   the persona's execution/voice/limits/scope prompt plus the injected context, i.e. what
    #   a deployment can change without changing code. NOT the fully rendered prompt (the
    #   stage composes that internally from these inputs plus the ctx). That is enough to be
    #   the LABEL an outcome is grouped by; the byte-exact rendered prompt is a refinement to
    #   make only if two calls sharing a sha are ever seen to behave materially differently.
    # Embedding telemetry for stages that call an Embedder (e.g. NOUMENO's
    # subject-continuity + drift similarity). Kept separate from the generate
    # tokens so LLM vs embedding cost stay distinguishable, but folded into
    # `tokens_total` so the stage's true token cost is a single number.
    embedding_tokens: int = 0  # tokens consumed by embedding calls (0 if cached/unreported)
    embedding_calls: int = 0   # number of embed() operations performed
    tokens_total: int = 0
    model: str

    def model_post_init(self, __context: Any) -> None:
        self.tokens_total = self.tokens_in + self.tokens_out + self.embedding_tokens


class NoumenoResult(BaseModel):
    """Result of the NOUMENO layer — perception and normalization of the input."""

    # ── Texts ─────────────────────────────────────────────
    original: str               # Raw user text, untouched
    rewritten: str              # Text rewritten into English
    context_turn: str           # Short context summary ("" if 1st turn or subject changed)

    # ── Language ──────────────────────────────────────────
    language: str               # Language detected in the original (BCP-47: "pt", "en")
    canonical_language: str = "en" # Default internal language (always "en")

    # ── Drift (rewrite distortion) ────────────────────────
    drift_score: float          # 1.0 - cosine(embed(original), embed(rewritten)) → [0.0, 1.0]
    drift_tag: str              # PASS_THROUGH | REWRITTEN | COMPRESSED | EXPANDED | DRIFT
    changed: bool               # True if the LLM made active structural/semantic changes
    confidence: float           # LLM confidence in preserving the intent [0.0, 1.0]

    # ── Subject continuity ────────────────────────────────
    change_subject: bool        # True if the subject changed vs. history
    subject_similarity: float   # cosine(embed(input), embed(last_rewritten)) → [0.0, 1.0]
    context_used: bool          # True if history was used (= bool(context_turn))

    # ── Preservation ──────────────────────────────────────
    preserved_terms: list[str]  # Terms preserved intact (names, URLs, emails...)
    rewrite_warnings: list[str] # Rewrite warnings (ambiguity, potential loss...)

    # ── Telemetry ─────────────────────────────────────────
    metrics: StageMetrics


class IntentResult(BaseModel):
    """Structured result of the semantic and intent analysis (NER Stage)."""

    # ── Semantic classification ───────────────────────────
    intent_class: str           # INFORMATION_REQUEST | ACTION_REQUEST | CLARIFICATION | CREATIVE_TASK | SOCIAL | UNKNOWN
    sentiment: str              # POSITIVE | NEGATIVE | NEUTRAL | CURIOUS | FRUSTRATED | URGENT | PLAYFUL
    confidence: float           # Confidence in the mapping [0.0, 1.0]
    temporal_class: str         # RECENT | HISTORICAL | TIMELESS | MIXED
    triad_signal: str           # ID | EGO | SUPEREGO | BALANCED

    # ── Named entities ────────────────────────────────────
    entities_people: list[str] = Field(default_factory=list)
    entities_objects: list[str] = Field(default_factory=list)
    entities_concepts: list[str] = Field(default_factory=list)

    # ── Geolocation ───────────────────────────────────────
    location: Optional[str] = None

    # ── Cognitive tags ────────────────────────────────────
    mandatory_tags: list[str] = Field(default_factory=list) # NER.SYSTEM, NER.MATH...
    aristotelian: dict[str, str] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)

    # ── Causal chain and goals ────────────────────────────
    goal: Optional[str] = None
    causal_chain: list[str] = Field(default_factory=list)

    # ── Language and speech ───────────────────────────────
    parole: Optional[str] = None
    langue: Optional[str] = None

    # ── Pragmatic constraints ─────────────────────────────
    negation: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    modality: Optional[str] = None
    speech_act: Optional[str] = None
    verbs: list[str] = Field(default_factory=list)

    # ── Lexical complexity ────────────────────────────────
    context_dependent: bool = False
    is_composite: bool = False
    is_sequential: bool = False

    # ── Personally identifiable information (PII) ─────────
    pii: list[str] = Field(default_factory=list)
    pii_risk: str = "NONE"


    # ── Telemetry ─────────────────────────────────────────
    metrics: StageMetrics
    raw_response: Optional[str] = None

    def aristo_tag(self, cat: str) -> str:
        """Return the Aristotelian tag."""
        val = self.aristotelian.get(cat, "")
        return val.split(" | ")[0].strip() if " | " in val else val.strip()

    def aristo_desc(self, cat: str) -> str:
        """Return the Aristotelian description."""
        val = self.aristotelian.get(cat, "")
        return val.split(" | ")[1].strip() if " | " in val else ""

    def aristo_parsed(self) -> dict[str, tuple[str, str]]:
        """Return the dict of categories mapped to (tag, description)."""
        result: dict[str, tuple[str, str]] = {}
        for cat, val in self.aristotelian.items():
            if " | " in val:
                tag, desc = val.split(" | ", 1)
                result[cat] = (tag.strip(), desc.strip())
            else:
                result[cat] = (val.strip(), "")
        return result


class DriftMetrics(BaseModel):
    """Semantic/epistemological drift metrics of the pipeline."""
    # Stage 1: Epistemological drift (NOUMENO → NER)
    word_count_original: int
    word_count_noumeno: int
    compression_ratio: float
    aristotelian_coverage: int
    drift_score: float

    # Stages 2–5 drift. None = "stage not computed yet" (distinct from 0.0 =
    # "computed, no drift"). compute_cumulative renormalizes over the stages
    # actually populated, so cumulative is on a full [0,1] scale at any point in
    # the pipeline build-out — not deflated by the stages that don't exist yet.
    ontological_drift: Optional[float] = None   # Stage 2 (NER)
    situational_drift: Optional[float] = None    # Stage 3 (ID)
    execution_drift: Optional[float] = None      # Stage 4 (EGO)
    synthesis_drift: Optional[float] = None       # Stage 5 (SUPEREGO)

    # Cumulative
    cumulative_drift: float = 0.0
    drift_action: str = "none"  # none | warn | ask_user | self_correct

    def to_tags(self) -> list[str]:
        """Generate diagnostic tags based on drift."""
        tags: list[str] = []

        if self.drift_score >= 0.4:
            tags.append("NOUMENO.DRIFT")

        if self.compression_ratio == 1.0:
            tags.append("NOUMENO.PASS_THROUGH")
        elif self.compression_ratio < 0.8:
            tags.append("NOUMENO.COMPRESSED")
        elif self.compression_ratio > 1.3:
            tags.append("NOUMENO.EXPANDED")
        else:
            tags.append("NOUMENO.REWRITTEN")

        # Cumulative drift tags
        if self.drift_action == "ask_user":
            tags.append("DRIFT.ASK_USER")
        elif self.drift_action == "self_correct":
            tags.append("DRIFT.SELF_CORRECT")
        elif self.drift_action == "warn":
            tags.append("DRIFT.WARN")

        return tags


class IdResult(BaseModel):
    """Result of the ID layer (Stage 3) — strategic routing and continuity.

    The ID layer is 100% heuristic (no LLM call): it only uses the Embedder for
    goal similarity (when the GoalManager reaches the semantic stage).
    `metrics.tokens_in`/`tokens_out` are 0; the embedding cost shows up in
    `metrics.embedding_tokens`/`embedding_calls`.
    """

    # ── Routing ───────────────────────────────────────────
    triad_route: str                # ID | EGO | SUPEREGO | BALANCED

    # ── Goal continuity (GoalManager) ─────────────────────
    active_goal: Optional[str] = None
    goal_status: str = "NEW"        # NEW | ONGOING | COMPLETED | ABANDONED
    goal_similarity: float = 1.0    # similarity that fed compute_situational

    # ── Intentions (IntentionTracker / BDI) ───────────────
    active_intentions: list[str] = Field(default_factory=list)

    # ── Attention (AttentionFilter) ───────────────────────
    attention_focus: list[str] = Field(default_factory=list)

    # ── Safety gate ───────────────────────────────────────
    blocked: bool = False           # True when pii_risk=CRITICAL → skip EGO
    block_reason: Optional[str] = None

    # ── Cross-turn signals ────────────────────────────────
    turn_number: int = 1
    # Effective temporal after stickiness. Recorded HERE (does not mutate the
    # IntentResult): the NER is stateless and must not be rewritten by a later stage.
    temporal_class: Optional[str] = None
    emotional_override: Optional[str] = None
    complexity: str = "LOW"         # LOW | MEDIUM | HIGH | EXPERT (advisory)
    # The user framed an action tentatively (interrogative / low certainty) →
    # this is a SIGNAL only. The host decides what to do: ask the user directly,
    # or route to the EGO in read-only mode (ctx.metadata["ego_readonly"]). The
    # ID never forces the EGO — it just flags the doubt.
    needs_confirmation: bool = False
    # NOUMENO×NER confidence disagreement: the rewrite *looked* clean but the
    # intent read came back murky (or vice-versa). Not the absolute LLM confidence
    # (the core distrusts that) — the DISAGREEMENT between two stages is the robust
    # signal. SOFT/advisory — the host may pause or ask the user. (2R-C)
    confidence_divergence: bool = False
    # The NOUMENO rewriter flagged ambiguity / potential loss (rewrite_warnings):
    # the host may consider asking the user to clarify. SIGNAL only. (2R-D)
    clarification_suggested: bool = False

    # ── Telemetria ───────────────────────────────────────
    metrics: StageMetrics


class ToolResult(BaseModel):
    """What the host's ``ToolDispatcher.execute`` returns for one tool call.

    The EGO consumes this and records it into the trace; it never inspects the
    DB/MCP itself (execution is delegated — "EGO = brain, dispatcher = hands").

    - ``ok=False`` is a *recoverable* failure (bad args, business rejection): the
      EGO feeds ``error`` back into the loop so the model can self-correct. The
      host catches arg/validation exceptions and surfaces them this way.
    - A *fatal* failure (infra) is signalled by raising ``MCPDispatchError`` from
      ``execute`` instead, never by this model.
    - ``returns_raw_json``/``compensating_tool`` deliberately do NOT exist: the
      EGO does not voice (the SUPEREGO does), and rollback/compensation is
      host-internal. ``side_effect`` is kept only for the turn's DB record.
    """
    output: str                          # tool result (text or JSON string)
    ok: bool = True                      # False = recoverable failure → fed back
    error: Optional[str] = None
    side_effect: bool = False            # host hint for the turn record (core does not act on it)
    # The skill RAN, decided it must not commit, and is asking first. It is a THIRD
    # confirmation source, and the two it joins answer different questions:
    #
    #   gate A (``mk.EGO_READONLY``)  the USER was tentative      → mask every write
    #   gate B (``requires_confirmation``)  the TOOL is destructive → hold it, never run it
    #   this field                   THIS CALL, on what the skill just read → propose
    #
    # The difference from gate B is what makes it worth having: gate B is per tool NAME and
    # decided BEFORE execution, so it cannot know that cancelling *this* appointment is two
    # hours away, or that *this* entry is a hundred times the usual one. The skill knows,
    # because it read. And because it ran, its ``output`` carries a proposal GROUNDED in real
    # data instead of a generic "are you sure?".
    #
    # A ``True`` here is a promise that NOTHING was committed. The EGO records the call with
    # ``ok=False`` for that reason: `committed_this_turn` requires ``ok`` AND ``side_effect``,
    # so a proposal can never be counted as a write however the skill fills the other fields.
    # The host confirms through the channel it already owns (the skill's own context/metadata
    # — see ``cogno_cortex.ToolContext``); the core never invents an argument name, because
    # what a skill needs in order to commit is the skill's business, not the pipeline's.
    #
    # The hold is per CALL, not per TURN, and that is worth stating because the name suggests
    # otherwise: it stops the LOOP, not the STEP. A step carrying two calls holds the one that
    # asked and still runs its sibling, so a turn can report "holding" and "committed" at once.
    # Inherited from gate B, not introduced here (measured identical on the same scenario) —
    # what changes is REACH: gate B only ever fires on a tool the host declared destructive,
    # while this one can be raised by any skill on any call. Making the hold turn-wide would
    # change gate B's semantics too, so it is deliberately a separate change.
    needs_confirmation: bool = False


class ToolExecution(BaseModel):
    """One tool call + its result, as recorded in the EGO trace.

    ``side_effect`` and ``tool_mutating`` are TWO facts that used to be one field, and the
    difference is WHEN each is known:

      ``tool_mutating``  is this tool the KIND that writes? Declared per NAME, known BEFORE
                         the call, from the host's policy. A property of the tool.
      ``side_effect``    did THIS call write? Known only AFTER it ran. A property of the call.

    One field carried both until 2026-09-01, and every consumer that wanted the second had to
    remember to conjoin ``ok``. Splitting them is what lets a trace be read OFFLINE: the
    per-name fact has no other carrier once the host, the dispatcher and the manifest are gone,
    and a trace that needs a live object to be interpreted is not a trace.
    """
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""                     # = ToolResult.output ("" when the call was blocked)
    ok: bool = True
    error: Optional[str] = None
    side_effect: bool = False            # did THIS call write? (needs `ok` to mean "committed")
    # Is this tool DECLARED to write? ``None`` = nobody declared — the same "no claim about the
    # tool" direction the duplicate-in-step guard takes. Deliberately tri-state: coerced to a
    # bool, an offline reader could not tell a silence from a declaration.
    tool_mutating: Optional[bool] = None


class EgoStep(BaseModel):
    """One iteration of the EGO agent loop = the source of truth for EgoResult."""
    index: int
    path: str                            # "native" | "fallback"
    assistant_text: str = ""             # text the model emitted this step
    tool_calls: list[ToolExecution] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class EgoResult(BaseModel):
    """Output of the EGO stage (Stage 4) — the EXECUTOR contract.

    The EGO executes the task (decides + runs tools, gathers data); it does NOT
    write the user-facing reply — the SUPEREGO does that, voicing in the active
    persona. So there is no ``response``/``response_source`` here: the EGO hands
    forward the ``tools_executed`` data + a ``draft`` (the model's last text) as
    raw material for the SUPEREGO.

    ``steps`` is the single source of truth; ``tools_executed``/``draft``/
    ``has_side_effects`` are derived. Counts (``len(steps)``, ``len(tools_executed)``,
    ``attempt``) are first-class so the host can persist them to its DB.
    """

    steps: list[EgoStep] = Field(default_factory=list)

    # Calls the model proposed but the core REFUSED to execute because the tool is
    # host-classified ``requires_confirmation`` and the host had not confirmed
    # (``ctx.metadata["ego_confirmed"]``). The core holds them (a side effect must
    # not happen pre-confirmation) and signals; the host runs its confirm UX and,
    # on the next turn with ego_confirmed set, the EGO executes them.
    pending_confirmation: list[ToolExecution] = Field(default_factory=list)

    # Signals (not exceptions): the loop stopped early on a budget/convergence bound.
    interrupted: bool = False
    interrupt_reason: Optional[str] = None   # "max_steps" | "duplicate_calls"

    attempt: int = 1                         # echoed from ctx.metadata["ego_correction"]
    persona: Optional[str] = None            # opaque host label (trace/billing)

    # The tool surface this call was OFFERED, after every mask the turn applied (the tenant's
    # allow-list, the identity's RBAC scope, gate A's read-only filter). Names only.
    #
    # It answers a question `tools_executed` cannot: whether the model chose not to act, or
    # was never given the option. Those look identical downstream and have opposite fixes —
    # one is a prompt problem, the other is wiring, and this project has spent whole rounds
    # confusing the two (`[[skill-wiring-structural-defect]]`: a prompt derived from a flag
    # instead of from the real capability).
    #
    # It is also the other half of what a prompt label means for THIS stage. The offered tools
    # travel to the model — as schemas on the native path, as rendered text on the fallback —
    # so "which text ran" is not fully answered by the persona prompt alone.
    #
    # Names, not schemas: the descriptions have exactly one home (the manifest) and belong in
    # the catalog, not copied per turn into a trace.
    tools_offered: list[str] = Field(default_factory=list)

    metrics: StageMetrics

    @property
    def tools_executed(self) -> list[ToolExecution]:
        """Flatten of every tool call across steps (host/SUPEREGO/idempotency)."""
        return [t for s in self.steps for t in s.tool_calls]

    @property
    def draft(self) -> str:
        """The model's final text — the SUPEREGO's raw material, NOT the user reply."""
        return self.steps[-1].assistant_text if self.steps else ""

    @property
    def has_side_effects(self) -> bool:
        """Did any tool call in this result actually change something?

        ``ok`` is conjoined for the same reason `committed_this_turn` conjoins it: a call that
        FAILED changed nothing, whatever ``side_effect`` says. That is not a hypothetical — a
        dispatcher stamps ``side_effect`` from the tool's NATURE, decided per NAME *before* the
        call, so the field describes the TOOL and this property has to describe the RESULT.
        Read alone it answered True for a booking that was rejected because the slot was taken.
        """
        return any(t.side_effect and t.ok for t in self.tools_executed)


class ScopeCheckResult(BaseModel):
    """Output of the SUPEREGO Early Input Scope Guard (pre-EGO).

    A cheap relevance pre-filter: BLOCK clearly off-topic input before the
    expensive EGO runs. ``refusal_message`` is generated by the guard's own LLM
    call (contextual, in the persona's language); fail-open (blocked=False) on
    any classifier error — a cost guard must never refuse a legitimate user.
    """
    blocked: bool = False
    refusal_message: str = ""
    metrics: StageMetrics


class PiiFinding(BaseModel):
    """One personal datum found in the OUTGOING text, and what the provenance rule did with it.

    **It carries no value, and that is the point.** This lands on ``SuperegoResult`` and the host
    persists that into ``turn_traces``; a field holding the CPF would move the leak from the
    reply into a table the identity purge does not walk. Type + provenance + verdict answer every
    question the record exists for ("what is the net masking, and is it getting in the way")
    without any of them being answerable about a PERSON.

    ``mask`` is the replacement that was — or, under the shipped observation mode, WOULD have
    been — spliced in (``"[EMAIL REDACTED]"``); never the value it replaced. Under `enforce` it
    locates the span in the shipped text; under `observe` nothing was spliced and the reply is
    byte-identical, so read ``redacted`` for what actually happened.
    """
    pii_type: str          # `security.pii.VALID_PII_TYPES`
    provenance: str        # `vocab.VALID_PII_PROVENANCE`
    redacted: bool         # True → the span was masked before the reply left
    mask: str = ""         # what replaced it; "" when the value was allowed out


class SuperegoResult(BaseModel):
    """Output of the SUPEREGO stage (Stage 5) — judge + voicer.

    EGO=executor, SUPEREGO=locutor: the SUPEREGO **writes** the user-facing
    response from the EGO's gathered data (it does not review a pre-written one).
    Two operations (A2): ``evaluate`` (the JUDGE — approve/critique, gates the
    correction loop; criterion #1 is goal↔execution: "asked X, did X not Y") and
    ``voice`` (writes ``response`` in the persona's voice + limits). Both return
    this type; the host wires the final ``voice`` into ``ctx.superego_result``
    and the scope/judge calls into ``ctx.retry_metrics``.
    """
    response: str = ""              # final voiced response (voice) OR protection/refusal (blocked)
    approved: bool = True           # JUDGE verdict; False → host retries the EGO with `critique`
    critique: Optional[str] = None  # why execution failed → fed back as ego_correction.reason
    blocked: bool = False           # PII-CRITICAL / scope block → response is a protection message
    # The voice's audit trail: every per-turn tone hint DETECTED (`tone:*`/`style:*`/`register:*`/
    # `pii:*`/`override:*`, or the `general:review` sentinel) — including one the turn chose not to
    # RENDER —, then the traits the turn actually rendered (`trait:*`, appended after the prompt).
    # Those are the EFFECTIVE traits: the persona's declared ones after the turn's modulation, so
    # a `trait:concise` here can come from an urgent message rather than from the tenant's
    # configuration. The declared→effective mapping is logged by `SuperegoStage._modulate_traits`
    # (INFO, `event=traits_modulated`) — the list itself records what was said, not what was
    # configured. Finally any output backstop flag (`pii:flagged_in_output`,
    # `preserved:mutated_in_output`, `voice:json_unwrapped`) — the last one counts how often the
    # voicer answered in a JSON envelope, so a prompt problem cannot hide behind its own net.
    adjustments: list[str] = Field(default_factory=list)
    # Which sections the rendered VOICE prompt carried and how long each was — slugs from a
    # closed list, never the matched text, so there is no contact data here and no purge path
    # is needed. It answers the first question anyone asks about a bad reply ("did this turn
    # get the memories block? the rejection block?"), which was unanswerable after the fact
    # because the rendered prompt is deliberately not persisted. Empty on `evaluate`/
    # `check_input_scope` — only `voice` fills it.
    prompt_blocks: list[dict[str, Any]] = Field(default_factory=list)
    # The rendered voice prompt, IN MEMORY, for the turn the host is holding right now.
    #
    # It carries contact data — the user's own words, retrieved memories, the graph block — and
    # the core NEVER persists it. `mk.PROMPT_SHAS` explains why a digest of rendered text is a
    # digest of the CONTACT; the same reasoning says this string must not reach a store by
    # accident. It exists because the host cannot rebuild it (the prompt is assembled here and
    # was discarded), and a defect that only shows in one turn out of 283 cannot be diagnosed
    # from an inventory of block LENGTHS alone.
    #
    # Populated on EVERY `voice` call, not only on an interesting one: the turn that looks
    # normal is the comparison half, and a capture that only fires on the anomaly has nothing
    # to compare it against. Slice it with :meth:`SuperegoStage.voice_prompt_block` — the host
    # must not re-derive the section headers, or the two sides drift.
    prompt_text: str = ""
    # What the outgoing-PII provenance rule found and decided, one entry per detected value.
    # Empty means the detector found nothing — NOT that the rule was off (an inactive rule is
    # not a state this stage has: absence of a host allowlist only makes it stricter). Carries
    # types and verdicts, never values; see :class:`PiiFinding`.
    pii_findings: list[PiiFinding] = Field(default_factory=list)
    cot_stripped: bool = False      # a <think> block was removed
    metrics: StageMetrics


def ordered_stage_metrics(ctx: "PipelineContext") -> "list[StageMetrics]":
    """The turn's calls in the order they RAN — or the raw list when nobody said what that is.

    ``PipelineContext.stage_metrics`` is "the populated canonical slots, then ``retry_metrics``",
    and each canonical slot holds the value that SURVIVED, so its order is not call order.
    ``StageMetrics.seq`` carries the real one, stamped by whichever layer sequences the stages.

    **"Partially stamped" is a real state and this is where it gets a name.** Every canonical
    metric is constructed INSIDE its stage, which knows nothing of the turn; an orchestrator
    stamps them afterwards. So a mixed-version deployment, a host driving the stages itself, or
    a new stage nobody remembered to stamp all produce a list where some entries carry ``seq``
    and some do not. Sorting that mix is worse than not sorting it: every unstamped entry has
    ``seq == 0`` and lands in FRONT, giving an order that is true for neither half while
    looking authoritative. So the rule is all-or-nothing, defined once, here — beside the type
    it reads, for the same reason :func:`committed_this_turn` is: three repos must agree on it,
    and a rule each consumer re-derives is a rule each consumer gets to re-derive wrongly.

    Returns a new list; never mutates ``ctx``. Use :func:`is_fully_sequenced` when the caller
    needs to KNOW whether the order it got is the real one.
    """
    metrics = list(ctx.stage_metrics)
    if not is_fully_sequenced(metrics):
        return metrics
    return sorted(metrics, key=lambda m: m.seq)


def is_fully_sequenced(metrics: "list[StageMetrics]") -> bool:
    """Did every call in this list get a ``seq``? Empty is not sequenced: there is no order to
    claim, and returning True would let a caller present nothing as authoritative."""
    return bool(metrics) and all(getattr(m, "seq", 0) > 0 for m in metrics)


# Mirrors ``metakeys.PRIOR_ATTEMPT_COMMITTED``. Inlined, not imported: this module is the
# bottom of the package and importing sideways for one string is how an import cycle starts.
# ``tests/unit/test_types.py`` pins the two equal, so they cannot drift.
_PRIOR_ATTEMPT_COMMITTED = "prior_attempt_committed"

# Mirrors ``metakeys.ROUTING_ONLY_TOOLS``, inlined for the same reason and pinned by the same
# test. Read ONLY by `wrote_for_the_contact` — `committed_this_turn` ignores it on purpose.
_ROUTING_ONLY_TOOLS = "routing_only_tools"


def committed_this_turn(ctx: "PipelineContext") -> bool:
    """Did anything happen that makes REPEATING this turn unsafe? (A mutating tool ran, on any
    attempt — routing included.)

    That first line is the whole point of this paragraph: the NAME says "committed", which reads
    as "wrote", and the predicate means something wider. A summary line is often all a caller
    ever sees — an IDE hover, `help()`, a docs index — so the correction has to live there and
    not below. It counts EVERY mutating tool: a semantic cache that replays a reply whose turn
    transferred the conversation promises the contact a handoff that never happens.

    **The other question lives next door.** *"Did the CONTACT's world change?"* is
    `wrote_for_the_contact`, and the two are not interchangeable — on 2026-09-01 a turn whose
    only side-effecting call was `transfer_persona` told the ledger it had written. The split
    was made by walking the eleven callers and asking which answer each one needs; the two
    lists below are the result, and they are the record that the division exists.

    `ok` is required: a mutation that FAILED (the slot was taken between propose and commit)
    changed nothing. `side_effect` is a host hint set per tool NAME, so a no-op ("already
    confirmed — no change was made") answers True here; that is the fail-CLOSED direction on
    purpose — a needless human check is recoverable, telling a user "nothing happened" over a
    row that changed is not.

    Being per NAME, the hint is decided BEFORE the call, so it describes the TOOL and not the
    RESULT — and until 2026-09-01 both shipped dispatchers copied it onto their failure branch
    too, calling a rejected booking a write. They now stamp ``False`` there. The core does not
    police it: a source that still arrives that way is LOGGED, not corrected (see
    ``EgoStage._warn_if_effect_without_success``), because rewriting a host's declaration in
    silence decides for the host and hides the next source that repeats the defect.

    ``EgoResult.has_side_effects`` conjoins ``ok`` for this same reason and is now the ONLY
    other place that does — it used to read ``side_effect`` alone, which answered True over a
    turn that wrote nothing. Both readings live here and there deliberately: this repo's
    standing lesson is that a rule each consumer re-derives is a rule each consumer gets wrong
    alone, and that property was the consumer that had.

    Falls back to `ego_result` for a turn whose orchestrator does not accumulate (a single-shot
    pipeline, or a host that reconstructs a context).

    THIRD source, and it exists because the first two can both be gone. Both live ON the
    context, so both die with it — and there is a path where the context dies mid-turn: the
    host's model-routing fallback. The first attempt can commit an ordinary write and a LATER
    stage then raise; the exception takes that context and its execution record with it, and
    the retry is a fresh turn with a fresh context in which the write is nowhere. There the
    host is the only layer that knows, so it says so (``mk.PRIOR_ATTEMPT_COMMITTED``) and this
    predicate believes it.

    Fixing it HERE and not in each caller is the whole point. FIVE places CALL this, measured
    2026-09-01 rather than recalled — and it is FIVE, not eleven, because the enumeration SPLIT:
    seven callers were asking a different question and moved to `wrote_for_the_contact` (its
    docstring carries that list). What stayed are the callers for whom the answer is *"repeating
    is unsafe"*, which is what this predicate means:

      * the semantic cache (`cache.py::default_cacheable`) — an action must always re-run, and a
        cached reply replayed without its persona transfer promises a handoff that never happens;
      * TWO repair guards (`service.py::_repair_repetition` and `::_repair_grounding`) —
        re-running a turn that already acted commits a SECOND time, and a re-run transfer starts
        a second conversation;
      * the discard guard (`service.py::_finish_repair`) — an attempt that did something
        irreversible must not have its context thrown away;
      * and `wrote_for_the_contact` itself, which delegates here when the host declares no
        routing set. Named for the same reason as the others: a delegation the enumeration does
        not list is a re-derivation hiding behind a call.

    The COUNT above is a derived fact asserted in two places (here and the host's
    `ANTI_FABRICATION.md`), while the NAMES have a guard of their own. The asymmetry has a price,
    measured on this very change: a migration across three repos costs two prose edits that exist
    only to let a number catch up. Named rather than acted on — the names are the substance.

    A rule each consumer re-derives is a rule each consumer gets wrong alone.

    The last layer that RE-DERIVED the rule instead of reading it closed on 2026-08-25 (the
    owner's decision; host PR `fix/the-auditor-reads-the-declaration`): the offline promise
    auditor (`turn_audit/promises.py::committed_from_trace`) now reads the stamp
    `trace["guards"]["committed"]` — which the host's `trace.py` writes by CALLING this
    predicate, so it carries this declaration — in UNION with the execution lists. It could
    never CALL this: it reads a persisted row weeks later with no context to pass, so the stamp
    is the only shape the declaration can reach it in. The lists stay as the floor on purpose:
    0 of the 277 rows on the demo box carried the stamp when this closed, and a visible write
    must count whatever the stamp says (a degraded readout stamps False; the declaration is
    TRUE-only). What made the auditor a defect rather than a ceiling was the host's OTHER
    offline reader — the grounding replay — already taking the same key, so two halves of one
    offline layer read one row two ways; `tests/unit/test_committed_parity.py` there pinned
    the divergence with an auto-invalidating message until it closed. It was two
    re-derivations: the grounding backstop was the other, and became a caller in host #429 —
    on the very turn this path describes it now SUPPRESSES every rule instead of rewriting a
    truthful confirmation into a denial.

    The declaration is TRUE-only: absent means "nothing to add", never "nothing was committed".
    It is also PER TURN — it describes an earlier attempt of the turn being retried, and a host
    that persisted it into the next turn would disarm both repairs and the cache for the rest of
    the session.

    Read with `getattr`, deliberately: this is a POLICY predicate on the hot path of hosts that
    pass duck-typed carriers (test doubles, replayed traces, a leaner context of their own). It
    must answer "did anything change?" for those too, and an AttributeError raised here would
    kill a turn whose reply was already produced — the failure mode is the opposite of the
    conservatism it exists to provide. Same reason the metadata read below is defensive: a
    carrier whose `metadata` is missing, or is not a mapping, degrades to the trace instead of
    raising.

    Be precise about the direction of THAT degradation: it answers False, which for SEVEN of the
    eleven callers is the RELEASING answer (cacheable, re-step allowed, "nothing was committed"
    rendered to the voice as a hard rule). The other four do not release, and getting that wrong
    is easy enough that TWO drafts of this very sentence did — the second by OMISSION: when the
    caller count went 8 to 10 this half stayed at "six of the EIGHT", because only the first
    number has a test. The trace only RECORDS the answer,
    and at the grounding backstop False is the answer that ARMS the rules — True is what
    suppresses them — so the backstop belongs in neither bucket that "releasing" describes. So it is fail-OPEN here, unlike the no-op→True choice
    above, which is conservative on purpose. It stays fail-open deliberately: answering True on
    an unreadable carrier would make every test double and replayed trace read as "committed",
    which is the worse error. On a real `PipelineContext` the handler is unreachable."""
    # UNION, not "first non-empty wins". The earlier shape read `turn_executions` and consulted
    # `ego_result` only when it was EMPTY — which rests on an invariant nobody states and nothing
    # pins: "if `turn_executions` is non-empty, it is COMPLETE". The orchestrator honours it
    # today (it extends right after the EGO stage), but this predicate ships in a public lib and
    # is read by hosts with leaner carriers, replayed traces and test doubles. Measured
    # 2026-08-24: with a write in `ego_result` and only a READ in `turn_executions`, the old
    # shape answered False over a turn that wrote — and False was then the RELEASING answer for
    # six of the EIGHT callers there were at the time (2026-08-24; eleven today, seven of them
    # releasing) — in a predicate whose own first paragraph claims a fail-CLOSED bias. The
    # number is left as it was MEASURED because the sentence describes that measurement; a
    # historical count that does not say it is historical reads exactly like a forgotten one,
    # which is why this now says so.
    #
    # The union costs nothing and cannot regress: adding a source can only turn False into True,
    # never the reverse, so it moves strictly toward the conservative side. No de-duplication is
    # needed either — `any()` does not count, so an execution present in both lists is harmless.
    # Derived rather than enumerated, so a source added later joins without a second edit here.
    #
    # Each source is read LAZILY, inside its own `try`. Reading them eagerly is what a first
    # version did, and it introduced a failure mode worse than the bug it fixed:
    # `EgoResult.tools_executed` is a DERIVED property, and derived can raise. The old shape
    # touched it only when `turn_executions` was empty; touching it ALWAYS turned a turn that
    # answered True (from a complete `turn_executions`) into a turn that RAISES — fail-open
    # traded for fail-FATAL, on a turn that COMMITTED, with eight callers, one of them inside the
    # orchestrator's loop. The docstring below objects to exactly this twice.
    #
    # `continue`, not a blanket `except: return False`: a broken source must degrade to
    # "this source says nothing", never to "nothing was committed" — the second would trade
    # raising for RELEASING, which is the very error this whole change exists to remove. The
    # other source still gets its say.
    return _committed_over(ctx, _EVERY_TOOL)


def _EVERY_TOOL(_name: str) -> bool:
    """The filter `committed_this_turn` uses: it counts every tool, by design."""
    return True


def _committed_over(ctx: "PipelineContext", keep: "Callable[[str], bool]") -> bool:
    """The source walk, in ONE place, with a filter on the tool NAME.

    Both predicates read the SAME three sources with the SAME failure discipline; only the set
    of tools they count differs. Written once because the comments below are the reason this
    function is shaped the way it is, and a second copy of them is a second copy to get wrong —
    which is the very failure `committed_this_turn` was created to end.
    """
    for read in (lambda: getattr(ctx, "turn_executions", None) or [],
                 lambda: getattr(getattr(ctx, "ego_result", None),
                                 "tools_executed", None) or []):
        try:
            src = read()
            hit = any(getattr(t, "side_effect", False) and getattr(t, "ok", False)
                      and keep(str(getattr(t, "tool", "") or "")) for t in src)
        except Exception:      # noqa: BLE001 — a source that breaks must not cost the turn
            continue
        if hit:
            return True
    try:
        # The attribute read is INSIDE the try on purpose: `getattr` with a default swallows
        # AttributeError and nothing else, so a carrier whose `metadata` is a property that
        # RAISES would propagate straight through it — the same hole `_turn_metrics` documents
        # in the host. A predicate this many layers trust must never be the reason a turn is lost.
        return bool(getattr(ctx, "metadata", None).get(_PRIOR_ATTEMPT_COMMITTED))  # type: ignore[union-attr]
    except Exception:      # noqa: BLE001
        return False


def wrote_for_the_contact(ctx: "PipelineContext") -> bool:
    """Did this turn change something the CONTACT can see?

    The OTHER question `committed_this_turn` was answering without saying so. They differ on
    exactly one class of tool — the ones that move the conversation between OUR personas — and
    the difference is not cosmetic: on 2026-09-01 a turn promised to delete a ledger entry,
    deleted nothing, ran `transfer_persona` as its only side-effecting call, and told the
    ledger's `committed` stamp that it had written.

    **Which predicate a caller wants depends on what it does with the answer**, and the two
    populations are real, not hypothetical:

      * *"did the contact's world change?"* — this one. The ledger stamp, the anti-fabrication
        net, the voice cues, the streak caps (whose comment says replacing the reply would hide
        a write from the contact), soma's commit gate.
      * *"did anything happen that makes REPEATING unsafe?"* — `committed_this_turn`. The
        semantic cache and the two repair guards. **A transfer belongs there**: a cached reply
        replayed without it promises the contact a handoff that never happens, which is the
        same defect as the ledger's, pointing the other way.

    SEVEN places call this, measured 2026-09-01 — the ones that moved off `committed_this_turn`
    when the enumeration split, and each of them was WRONG before the move:

      * the ledger stamp (`trace.py::guard_outcomes`) — the defect that started this: a turn
        whose only side-effecting call was a transfer told the ledger it had written;
      * the anti-fabrication net (`grounding.py::_turn_committed`);
      * the two streak caps (`service.py::_apply_repeat_streak`, `::_apply_grounding_streak`),
        whose own comment says replacing the reply would hide a write from the contact;
      * the two voice policies (`emotion.py::_wrote_something`,
        `delivery.py::_completed_something_real`) — a lift for something real being done;
      * soma's commit gate (`pipeline.py::run_turn`), which keeps a conversation alive rather
        than dead-ending in a handoff when nothing was actually committed.

    The excluded set is HOST-DECLARED (``mk.ROUTING_ONLY_TOOLS``) because only the host knows
    which of its tools are routing — the same reason, and the same channel, as
    ``mk.PRIOR_ATTEMPT_COMMITTED`` two paragraphs down in that predicate.

    **Absent metadata means this answers exactly like `committed_this_turn`**, and that default
    is not caution: it is the only one available. The same value is RIGHT for one consumer and
    WRONG for another, so there is no filter that could be applied by default to all of them.

    KNOWN GAP, and it is named rather than papered over: the third source
    (``PRIOR_ATTEMPT_COMMITTED``) is a bare bool the host declares, carrying no tool name — so
    it CANNOT be filtered here. It is honoured anyway, because dropping it would lose a REAL
    write whose own record died with its context, and losing a real write is the failure this
    whole family exists to prevent. The fix belongs where that bool is computed.
    """
    try:
        raw = (getattr(ctx, "metadata", None) or {}).get(_ROUTING_ONLY_TOOLS) or ()
        routing = frozenset(str(x) for x in raw)
    except Exception:      # noqa: BLE001 — an unusable declaration must not cost the turn
        routing = frozenset()
    if not routing:
        return committed_this_turn(ctx)
    return _committed_over(ctx, lambda name: name not in routing)


class PipelineContext(BaseModel):
    """Carrier object that flows through the entire pipeline carrying intermediate results."""
    user_input: str
    force_language: Optional[str] = None
    
    # Results populated by stages
    noumeno: Optional[NoumenoResult] = None
    intent: Optional[IntentResult] = None
    id_result: Optional[IdResult] = None
    ego_result: Optional[EgoResult] = None
    superego_result: Optional[SuperegoResult] = None
    drift: Optional[DriftMetrics] = None
    # EVERY tool this turn executed, across every correction attempt — `ego_result` holds only
    # the attempt that SURVIVED (an orchestrator replaces it on each retry), and a write does
    # not un-happen because a later attempt only read. Six consumers were reading the survivor
    # as if it were the turn (measured 2026-08-20): the fail-closed handoff, the response
    # cache, the host's two double-commit guards, the reminder/notification producers and the
    # anti-fabrication backstop — so a write from a judge-rejected attempt could be committed
    # a SECOND time, cached as a read, left without its reminder, and have the truthful reply
    # that mentioned it rewritten into a denial.
    #
    # The orchestrator appends; stages never read this (they see their own `ego_result`). Use
    # `committed_this_turn(ctx)` for the "did anything change?" question — the definition of
    # a commit lives in ONE place because three repos must agree on it.
    turn_executions: list[ToolExecution] = Field(default_factory=list)
    
    # Custom metadata for the host to pass/read business or infra context
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Control info — terminal signal for the host (see vocab.VALID_STOP_REASONS).
    stop_reason: str = "completed"
    # Escalation signal: the core sets this (e.g. host on retry exhaustion, or a
    # confidence-floor trigger) and the HOST performs the actual handoff to a
    # human. Pairs with stop_reason="human_handoff".
    needs_handoff: bool = False
    
    # Accumulators
    retry_metrics: list[StageMetrics] = Field(default_factory=list)

    @property
    def noumeno_metrics(self) -> Optional[StageMetrics]:
        return self.noumeno.metrics if self.noumeno else None

    @property
    def ner_metrics(self) -> Optional[StageMetrics]:
        return self.intent.metrics if self.intent else None

    @property
    def id_metrics(self) -> Optional[StageMetrics]:
        return self.id_result.metrics if self.id_result else None

    @property
    def ego_metrics(self) -> Optional[StageMetrics]:
        return self.ego_result.metrics if self.ego_result else None

    @property
    def superego_metrics(self) -> Optional[StageMetrics]:
        return self.superego_result.metrics if self.superego_result else None

    @property
    def stage_metrics(self) -> list[StageMetrics]:
        base = [self.noumeno_metrics, self.ner_metrics, self.id_metrics,
                self.ego_metrics, self.superego_metrics]
        return [m for m in base if m is not None] + self.retry_metrics

    @property
    def total_tokens(self) -> int:
        """Total tokens across all stages, including embedding tokens."""
        return sum(m.tokens_total for m in self.stage_metrics)

    @property
    def total_llm_tokens(self) -> int:
        """LLM generate tokens only (prompt + completion), excluding embeddings."""
        return sum(m.tokens_in + m.tokens_out for m in self.stage_metrics)

    @property
    def total_embedding_tokens(self) -> int:
        """Tokens consumed by embedding calls across all stages."""
        return sum(m.embedding_tokens for m in self.stage_metrics)

    @property
    def total_elapsed_ms(self) -> float:
        return sum(m.elapsed_ms for m in self.stage_metrics)
