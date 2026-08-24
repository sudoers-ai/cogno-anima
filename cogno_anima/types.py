from __future__ import annotations

from typing import Optional, Any
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


class ToolExecution(BaseModel):
    """One tool call + its result, as recorded in the EGO trace."""
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""                     # = ToolResult.output ("" when the call was blocked)
    ok: bool = True
    error: Optional[str] = None
    side_effect: bool = False


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
        return any(t.side_effect for t in self.tools_executed)


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
    # `preserved:mutated_in_output`).
    adjustments: list[str] = Field(default_factory=list)
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


def committed_this_turn(ctx: "PipelineContext") -> bool:
    """Did this turn SUCCESSFULLY run a mutating tool, on any attempt?

    `ok` is required: a mutation that FAILED (the slot was taken between propose and commit)
    changed nothing. `side_effect` is a host hint set per tool NAME, so a no-op ("already
    confirmed — no change was made") answers True here; that is the fail-CLOSED direction on
    purpose — a needless human check is recoverable, telling a user "nothing happened" over a
    row that changed is not.

    Falls back to `ego_result` for a turn whose orchestrator does not accumulate (a single-shot
    pipeline, or a host that reconstructs a context).

    Read with `getattr`, deliberately: this is a POLICY predicate on the hot path of hosts that
    pass duck-typed carriers (test doubles, replayed traces, a leaner context of their own). It
    must answer "did anything change?" for those too, and an AttributeError raised here would
    kill a turn whose reply was already produced — the failure mode is the opposite of the
    conservatism it exists to provide."""
    execs = getattr(ctx, "turn_executions", None)
    if not execs:
        ego = getattr(ctx, "ego_result", None)
        execs = getattr(ego, "tools_executed", None) or []
    return any(getattr(t, "side_effect", False) and getattr(t, "ok", False) for t in execs)


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
