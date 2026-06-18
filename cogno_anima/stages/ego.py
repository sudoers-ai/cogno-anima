"""
cogno_anima.stages.ego — EgoStage: executor & tool dispatch (Stage 4).

EGO = executor, SUPEREGO = locutor. The EGO runs an agent loop — decide a tool,
call it via the host ``ToolDispatcher``, feed the result back, repeat — and
gathers the data; it does NOT write the user-facing reply (the SUPEREGO voices
it). So the output (`EgoResult`) is a *trace* (steps + tools_executed) plus a
``draft`` (the model's last text) for the SUPEREGO to voice — never a final
response.

Dual-path: a backend that satisfies ``ToolCallingBackend`` uses native function
calling; any plain ``LLMBackend`` (a stub, the distilled student) uses the
text-fallback path (``<TOOL_CALL>`` tags parsed by ``parse_tool_calls_from_text``).
Execution is delegated to the host dispatcher, so atomicity/rollback/outbox are
host concerns and the core never touches the DB.

Errors: a recoverable tool failure (``ToolResult(ok=False)``) is fed back so the
model self-corrects; a fatal one (the dispatcher raises ``MCPDispatchError``)
propagates; a stray exception is wrapped in ``ToolExecutionError`` and
propagated (the EGO never guesses recoverability). Budget/convergence bounds
(`max_steps`, duplicate calls) are signals: `interrupted=True` + a partial result.
"""

from __future__ import annotations

import time
import json
import hashlib
import logging
from typing import Optional

from cogno_anima.types import (
    PipelineContext,
    StageMetrics,
    ToolExecution,
    EgoStep,
    EgoResult,
)
from cogno_anima.llm import LLMBackend
from cogno_anima.llm.base import ToolCallingBackend
from cogno_anima.llm.tool_parsing import parse_tool_calls_from_text
from cogno_anima.tools import ToolDispatcher, ToolPolicyDispatcher
from cogno_anima.errors import MCPDispatchError, ToolExecutionError

logger = logging.getLogger("cogno_anima.ego")

STAGE_NAME = "ego"

# How to call tools on the text-fallback path (omitted on native FC — the API
# carries the tool format). The persona prompt must NOT contain this; the core
# owns it and never edits the host's text.
_TOOL_MECHANICS = (
    "# Tool calls\n"
    "To use a tool, emit EXACTLY one block per call, nothing else around it:\n"
    '<TOOL_CALL>{"tool": "<name>", "args": {<json args>}}</TOOL_CALL>\n'
    "Call tools as needed. When you are done, reply with your final answer and "
    "no <TOOL_CALL> block."
)


class EgoStage:
    """The executor. One LLM-driven agent loop; execution delegated to the host."""

    name = STAGE_NAME

    MAX_STEPS_DEFAULT = 5
    MAX_STEPS_COMPOSITE = 8        # multi-task request (intent.is_composite) → more loop budget
    MAX_DUPLICATE_CALLS = 2        # same (tool,args) seen this many times → block + warn
    MAX_CONSECUTIVE_BLOCKS = 2     # this many all-blocked steps in a row → abort the loop

    async def process(
        self,
        ctx: PipelineContext,
        backend: LLMBackend,
        dispatcher: ToolDispatcher,
        *,
        system_prompt: str,
    ) -> PipelineContext:
        t0 = time.perf_counter()
        if not ctx.noumeno or not ctx.intent:
            raise ValueError("NOUMENO and NER must be populated before running EgoStage")

        fc_backend: Optional[ToolCallingBackend] = (
            backend if isinstance(backend, ToolCallingBackend) and backend.supports_native_tools()
            else None
        )
        use_native = fc_backend is not None
        path = "native" if use_native else "fallback"

        # Host-declared tool classification (optional; mirrors ToolCallingBackend).
        policy: Optional[ToolPolicyDispatcher] = (
            dispatcher if isinstance(dispatcher, ToolPolicyDispatcher) else None
        )
        confirmed = ctx.metadata.get("ego_confirmed")  # host says "user confirmed"

        # ── Read-only mask (Fonte A) ──────────────────────────────────────
        # The host sets ego_readonly when the user was tentative (from the ID's
        # needs_confirmation signal). In read-only mode the EGO offers ONLY
        # non-mutating tools, so the model consults + proposes, never commits.
        # Fail-safe: no policy → mask everything (propose via draft, touch nothing).
        readonly = bool(ctx.metadata.get("ego_readonly"))
        tools = dispatcher.tools_schema()
        if readonly:
            tools = [t for t in tools if policy is not None
                     and not policy.is_mutating(t.get("function", {}).get("name", ""))]
        valid_names = {t.get("function", {}).get("name", "") for t in tools} - {""}
        # A composite (multi-task) request needs more loop budget; the host's
        # explicit ego_max_steps always wins. is_sequential only adds ordering
        # (rendered into the task context), not budget — it's a subset of composite.
        default_steps = self.MAX_STEPS_COMPOSITE if ctx.intent.is_composite else self.MAX_STEPS_DEFAULT
        max_steps = int(ctx.metadata.get("ego_max_steps", default_steps))
        # Force a tool on iteration 1 for actions — but never in read-only mode
        # (a propose turn must be free to answer/clarify instead of dispatching).
        force_first = ctx.intent.intent_class == "ACTION_REQUEST" and not readonly

        system = self._build_system(ctx, system_prompt, use_native, tools)
        task = ctx.noumeno.rewritten or ctx.user_input

        # Native keeps an OpenAI-format message list; fallback grows a text prompt.
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        user_prompt = task

        steps: list[EgoStep] = []
        pending_confirmation: list[ToolExecution] = []
        total_in = total_out = 0
        seen_calls: dict[str, int] = {}
        failed_calls: set[str] = set()
        consecutive_blocks = 0
        interrupted = False
        interrupt_reason: Optional[str] = None

        attempt_no = int(ctx.metadata.get("ego_correction", {}).get("attempt", 1))
        logger.info("EGO start path=%s tools=%d max_steps=%d attempt=%d",
                    path, len(tools), max_steps, attempt_no)

        for i in range(max_steps):
            # ── call the model ────────────────────────────────────────
            if fc_backend is not None:
                tool_choice = "required" if (i == 0 and tools and force_first) else None
                msg, ti, to = await fc_backend.chat_with_tools(messages, tools, tool_choice)
                assistant_text = msg.get("content", "") or ""
                raw_calls = msg.get("tool_calls") or parse_tool_calls_from_text(assistant_text, tools) or []
            else:
                assistant_text, ti, to = await backend.generate(system, user_prompt)
                raw_calls = parse_tool_calls_from_text(assistant_text, tools) or []
            total_in += ti
            total_out += to

            # ── natural termination: no tool calls → draft is the text ─
            if not raw_calls:
                steps.append(EgoStep(index=i, path=path, assistant_text=assistant_text,
                                     tokens_in=ti, tokens_out=to))
                break

            # ── execute / block each call ─────────────────────────────
            execs: list[ToolExecution] = []
            executed_any = False
            for tc in raw_calls:
                name, args = self._name_args(tc)
                if name not in valid_names:
                    execs.append(ToolExecution(tool=name, arguments=args, result="",
                                               ok=False, error=f"unknown tool {name!r}"))
                    continue
                sig = self._sig(name, args)
                if sig in failed_calls:
                    execs.append(ToolExecution(
                        tool=name, arguments=args, ok=False, error="blocked_retry",
                        result=(f"[BLOCKED] '{name}' with these args already FAILED. "
                                "Do NOT retry it — change the arguments, try a different "
                                "tool, or give your final answer with what you have."),
                    ))
                    continue
                if seen_calls.get(sig, 0) >= self.MAX_DUPLICATE_CALLS:
                    execs.append(ToolExecution(
                        tool=name, arguments=args, ok=False, error="duplicate",
                        result=(f"[DUPLICATE] You already called '{name}' with these exact "
                                "args. Use the data you already have to answer, or try "
                                "something different."),
                    ))
                    continue
                # ── Confirmation gate (Fonte B) ───────────────────────
                # A host-classified destructive tool must not run before the host
                # confirms. Hold it (NEVER execute), record it as pending + signal.
                if (policy is not None and policy.requires_confirmation(name)
                        and not self._is_confirmed(confirmed, name)):
                    held = ToolExecution(
                        tool=name, arguments=args, ok=False, error="needs_confirmation",
                        result=(f"[PENDING CONFIRMATION] '{name}' is destructive and was "
                                "NOT executed; it needs explicit user confirmation first."),
                    )
                    execs.append(held)
                    pending_confirmation.append(held)
                    continue
                # actually run it (delegated to the host)
                seen_calls[sig] = seen_calls.get(sig, 0) + 1
                executed_any = True
                try:
                    r = await dispatcher.execute(name, args)
                except MCPDispatchError:
                    raise                                         # fatal → propagate
                except Exception as exc:                          # stray → wrap + propagate
                    raise ToolExecutionError(name, args, exc) from exc
                if not r.ok:
                    failed_calls.add(sig)
                logger.info("EGO step=%d tool=%s ok=%s side_effect=%s", i, name, r.ok, r.side_effect)
                execs.append(ToolExecution(tool=name, arguments=args, result=r.output,
                                           ok=r.ok, error=r.error, side_effect=r.side_effect))

            steps.append(EgoStep(index=i, path=path, assistant_text=assistant_text,
                                 tool_calls=execs, tokens_in=ti, tokens_out=to))

            # ── confirmation pending → stop and propose (host confirms) ─
            if pending_confirmation:
                break

            # ── convergence guard: all-blocked steps in a row → abort ─
            if executed_any:
                consecutive_blocks = 0
            else:
                consecutive_blocks += 1
                if consecutive_blocks >= self.MAX_CONSECUTIVE_BLOCKS:
                    interrupted, interrupt_reason = True, "duplicate_calls"
                    break

            # ── feed results back for the next iteration ──────────────
            self._feed_back(use_native, messages, raw_calls, execs, assistant_text)
            if not use_native:
                user_prompt = self._extend_prompt(user_prompt, assistant_text, execs)
        else:
            interrupted, interrupt_reason = True, "max_steps"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ctx.ego_result = EgoResult(
            steps=steps,
            pending_confirmation=pending_confirmation,
            interrupted=interrupted,
            interrupt_reason=interrupt_reason,
            attempt=int(ctx.metadata.get("ego_correction", {}).get("attempt", 1)),
            persona=ctx.metadata.get("ego_persona"),
            metrics=StageMetrics(
                stage=STAGE_NAME, elapsed_ms=elapsed_ms,
                tokens_in=total_in, tokens_out=total_out, model=getattr(backend, "model", "unknown"),
            ),
        )
        logger.info("EGO done steps=%d tools=%d interrupted=%s reason=%s",
                    len(steps), len(ctx.ego_result.tools_executed), interrupted, interrupt_reason)
        return ctx

    # ── prompt assembly ──────────────────────────────────────────────

    def _build_system(
        self, ctx: PipelineContext, system_prompt: str, native: bool, tools: list[dict],
    ) -> str:
        """[host persona-exec] + [task ctx] + [host injected text] +
        [ACTIONS ALREADY EXECUTED] + [tool list + mechanics — fallback only].

        On native FC the tool schemas travel via the API, so they are NOT
        rendered into the prompt; on the fallback path the model can only see
        tools that are written here, so they (and the <TOOL_CALL> format) are.
        """
        parts: list[str] = [system_prompt.strip()]

        task_ctx = self._task_context(ctx)
        if task_ctx:
            parts.append(task_ctx)

        injected = ctx.metadata.get("ego_context")
        if injected:
            parts.append(str(injected).strip())

        actions = self._actions_already_executed(ctx)
        if actions:
            parts.append(actions)

        if not native:
            rendered = self._render_tools(tools)
            if rendered:
                parts.append(rendered)
            parts.append(_TOOL_MECHANICS)

        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _render_tools(tools: list[dict]) -> str:
        if not tools:
            return ""
        lines = ["# Available tools"]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "") or ""
            props = fn.get("parameters", {}).get("properties", {})
            sig = ", ".join(props.keys())
            lines.append(f"- {name}({sig}): {desc}")
        return "\n".join(lines)

    def _task_context(self, ctx: PipelineContext) -> str:
        intent = ctx.intent
        if not intent:
            return ""
        lines = [f"User intent: {intent.intent_class}"]
        if intent.goal:
            lines.append(f"Goal: {intent.goal}")
        if intent.domains:
            lines.append(f"Domains: {', '.join(intent.domains)}")
        if intent.entities_objects:
            lines.append(f"Entities: {', '.join(intent.entities_objects)}")
        # Pragmatic restrictions — the loop MUST honor these (host-facing NER
        # signals previously dropped). constraints = positive limits, negation =
        # things the user explicitly forbade.
        if intent.constraints:
            lines.append(f"Constraints (must respect): {', '.join(intent.constraints)}")
        if intent.negation:
            lines.append(f"Must NOT: {', '.join(intent.negation)}")
        # Order-dependent multi-task request (2R-B): tell the loop the sub-tasks
        # must run in sequence and surface the user's causal chain as a supporting
        # plan (a hint — the loop still decides the real tool order).
        if intent.is_sequential:
            lines.append(
                "Execution order: the sub-tasks are order-dependent — perform them "
                "in the sequence stated; each step may depend on the previous one."
            )
            if intent.causal_chain:
                plan = "; ".join(f"{i + 1}) {step}" for i, step in enumerate(intent.causal_chain))
                lines.append(f"Sequence (user's reasoning, supporting hint): {plan}")
        # Read-only / PROPOSE mode (Fonte A): the host masked the mutating tools
        # this turn (the user was tentative). Tell the model WHY, so it consults
        # and proposes instead of erroring on the missing write tools.
        if ctx.metadata.get("ego_readonly"):
            lines.append(
                "PROPOSE mode: gather read-only information and propose an action "
                "for the user to confirm; do NOT commit — mutating tools are "
                "intentionally unavailable this turn."
            )
        return "# Task context\n" + "\n".join(lines)

    @staticmethod
    def _is_confirmed(confirmed: object, name: str) -> bool:
        """Did the host confirm this destructive tool? ``ego_confirmed`` is either
        True (confirm all of this turn's actions) or a collection of tool names."""
        if confirmed is True:
            return True
        if isinstance(confirmed, (list, set, tuple)):
            return name in confirmed
        return False

    def _actions_already_executed(self, ctx: PipelineContext) -> str:
        """Built from the prior EgoResult on a SUPEREGO-driven retry. Renders
        whatever trace the host hands back — the core does NOT assume the prior
        actions persisted (host rollback → empty trace → fresh retry)."""
        correction = ctx.metadata.get("ego_correction")
        if not correction:
            return ""
        lines: list[str] = []
        prior = ctx.ego_result
        if prior:
            done = [t for t in prior.tools_executed if t.ok and t.side_effect]
            for t in done:
                lines.append(f"- {t.tool}({json.dumps(t.arguments, ensure_ascii=False)})")
        block = ""
        if lines:
            block += "# ACTIONS ALREADY EXECUTED (do NOT repeat these)\n" + "\n".join(lines) + "\n\n"
        reason = correction.get("reason")
        if reason:
            block += f"# Correction requested\n{reason}"
        return block.strip()

    # ── loop helpers ─────────────────────────────────────────────────

    @staticmethod
    def _name_args(tc: dict) -> tuple[str, dict]:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        return name, args

    @staticmethod
    def _sig(name: str, args: dict) -> str:
        digest = hashlib.md5(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return f"{name}|{digest}"

    @staticmethod
    def _feed_back(
        native: bool, messages: list[dict], raw_calls: list[dict],
        execs: list[ToolExecution], assistant_text: str,
    ) -> None:
        if not native:
            return  # fallback feeds back via _extend_prompt
        messages.append({"role": "assistant", "content": assistant_text or "", "tool_calls": raw_calls})
        for tc, ex in zip(raw_calls, execs):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": ex.result or ex.error or "",
            })

    @staticmethod
    def _extend_prompt(user_prompt: str, assistant_text: str, execs: list[ToolExecution]) -> str:
        chunk = ["", "[TOOL RESULTS]"]
        for ex in execs:
            chunk.append(f"{ex.tool}: {ex.result or ex.error or ''}")
        chunk.append("Continue with another <TOOL_CALL> if needed, otherwise give your final answer.")
        return user_prompt + "\n".join(chunk)
