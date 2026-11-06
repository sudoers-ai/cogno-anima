# Architecture — who does what in the Cogno ecosystem

This page is the boundary map of the ecosystem: what `cogno-anima` does on its own, what it needs
from the other libraries, and what belongs to the application that runs it (the *host*). Each
row was read in the code that implements it, not inferred from a repository's name.

## Three layers

| Layer | What it is | Repositories |
| --- | --- | --- |
| **Public libraries** | Each one owns one concern and stays infra-agnostic: no database, queue or tenant model of its own unless that is its concern | `cogno-anima`, `cogno-synapse`, `cogno-homeo`, `cogno-engram`, `cogno-persona`, `cogno-praxis`, `cogno-cortex`, `cogno-mcp`, `cogno-gateway`, `cogno-vox`, `cogno-herald`, `cogno-meter`, `cogno-sync`, `cogno-aegis`, `cogno-observability` |
| **Reference orchestrator** | The public glue that runs one turn end to end over the anima's stages | `cogno-soma` |
| **Host (product layer)** | The application that assembles the libraries for a business: tenants, identities, RBAC, persistence, channels, billing, the dashboard | Your own host, or the reference one: `cogno-host` and its `cogno-ui` dashboard (both private) |

`cogno-anima` and `cogno-soma` are public libraries too; the orchestrator is listed apart because
it is the piece that turns the stages into a turn.

## Responsibilities, confirmed in the code

| Concern | Owner | Where to read it |
| --- | --- | --- |
| The cognitive stages: NOUMENO, NER, ID, EGO, SUPEREGO (scope guard, judge, voice), Drift | **cogno-anima** | `cogno_anima/stages/` |
| Running the stages in order and connecting them; the PII, scope and handoff gates; the hooks | **cogno-soma** | `cogno_soma/pipeline.py`: `Pipeline.run_turn` |
| The EGO ⇄ SUPEREGO correction loop: the judge's critique goes back to the executor and the loop runs until the judge approves or the budget is spent | **cogno-soma** runs it; the **host** sets the budget per turn | `cogno_soma/pipeline.py`; `cogno_soma/config.py`: `TurnConfig.max_corrections` (default 2) |
| State between turns | The stages are **stateless** (everything rides in `ctx.metadata`, e.g. `id_state`); **cogno-soma**'s `SessionRunner` carries it from turn to turn and exports it as a plain dict; the **host** persists that dict | `cogno_soma/session.py` |
| Which conversation a message belongs to (the thread key), and when a new one starts | **host** | in the reference host, `channels.py: session_id_for` (tenant + channel + sender) |
| What part of the history reaches the model verbatim | **cogno-soma** (`CONTEXT_WINDOW_EXCHANGES` = 6, `CONTEXT_WINDOW_GAP_SECONDS` = 4 h; both overridable) | `cogno_soma/session.py` |
| The prompts of the stages themselves (NOUMENO and NER templates, the judge's and the voice's scaffolding) | **cogno-anima** | `cogno_anima/prompt_templates/`, `cogno_anima/stages/superego.py` |
| A persona's declaration: scope, system, limits and voice prompts, identity, the modules it may use | **cogno-persona** (the store and selector); **cogno-praxis** ships ready-made personas with their verticals | `cogno-persona`; `cogno-praxis/cogno_praxis/<vertical>/prompts/` |
| Which persona answers this turn, and the tenant's own rules for it | **host** (it may use `cogno-persona`'s `PersonaSelector`) | — |
| Deciding which tool to call, with what arguments, and holding a write for confirmation | **cogno-anima** (the EGO) | `cogno_anima/stages/ego.py`; `docs/ACT_CONFIRM_READONLY.md` |
| The tool contract and the generic wrappers around it (one table over many sources, idempotency, id provenance, the pre-judge) | **cogno-anima** | `cogno_anima/tools/` |
| Tool sources | **cogno-cortex** (in-process skills), **cogno-mcp** (any MCP server as tools), **cogno-praxis** (business verticals served over MCP) | each repository |
| Composing the sources for a persona, executing the side effects, RBAC over tools | **host** | — |
| Memory: sessions, turns, long-term memories, the knowledge graph, documents | **cogno-engram** (stores and their Postgres/pgvector adapters) | `cogno-engram` |
| What is retrieved for a turn, and writing the turn back | **host** (calls `cogno-engram`; hands the result to `SessionRunner.run(memories=…)`) | — |
| LLM and embedding backends, the provider factory, failover | **cogno-synapse** (failover over **cogno-homeo**) | `cogno-synapse` |
| Channels (Telegram, WhatsApp, web), audio in and out | **cogno-gateway**, **cogno-vox** | each repository |
| E-mail, iCalendar, OTP; metering; locks and rate limits; security primitives; logs and traces | **cogno-herald**, **cogno-meter**, **cogno-sync**, **cogno-aegis**, **cogno-observability** | each repository |
| Tenants, identities and the org chart, plans and billing, the admin API, the dashboard, business rules that are not a vertical's domain | **host** | `cogno-host`, `cogno-ui` (private) |

Business rules live in two places on purpose: **domain** rules inside a vertical (`cogno-praxis`:
how the agenda books a slot), **orchestration** rules in the host (which persona a contact may
reach, which tool a role may use).

## What `cogno-anima` does alone, and what it needs

**Alone** (with only `cogno-synapse` for a model): every stage runs on a `PipelineContext` you
build. The perception half — NOUMENO → NER → ID — needs nothing else (see the Quickstart in the
README). The EGO needs a `ToolDispatcher`; the SUPEREGO needs the persona's prompts as strings.

**To run a full turn** you need the glue that orders the stages, applies the gates and runs the
correction loop. Three ways, from most to least supported:

1. **`cogno-soma`** — the shipped orchestrator (`Pipeline`, `TurnConfig`, `SessionRunner`). Its
   guide [«Build your own host»](https://github.com/sudoers-ai/cogno-soma/blob/main/docs/BUILD_YOUR_OWN_HOST.md)
   goes from zero to a running agent with the public libraries only.
2. **Your own glue** — `examples/host_min.py` wires the stages by hand; `docs/HOST_INTEGRATION.md`
   explains every seam.
3. **`cognobench/pipeline.py: ReferencePipeline`** — the benchmark's own glue. It is not shipped in
   the wheel and exists to drive CognoBench.

**To run an agent people talk to** you need a host: a channel (`cogno-gateway`), memory
(`cogno-engram`), tools (`cogno-cortex`, `cogno-mcp`, `cogno-praxis`), personas (`cogno-persona`)
and the persistence of the session state.

## One interaction, end to end

```
contact ──▶ cogno-gateway        receive, verify, normalise the message
              │
              ▼
            host                 who is this (identity, role), which conversation (thread key),
              │                  which persona, budget; load the session state; retrieve memories
              │                  (cogno-engram); build the tool dispatcher (cogno-cortex,
              │                  cogno-mcp, cogno-praxis, cogno_anima.tools wrappers)
              ▼
            cogno-soma           SessionRunner.run → Pipeline.run_turn:
              │                    NOUMENO → NER → ID → [PII gate] → [scope gate]
              │                    → EGO ⇄ SUPEREGO(judge)   (the correction loop)
              │                    → SUPEREGO(voice)
              │                  the stages are cogno-anima's; every model call goes through
              │                  cogno-synapse; tool calls go through the host's dispatcher
              ▼
            host                 act on the signals (stop_reason, needs_handoff), persist the
              │                  session state and the turn (cogno-engram), meter the cost
              ▼
contact ◀── cogno-gateway        format for the channel, send
```

## Evolution milestones (planned, not current behaviour)

The direction is a **lean host**: the public libraries should be enough for anyone to assemble a
working host, and the reference host should keep only what is a product decision. The milestones
below are plans. Each step moves code to a library that already exists (no new repository),
keeps behaviour byte for byte (checked by digest before and after) and lands on its own.

- **M1 — a minimal host as an acceptance test.** `cogno-soma` already has a guide and an offline
  example (`docs/BUILD_YOUR_OWN_HOST.md`, `examples/host_min.py`). The milestone is a host a
  person would deploy: one persona, one channel, memory, one tool and one cloud model, assembled
  only from the public libraries in about a hundred lines. It is written first, as a test that
  fails, and its failures say what is missing.
- **M2 — the generic turn safety nets.** Checks the reference host runs after the pipeline (a
  reply that claims an action no tool performed, totals no tool returned, a reply repeated word
  for word) move to `cogno-soma` and `cogno-anima`; the wording a business chooses stays in the
  host.
- **M3 — persona declarations and the generic channel service.** Everything that declares a
  persona (prompts, rules, the prompt used in group rooms) in `cogno-persona`; deduplication and
  per-channel session handling in `cogno-gateway`.
- **M4 — memory, e-mail, metering and traces.** The remaining adapters go to `cogno-engram`,
  `cogno-herald`, `cogno-meter` and `cogno-observability`.
- **M5 — the host keeps the product.** Tenants, billing and plans, the admin API, the dashboard,
  the administrator persona's access rules and operations.
- **M6 — the guide covers it.** `BUILD_YOUR_OWN_HOST.md` walks to the M1 host, and the M1 example
  runs green in CI.

Two specific moves are planned along the way:

- **Group rooms.** The mechanism (a room reads only data about nobody, an override can only
  close, which roles may send an individual notice from a room, a room never notifies another
  room) as a tool wrapper in `cogno_anima.tools`; the table of which tools touch people's data,
  and the roles allowed, stay the host's.
- **The administrator persona.** It is already a vertical in `cogno-praxis` (`admin`); its reads
  and its change-request records follow it there, behind ports the host implements. Who may reach
  it stays a host rule.

### Near term, in the public libraries

Planned for the next releases; each one is measured before and after it lands, because most of
them change what a model reads:

- **The judge** (`cogno-anima`) — accept a correct answer that does not show its arithmetic or
  cite every detail, and still reject what is invented. Today a correct execution is too often
  rejected and the turn ends in a cautious reply.
- **The scope guard reads the conversation** (`cogno-anima`, `cogno-soma`) — a short follow-up
  («and October?») is judged together with the subject of the conversation, not alone.
- **Drift across languages** (`cogno-anima`) — the epistemological drift compares a message with
  its English rewrite through an embedder that is not multilingual, so a change of language
  reads as a change of meaning. The measure, not the rewrite, is what needs fixing.
- **Group rooms in more channels** (`cogno-gateway`) — WhatsApp groups (Telegram already has an
  opt-in), and new channels on the same webhook family as WhatsApp Cloud (Instagram Direct,
  Messenger).
- **Releases** — every library on PyPI at the version its sources declare, `cogno-sync`
  included, and the install instructions moved from git to `pip install`.

Publishing today: every public library above is on PyPI except `cogno-sync`, which is installed
from git for now. Some releases are older than their sources; that is the last item above.
