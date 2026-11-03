---
name: create-testplan
description: >-
  Produces a structured test plan for one story at a time (unless the user
  explicitly asks for a wider scope), covering regression, UAT, integration,
  black-box, contract testing, smoke, performance, security, accessibility, and
  related types. Use when the user asks for a test plan, QA strategy, UAT plan,
  contract-test approach, regression outline, or how to validate a change before
  release. Every planned test maps to the story’s features and acceptance
  criteria (test-to-feature traceability).
---

# Create test plan

## Goal

Produce a **written test plan** that teams can execute or automate, aligned to **one primary story** (default), **`docs/prd.md`**, and **`docs/testing-guide.md`** (tooling and conventions) where it exists.

**Test-to-feature rule:** Nothing in the plan is “generic QA.” Every **test case, scenario, or check** must trace to a **concrete row** in the **story feature → acceptance criteria → tests** matrix (see template). If you cannot link a test to an AC, either **tie it to an explicit feature bullet** copied from the story summary or **drop it** from this plan.

## One story per plan (default)

- **Default:** plan for **exactly one** story file, `docs/stories/<slug>.md` — the user names the slug or `S-*` id; if unclear, ask which single story before drafting.
- **Wider scope** (release, sprint, epic, multiple stories): only when the user **explicitly** asks; if they list several stories without saying “combined plan,” create **separate** `docs/test-plans/<slug>.md` files **one per story** (or confirm they want one merged plan).
- **Traceability:** tie every **test** to **story features** (summary + ACs) and to **PRD FR ids** where the story lists them—**no orphan tests**.

## Inputs

- **Primary story:** `docs/stories/<slug>.md` — **one file** (required unless user explicitly requests a multi-story or release-level plan).
- **Wider scope (optional):** release version, milestone, or epic — only when explicitly requested; still list **primary story ids** per subsection if multiple.
- **`docs/prd.md`** — sections / FR ids linked from the story.
- **`docs/testing-guide.md`** — respect commands, directories, and quality bar; extend the plan where the guide is silent.
- **`docs/architecture.md`** — environments, integrations, NFRs, and risk hotspots.

## Output

- **Primary:** `docs/test-plans/<slug>.md`  
  Use a **kebab-case slug** derived from the **one** story (e.g. `S-014-export-csv` → `s-014-export-csv` or `story-export-csv`). For explicit release-wide plans only, use something like `release-2-3`.

Create **`docs/test-plans/`** if it does not exist.

## Test types catalog (pick what fits the scope)

Include only types that add value for this scope; list **N/A** with one-line rationale for consciously omitted types.

| Type | Typical purpose |
|------|------------------|
| **Unit** | Fast feedback on isolated logic; usually owned by devs (often overlaps **`test-story`**). |
| **Integration** | Interfaces between modules, services, DB, queues, third-party APIs. |
| **System / E2E** | Full user journeys through the deployed stack (or realistic test env). |
| **Regression** | Ensure existing behavior still works after changes; suite selection strategy. |
| **Smoke / sanity** | Minimal post-deploy checks before deeper testing. |
| **Black-box / functional** | Behavior vs requirements/ACs without knowledge of internals. |
| **White-box / structural** | Coverage of paths, branches, or code-level risks (when justified). |
| **UAT** | Business or user validation against agreed acceptance; roles, scenarios, sign-off. |
| **API testing** | HTTP/gRPC behavior: endpoints, status codes, auth, payloads, error shapes (can overlap integration). |
| **Contract testing** | **Consumer–provider contracts** (e.g. Pact), **OpenAPI/AsyncAPI/JSON Schema** conformance, protobuf/message compatibility, **consumer tests + provider verification** in CI, breaking-change policy vs versioned APIs. |
| **Performance / load / stress** | Latency, throughput, capacity under load (tie to NFRs). |
| **Security** | AuthZ/authN, OWASP-style checks, secrets, dependency scanning scope (not a full pentest unless requested). |
| **Accessibility** | WCAG-oriented checks, keyboard, screen reader critical paths. |
| **Exploratory / charter-based** | Time-boxed sessions with missions and notes template. |
| **Compatibility** | Browsers, OS, devices, or versions matrix. |
| **Data / migration** | Migration scripts, rollback, data integrity after changes. |
| **Resilience / failover** (if relevant) | Degraded modes, retries, idempotency. |

## Document template (`docs/test-plans/<slug>.md`)

```markdown
# Test plan: [title]

## Meta
- Primary story: S-… — `docs/stories/<slug>.md` (one story per document)
- Wider scope (if any): release / sprint / epic id
- Owner / author:
- Last updated:
- Related: PRD version, FR ids, build / release id

## Story scope (copy-in for matching)
- **Story title:** (from file)
- **Summary / user value:** (1–3 lines from story)
- **Features / capabilities under test** (derive from summary + ACs; short bullets, stable ids):
  - F1 — …
  - F2 — …

## Acceptance criteria (copy verbatim from story)
1. …
2. …

## Test-to-feature matrix (required)

Every planned test **must** appear here before detail sections. Use **TC-001** style ids for test cases.

| Test id | Feature id | AC # | Test intent (one line) | Test type (unit, integration, UAT, …) | Owner / automation |
|---------|------------|------|-------------------------|----------------------------------------|--------------------|
| TC-001 | F1 | 1 | … | … | … |
| TC-002 | F1 | 2 | … | … | … |

**Rules:**
- Each row: **Feature id** references a bullet from **Features / capabilities**; **AC #** is the story’s numbered AC.
- **No row** without **both** Feature id and AC # unless the story has a single implicit AC—then use **AC 1** and state that in Meta.
- **Regression / smoke** rows still name **which AC or feature** they protect (e.g. “regression guard for F2 / AC 3”).

## Objectives and out of scope
- …

## References
- docs/prd.md — sections / FR ids (must match story traceability)
- docs/stories/<slug>.md — source of features + ACs
- docs/testing-guide.md — …
- docs/architecture.md — environments, integrations

## Test types in scope
(For each selected type: objective, owner, tooling, where results are recorded.)

### [Type name]
- Objective:
- Scope / boundaries:
- Entry criteria:
- Exit criteria:
- Environments / data:
- **Tests in this section:** list **TC-ids** from the matrix only (no new ids here).

## Schedule and dependencies
- …

## Risks and mitigations
- …

## Open questions
- (AC or feature without TC — owner / decision)

## Deliverables
- e.g. executed checklist, automation PR links, UAT sign-off table — each item references **TC-ids**
```

## Process

1. **Lock one story** — identify `docs/stories/<slug>.md`; if the user gave multiple stories without asking for one combined plan, plan **the first** and say you will use a separate file per story, or ask which single story to start with.
2. **Confirm scope and stakeholders** — who approves UAT, who runs security/perf/contract verification (if any).
3. **Read** that story file, `docs/testing-guide.md`, and the PRD sections the story references.
4. **Extract features** — from the story **Summary** and **Acceptance criteria**, write **F1, F2, …** capability bullets (short, testable phrasing); each AC should map to at least one **F** (merge features if redundant).
5. **Build the test-to-feature matrix first** — add **TC-*** rows only when **Feature id + AC #** are filled; then select test types per row.
6. **Select test types** from the catalog; **include contract testing** whenever the story touches an API surface, events, or shared schema consumed by another team or service—otherwise mark **N/A** with rationale.
7. **Write** `docs/test-plans/<slug>.md`: complete the **matrix first**, then per-type sections listing **only TC-ids** from the matrix (no orphan cases); keep the matrix the source of truth for “what we test.”
8. **Summarize in chat** — story id, feature count, TC count, matrix coverage (any AC without a TC is a **gap** in **Open questions**).

## Quality bar

- **Every AC** from the story has **≥1 TC row** in the matrix, or **Open questions** documents the gap and owner.
- **Every TC** has **Feature id + AC #**; sections under test types cite **TC-ids** only.
- **Exit criteria** are observable (not “testing complete”).
- **Regression** strategy says what is **always** run vs **risk-based** subset (for **this** story’s blast radius), with TC-ids tied to **features/ACs**.
- **Contract testing** (when in scope): names **artifacts** (OpenAPI spec, Pact files, schema registry subjects), **who runs provider verification**, and **CI gate** (pass/fail on break)—each as **TC-*** rows in the matrix.
- **UAT** names **scenarios**, **participants**, and **sign-off** mechanism if applicable—mapped to **TC-ids** and **ACs**.

## Handoff

- **Implementation:** **`implement-story`** for **the same** story slug (plan informs what must be observable and testable).
- **Automated / dev tests:** **`test-story`** for **that** story; link this plan file in the story **Notes**.
- **Manual / UAT / perf / security runs:** execute outside the agent or with explicit human ownership as named in the plan.
