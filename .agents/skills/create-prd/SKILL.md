---
name: create-prd
description: >-
  Drafts or revises a PRD at docs/prd.md from a wishlist or stated goals, with
  clarifying questions, quality checks, and traceability to wishes. Use when
  the user wants a PRD, product requirements, a scope document, or formal
  requirements before implementation. Triggers on: create a prd, write prd for,
  plan this feature, requirements for, spec out.
---

# Create PRD

## Goal

Produce **`docs/prd.md`** that is **complete enough to derive stories and tests**, with requirements **traceable** to wishes or goals.

**Do not implement.** Stop after `docs/prd.md` is written.

## Inputs (in order)

1. `docs/wishlist.md` if it exists; else a short **Goals** section from the user.
2. `docs/architecture.md` **only** for feasibility notes—do not let it silently shrink user-facing scope (call out conflicts in the PRD).

## Output

- **Primary:** `docs/prd.md`

## Clarify first

Before drafting, ask **3–5** lettered questions only where the wishlist or prompt is ambiguous. Skip anything already answered.

Focus on:

- **Problem/Goal:** What problem does this solve?
- **Core behavior:** What are the key actions?
- **Boundaries:** What should it not do?
- **Success:** How do we know it is done?

Indent the options so the user can reply `1A, 2C`:

```
1. What is the primary goal?
   A. ...
   B. ...
   C. ...
   D. Other: [please specify]
```

If the user says to proceed, record the chosen default as **(assumption)** in the PRD.

## Required sections in `docs/prd.md`

Use clear headings; adapt subsections if the domain is tiny, but **do not omit** the checklist in “PRD quality gates.”

1. **Overview** — problem, target users, success definition (measurable where possible).
2. **Goals and non-goals** — bullet lists.
3. **Personas / actors** (lightweight is fine).
4. **Scope** — in-scope vs out-of-scope; **Won’t (this release)** explicit.
5. **User journeys** — 1–3 flows; link steps to requirement ids.
6. **Functional requirements** — table or list: `FR-001`, … each with **priority** and **rationale**.
7. **Non-functional requirements** — performance, security, accessibility, reliability, compliance (mark N/A with reason if not applicable).
8. **Acceptance criteria (release level)** — what “done” means for the release.
9. **Dependencies & assumptions** — systems, data, teams; label assumptions.
10. **Risks & mitigations** — at least top 3.
11. **Traceability** — matrix: `FR-*` → `W-*` (wishlist ids) or → **Goal** if no wishlist.
12. **Open questions** — numbered, owner **TBD** allowed.

Do not add `US-*` stories. **`create-story`** owns `S-*` files and their acceptance criteria.

## Write for a junior reader

The reader may be a junior developer or an agent:

- Be explicit and unambiguous. Explain jargon.
- Keep requirement ids stable and numbered.
- Use concrete examples.

Bad: "Works correctly."

Good: "The delete control opens a confirmation dialog before the interview is removed."

Apply that bar to functional requirements. Do not add a browser-verification line.

## PRD quality gates (run before finalizing)

Self-check and **fix or flag** in the Open questions section:

- [ ] Every **Must** FR has testable acceptance wording.
- [ ] No duplicate FRs under different ids; merge or cross-reference.
- [ ] Non-goals do not contradict Must FRs.
- [ ] Terms (nouns used in FRs) are **glossary-consistent** (add a tiny **Glossary** if needed).
- [ ] Security/privacy touched if there is PII, auth, or payments (even “not applicable” with rationale).
- [ ] Each FR maps to **at least one** wish or stated goal, or is marked **New** with justification.

## Process

1. Read inputs; list **ambiguities**.
2. Ask the clarifying questions above. Label unresolved defaults **(assumption)** in the PRD.
3. Draft `docs/prd.md` with ids **stable** (`FR-001`…); do not renumber existing ids when updating—append new ones.
4. Run **PRD quality gates**; iterate once for obvious gaps.
5. Summarize in chat: scope, count of FRs, top risks, open questions count.

## Handoff

State that the next typical steps are **`validate-prd`** then **`create-story`**.
