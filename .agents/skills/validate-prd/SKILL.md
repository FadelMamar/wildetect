---
name: validate-prd
description: >-
  Reviews docs/prd.md for consistency, testability, traceability, and delivery risk,
  producing a structured validation report and concrete edit suggestions. Use when
  the user asks to validate, review, or sign off on a PRD before stories or coding.
---

# Validate PRD

## Goal

**Reduce rework** by catching gaps and contradictions in `docs/prd.md` before stories and implementation.

## Input

- **`docs/prd.md`** (required)
- **`docs/wishlist.md`** (optional, for traceability checks)
- **`docs/architecture.md`** (optional, for feasibility / NFR alignment)

## Output

1. **`docs/prd-validation.md`** — structured report (create or replace)
2. Short chat summary: **Pass / Pass with actions / Block**

## Validation dimensions

Score each **Green / Amber / Red** with evidence (file headings, section names, or quoted ids—not huge paste).

| Dimension | What to check |
|-----------|----------------|
| **Completeness** | Goals, non-goals, scope, FRs, NFRs, acceptance, risks, open questions present |
| **Consistency** | Terms, priorities, no FR vs non-goal conflict |
| **Testability** | Each Must FR observable or measurable; vague verbs flagged |
| **Traceability** | FR ↔ wish/goal links; orphan FRs or orphan wishes |
| **Feasibility** | NFRs vs `docs/architecture.md`; call out mismatches |
| **Security & privacy** | Auth, data handling, threats at PRD level if applicable |

## Report template (`docs/prd-validation.md`)

```markdown
# PRD validation

- PRD path: docs/prd.md
- Validated:
- Verdict: Pass | Pass with actions | Block

## Summary scores
| Dimension | Status | Notes |
|-----------|--------|-------|

## Findings

### Blockers (must fix before story breakdown)
1. …

### Should-fix
1. …

### Nice-to-have
1. …

## Suggested edits
- …

## Questions for authors
1. …
```

## Rules

- **Block** if any Must FR is untestable, or if goals and non-goals contradict.
- **Pass with actions** for missing NFR rationale, weak risks, or traceability gaps.
- **Pass** only when there are **no blockers** and at most minor nice-to-haves.

## After validation

Tell the user the verdict and whether to run **`create-story`** next (after fixing blockers if any).
