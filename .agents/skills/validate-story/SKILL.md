---
name: validate-story
description: >-
  Validates a single story file for clarity, testability, scope, and alignment to
  the PRD, producing docs/stories/<slug>-validation.md and actionable fixes. Use
  when reviewing one story before sizing or implementation.
---

# Validate story

## Goal

Ensure **one story file** is ready for sizing and implementation **without hidden scope**.

## Input

- **Story file:** `docs/stories/<slug>.md` (user specifies which; if unclear, ask)
- **`docs/prd.md`** for traceability
- **`docs/wishlist.md`** (optional)

## Output

- **`docs/stories/<slug>-validation.md`**

## Checks

1. **Title and summary** — single outcome; fix vague verbs (“handle”, “support” → specific behavior).
2. **Acceptance criteria** — each AC is testable; no OR-soup; prefer Given/When/Then or explicit bullets the tester can tick.
3. **Traceability** — FR ids exist in PRD; flag **orphan** or **out-of-PRD** with rationale.
4. **Scope boundaries** — “Out of scope” present or explicitly “None”; no duplicate PRD content that belongs in architecture only.
5. **Dependencies** — if other stories must land first, ids named; cycles flagged.
6. **NFR touchpoints** — security, performance, a11y if implied by ACs.

## Report template (`docs/stories/<slug>-validation.md`)

```markdown
# Story validation: S-XXX — [title]

- Story file: docs/stories/<slug>.md
- Validated:
- Verdict: Pass | Pass with actions | Block

## Findings
### Blockers
### Should-fix
### Nice-to-have

## Suggested edits (concrete)
- …

## PRD alignment
- FRs: …
- Gaps: …
```

## Verdict rules

- **Block** if ACs are untestable or story bundles multiple unrelated outcomes.
- **Pass with actions** for missing out-of-scope, weak traceability, or unclear dependencies.
- **Pass** when no blockers.

## After validation

Recommend **`check-story-size`** next for the same story (or batch after all Pass).
