---
name: test-story
description: >-
  Verifies a implemented story using existing tests or by adding targeted tests,
  runs them, and fixes failures related to the story scope. Reads docs/testing-guide.md
  before executing. Use after implement-story or when the user asks to test, verify,
  or harden a story.
---

# Test story

## Goal

Provide **evidence** that **`docs/stories/<slug>.md`** acceptance criteria are met, using the project’s real test stack.

## Inputs

- **`docs/testing-guide.md`** — **read fully before any test mapping, commands, or new tests**; follow its paths, commands, and conventions unless the user explicitly overrides them for this session.
- **Story file:** `docs/stories/<slug>.md`
- **`docs/architecture.md`** — optional; use when tests touch boundaries, NFRs, or integration points described there.
- **`docs/test-plans/<slug>.md`** — optional but recommended; when present, use its **test-to-feature matrix** (**TC-*** ↔ **Feature** ↔ **AC**) as the checklist: every automated or manual run should **map results to TC-ids and story ACs** in `docs/stories/<slug>-test.md`. Do not add tests that are not tied to this story’s **features/ACs** unless the user expands scope.
- **Implementation** from the same branch/session (or user points to PR/commit).
- **Project test commands** — must match `docs/testing-guide.md` when that file defines them; otherwise discover from README, package scripts, or CI.

## Process

1. **Read `docs/testing-guide.md`**  
   Confirm test locations, commands, naming, and mocking rules. If the file is empty or placeholder-only, infer from the repo and **note gaps** in the test report.

2. **Map feature / AC → tests (test-to-feature)**  
   For each **AC** (and **feature F-*** if a test plan defines them), list **existing** test file(s) and case names that cover it, or mark **gap**. If a **`docs/test-plans/`** file exists for this story, align rows with **TC-ids** from its matrix.

3. **Prefer extending existing suites**  
   Add the smallest tests that fail without the fix and pass with it, consistent with the testing guide.

4. **Run tests**  
   Use the commands from **`docs/testing-guide.md`** when specified; run **scoped** tests when possible for speed.

5. **Fix failures**  
   - If failure is **in story scope**: fix implementation or test as appropriate.  
   - If **pre-existing**: note it; do not expand scope unless the user asks.

6. **Report**  
   Write **`docs/stories/<slug>-test.md`** (create or update):

```markdown
# Test report: S-XXX — [title]

## Commands run
- …

## AC / feature coverage
| AC # | Feature (F#) | Plan TC-id (if any) | Evidence (test name / file) | Status |
|------|--------------|---------------------|-----------------------------|--------|

## Gaps filled
- …

## Failures fixed
- …

## Remaining risks / follow-ups
- …
```

7. **Story status**  
   If all ACs covered and tests green: set story **Status** to `Done` (when editing the story is allowed).

## Rules

- Do not delete tests unless they are invalid; **update** for intentional behavior change.
- Avoid flaky patterns (timers, real network) per **`docs/testing-guide.md`** and project norms.
- Keep new tests **readable** and **aligned to AC wording**.

## Handoff

If gaps remain: return to **`implement-story`** or split a new story for test-only debt—user decision.
