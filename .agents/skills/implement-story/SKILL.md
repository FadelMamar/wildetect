---
name: implement-story
description: >-
  Implements a single validated, right-sized story in code following docs/architecture.md
  and project conventions, keeping scope aligned to the story file. Use when coding
  a specific story after sizing, or when the user asks to implement a story.
---

# Implement story

## Goal

Deliver the **smallest correct change** that satisfies **`docs/stories/<slug>.md`** acceptance criteria, **consistent with `docs/architecture.md`**.

## Inputs

- **Story file:** `docs/stories/<slug>.md`
- **`docs/architecture.md`** (required for technical constraints and structure)
- **Codebase** — follow existing patterns (naming, layout, tooling).

## Rules

1. **Scope** — implement only what ACs require; defer nice-to-haves unless trivial and zero-risk.
2. **Architecture** — respect boundaries, stacks, and integration patterns in `docs/architecture.md`; if the story conflicts, **stop** and summarize the conflict instead of guessing.
3. **Quality** — match project style; no drive-by refactors unrelated to the story.
4. **Traceability** — in PR description or a short `## Implementation notes` in the story file (if the team allows edits), map ACs to key files or commits (optional but helpful).

## Process

1. Read the story ACs; list **tasks** as a short checklist in chat.
2. Read **`docs/architecture.md`** sections relevant to touched components.
3. Locate existing code paths; **extend** before inventing new abstractions.
4. Implement; run **format/lint/typecheck** if the project uses them (use project’s commands).
5. Update **story Status** to `Ready for test` or team equivalent when code is complete (if editing story files is acceptable).

## If blocked

- Missing decision, secret, or API access → document under **Notes** in the story or stop with a clear question.
- AC impossible under current architecture → propose **minimal** architecture amendment as a separate follow-up (do not silently change `docs/architecture.md` unless the user explicitly allows it for this task).

## Handoff

Next: **`test-story`** on the same slug.
