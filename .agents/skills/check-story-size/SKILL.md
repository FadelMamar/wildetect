---
name: check-story-size
description: >-
  Assesses whether one story is appropriately sized, proposes splits if it is too
  large, and updates execution order hints across related stories. Use before
  implementation when the user asks for sizing, splitting, ordering, or backlog
  refinement for a single story.
---

# Check story size

## Goal

Answer: **Can this story be implemented in one focused pass?** If not, **split** and **order** work safely.

## Input

- **Story file:** `docs/stories/<slug>.md`
- **`docs/architecture.md`** — complexity / touchpoints (layers, integrations).
- Other **`docs/stories/*.md`** (scan headings `S-*` and Dependencies) for ordering.

## Output

- Append a **Sizing** section to the same story file **or** write `docs/stories/<slug>-sizing.md` if the user prefers no inline mutation (ask once; default: **separate `-sizing.md`**).

## Default output file: `docs/stories/<slug>-sizing.md`

```markdown
# Sizing: S-XXX — [title]

## Verdict
Right-sized | Too large | Too small

## Rationale
- Touchpoints (modules/services/UI): …
- AC count and coupling: …
- Test surface: …

## Split proposal (if Too large)
### S-XXXa — …
- ACs moved: …
### S-XXXb — …
- ACs moved: …

## Execution order
1. S-… (reason: dependency / risk / data migration)
2. S-…
…

## Notes
Flags: migration, feature flag, API break, parallelizable work
```

## Sizing heuristics (guidance, not law)

- **Too large** if: >5–7 ACs **or** multiple subsystems **or** needs parallel UX + backend + data with independent releases **or** unclear single “done” demo.
- **Too small** if: single trivial AC that adds process overhead only—consider merging with adjacent story (ask user).
- Prefer splits by **vertical slice** (user-visible thin end-to-end) over layer-only splits unless architecture forces it.

## Ordering rules

1. **Dependencies first** (data model, auth, config).
2. **Risk reduction** early (unknown integrations, spikes as separate `Spike` stories if huge).
3. **Flag parallel** stories explicitly.

## Process

1. Read the story and `docs/architecture.md`; list concrete touchpoints.
2. Decide verdict; if Too large, produce **child story outlines** (titles + moved ACs)—do not fully author child files unless the user asks; otherwise **`create-story`** will formalize.
3. Build **execution order** across dependencies mentioned in the story and neighbors.
4. Summarize in chat: verdict, split count, first executable story id.

## Handoff

If Right-sized or splits agreed: **`implement-story`** for the **first** item in execution order.
