---
name: create-story
description: >-
  Creates one or more user stories from docs/prd.md or from a new idea, each as
  its own markdown file under docs/stories/ with acceptance criteria and PRD links.
  Use when breaking down a PRD, adding a story, or writing backlog items.
---

# Create story

## Goal

Create **one file per story** under `docs/stories/` so stories can be validated, sized, implemented, and tested independently.

## Inputs

- **`docs/prd.md`** when stories come from the PRD (required for PRD-driven work).
- User description when creating a **standalone** story—still align with PRD if one exists; flag **Out-of-PRD** with rationale if not.

## Output

- **`docs/stories/<story-slug>.md`** — one or more files (user may ask for a batch).

## Story id convention

- Story ids: `S-001`, `S-002`, … (project-wide unique). When adding stories, scan existing files in `docs/stories/` for the highest `S-*` and continue.

## File template (`docs/stories/<slug>.md`)

```markdown
# S-XXX — [short title]

## Type
Feature | Bug | Chore | Spike

## Status
Draft

## Summary
One paragraph: value to the user.

## PRD traceability
- Related: FR-…, FR-…
- Wishes: W-… (if known)

## Acceptance criteria
1. Given … when … then …
2. …

## Out of scope
- …

## Notes
Dependencies, data, feature flags, migrations.

## Definition of done
- [ ] Code complete
- [ ] Tests updated or added
- [ ] Docs updated if user-visible
```

## Process

1. If from PRD: extract **one coherent user outcome** per story; **do not** bundle unrelated FRs.
2. Name file with **kebab-case slug** derived from title (e.g. `S-014-export-csv.md` or `export-csv.md` with id inside—**keep id inside file** either way). Prefer **`S-014-export-csv.md`** for sortability if numbers are stable.
3. Write **Given/When/Then** (or clear checklist ACs) so **`validate-story`** and **`test-story`** can map checks to lines.
4. List **dependencies** on other stories if order matters.
5. For multiple stories: print an index in chat: id, title, file path.

## Quality bar

- Each story has a **single** primary user outcome.
- Every AC references **observable** behavior or data.

## Handoff

Next steps: **`validate-story`** per file, then **`check-story-size`**, then **`implement-story`**.
