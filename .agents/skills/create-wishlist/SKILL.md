---
name: create-wishlist
description: >-
  Collects and structures stakeholder wishes through interactive elicitation, then
  writes or updates docs/wishlist.md. Use when starting discovery, capturing ideas
  before a PRD, or when the user asks for a wishlist, brainstorming, or requirements
  gathering from wishes.
---

# Create wishlist

## Goal

Produce a **single, reviewable wishlist** (`docs/wishlist.md`) that captures what people want, in their words first—then normalized for traceability into later PRD work.

## Outputs

- **Primary:** `docs/wishlist.md` (create or update)
- **Optional:** short summary in chat when done

## Conventions

- Use `docs/` for artifacts unless the project already standardizes another path; if so, follow the repo and mention the path in chat.
- Each wish gets a **stable id**: `W-001`, `W-002`, … (continue numbering when updating).

## Interactive workflow

1. **Frame the scope**  
   Ask what product, audience, and time horizon this wishlist is for. If unknown, propose assumptions and label them **(assumption)**.

2. **Elicit wishes** (round-based)  
   For each round:
   - Ask: *What should the system help users achieve? What should never happen? What would delight users?*  
   - Accept bullet answers; ask **one clarifying question** per vague item (who, when, measurable outcome).

3. **Normalize entries** (do not skip)  
   For each wish, capture:
   - **Id**
   - **Title** (short)
   - **User voice** (original intent, 1–3 sentences)
   - **Outcome** (observable result)
   - **Priority** (Must / Should / Could / Won’t for now) — user picks; if unsure, mark **TBD**
   - **Notes** (constraints, dependencies, open questions)

4. **Conflict and duplicate pass**  
   Surface duplicates or contradictions; ask the user to resolve or mark **CONFLICT** with options.

5. **Write `docs/wishlist.md`**  
   Use this structure:

```markdown
# Wishlist

## Meta
- Product/context:
- Last updated:
- Participants (roles, not PII):

## Wishes

### W-001 — [title]
- **User voice:** …
- **Outcome:** …
- **Priority:** …
- **Notes:** …

## Parking lot
- Ideas deferred or out of scope for this pass
```

6. **Read-back**  
   Give a 5-bullet summary and ask for **explicit confirmation** before treating the wishlist as “done for this step.”

## Quality bar

- Every **Must** has a clear **outcome** or is flagged **TBD** with a follow-up question.
- No silent merging of conflicting wishes.
