---
name: promote-notes
description: Promote selected claims from a source page's My Notes section into the relevant concept page's My Position section, officially entering your stance into the wiki. Use when the user says "promote notes", "提炼我的笔记", "把笔记升级成观点", or wants to formalize their reading notes on a paper into durable concept-level stance. Always interactive — asks the user to confirm each candidate claim.
---

# promote-notes

Promote selected notes from `wiki/sources/<source-slug>.md`'s `## My Notes` section into the relevant concept pages' `## My Position`.

Karpathy's framing says the wiki is agent-owned. `## My Notes` is a pragmatic extension: a user-owned co-located scratch pad on each source page. This skill is the **explicit** bridge from "scratch thought in My Notes" to "officially-held stance on a concept page". It never fires automatically — promotion is deliberate.

## Invocation

```
/promote-notes <source-slug>
```

## Flow

1. Read `wiki/sources/<source-slug>.md`. Extract the `## My Notes` section.
2. Identify **candidate promotions** — short standalone claims that relate to an existing concept page. Two signals:
   - Explicit wikilinks in the note (`[[concept-slug]]`).
   - Topical alignment: the note's subject maps to a concept slug/alias already in `wiki/concepts/`.
3. For each candidate, present to the user:
   ```
   Candidate: "<claim text>"
   Target concept: [[concept-slug]]
   Promote? (y/n/skip)
   ```
4. On user `y`:
   - Append to the target concept's `## My Position` section:
     ```
     (personal stance from [[<source-slug>]]) <claim>
     ```
   - Append to the concept's `## Evolution Log`:
     ```
     - YYYY-MM-DD promote-notes [[<source-slug>]]: established stance.
     ```
   - Update the concept page's `updated` and `last_reviewed` frontmatter to today.
5. The source's `## My Notes` is **not** modified. It remains the append-only scratch buffer; promoted claims stay there as the original trail.
6. Append to `wiki/log.md`: `YYYY-MM-DD HH:MM | promote-notes | <source-slug>`.

## Edge cases

- **No matching concept** for a candidate claim: suggest two paths to the user — (a) create a new concept page first (via a small personal-writing ingestion to `raw/personal/`), then re-run promote-notes; or (b) skip.
- **Claim conflicts with existing `## My Position`**: show the current content + the new candidate, ask the user how to resolve (replace / append / skip).
- **Claim references multiple concepts**: ask the user which target(s) to promote to.

## Non-goals

- Does not modify `## My Notes`. The note is appendable via `ADD-NOTE` only.
- Does not extract claims that aren't aligned with existing concepts — for wholly new stances, write a piece in `raw/personal/` and ingest it via `/ingest` (personal-writing flow).
- Does not run in `/batch-ingest`. Promotion is a per-paper deliberate action.
