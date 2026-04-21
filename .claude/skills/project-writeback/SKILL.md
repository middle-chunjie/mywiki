---
name: project-writeback
description: "Write back project discoveries to wiki, optionally archive the project. Use when the user says \"writeback\", \"写回wiki\", \"archive project\", \"归档项目\", or \"/project-writeback [slug] [--archive]\"."
---

# PROJECT-WRITEBACK — Write Back to Wiki + Archive

Review and commit project discoveries back to the wiki. Optionally archive the project afterwards. Cross-cutting rules live in `CLAUDE.md`.

## Steps

1. **Resolve slug**: If no slug, read `wiki/hot.md` `## Current Focus` for the active project.

2. **Read pending files**:
   - `projects/<slug>/wiki-contributions/pending-sources.md`
   - `projects/<slug>/wiki-contributions/pending-concepts.md`
   - `projects/<slug>/wiki-contributions/pending-synthesis.md`

3. **Present each pending item** to the user for approval:
   - Sources: title, URL/ID, reason for inclusion.
   - Concepts: proposed definition, key points, source links.
   - Synthesis: proposed cross-source insight.

4. **On approval**, chain to wiki operations:
   - **Sources**: `/save-paper` for each approved paper.
   - **Concepts**: create concept page from template, run `python scripts/wiki_ops.py cascade-update`.
   - **Synthesis**: `/reflect` with approved topic, or create `wiki/synthesis/<slug>.md` directly.

5. **Clear approved entries** from pending files. Mark rejected: `<!-- rejected YYYY-MM-DD: reason -->`.

6. **Log**: `python scripts/wiki_ops.py log-append project-writeback "<slug>: N sources, N concepts, N synthesis"`

7. **Report**: Summarize what was written back and what remains pending.

---

## Archive (when `--archive` flag or user says "归档项目", "archive project")

After writeback completes (or if no pending items):

8. **Archive**: `python scripts/project_ops.py archive --slug <slug>`

9. **Clear hot.md**: If `wiki/hot.md` `## Current Focus` references this project, reset to `_not set_`.

10. **Log**: `python scripts/wiki_ops.py log-append archive-project "<slug>"`

11. **Report**: Confirm archival, show final stage history from PROJECT.md.
